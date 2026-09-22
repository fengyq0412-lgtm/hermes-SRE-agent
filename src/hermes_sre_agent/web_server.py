"""本地项目选择与代码问答服务；密钥只保存在服务端。"""

import copy
import hashlib
import json
import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .code_review import CodeReviewAgent
from .env_loader import load_dotenv
from .model_client import ModelConfig, ModelConfigurationError, ModelRequestError, OpenAICompatibleClient


MAX_REQUEST_BYTES = 16_384


class ProjectRegistry:
    """项目由用户通过页面显式添加，记住选择结果以便下次启动使用。"""

    def __init__(self, roots=None, storage=None):
        self.storage = Path(storage) if storage else None
        self.lock = threading.RLock()
        self.projects = {}
        entries = []
        if self.storage and self.storage.is_file():
            try:
                saved = json.loads(self.storage.read_text(encoding="utf-8"))
                entries = saved if isinstance(saved, list) else []
            except (ValueError, OSError):
                entries = []
        if roots is not None:
            entries.extend(roots.split(os.pathsep))
        for entry in entries:
            if isinstance(entry, str) and entry.strip() and Path(entry).expanduser().is_dir():
                self.add(entry, save=False)

    def add(self, directory, save=True):
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("请选择项目文件夹。")
        path = Path(directory).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("目录不存在，请选择有效的本机文件夹。")
        path = path.resolve()
        project_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        with self.lock:
            self.projects[project_id] = path
            if save and self.storage:
                self.storage.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.storage.with_suffix(".tmp")
                temporary.write_text(json.dumps([str(p) for p in self.projects.values()], ensure_ascii=False), encoding="utf-8")
                temporary.replace(self.storage)
        return {"id": project_id, "name": path.name, "path": str(path)}

    def list_public(self):
        with self.lock:
            return [{"id": key, "name": path.name, "path": str(path)}
                    for key, path in sorted(self.projects.items(), key=lambda item: item[1].name.lower())]

    def resolve(self, project_id):
        with self.lock:
            if project_id not in self.projects:
                raise KeyError("请先在页面选择项目。")
            return self.projects[project_id]


def browse_directories(directory=None):
    """目录选择器只返回文件夹名称；源码要等用户选择项目后才可读取。"""
    path = Path(directory).expanduser() if directory else Path.home()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("文件夹不存在。")
    path = path.resolve()
    children = []
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.name.startswith(".") and entry.is_dir(follow_symlinks=False):
                children.append({"name": entry.name, "path": entry.path})
                if len(children) >= 500:
                    break
    return {"path": str(path), "parent": str(path.parent), "directories": sorted(children, key=lambda item: item["name"].lower())}


class ReviewJobs:
    """后台问答任务。对话上下文来自同项目的已完成任务，不信任客户端伪造历史。"""

    def __init__(self, registry, client_factory=None):
        self.registry = registry
        self.jobs = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hermes-review")
        self.client_factory = client_factory or (lambda: OpenAICompatibleClient(ModelConfig.from_environment()))

    def create(self, project_id, question, parent_id=None):
        project = self.registry.resolve(project_id)
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
            raise ValueError("请输入 1 到 4000 个字符的审查问题。")
        with self.lock:
            if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= 2:
                raise ValueError("已有两项审查正在运行，请等待完成。")
            if len(self.jobs) >= 100:
                raise ValueError("本次会话已达到 100 条任务上限，请重启控制台。")
            history = []
            if parent_id:
                parent = self.jobs.get(parent_id)
                if not parent or parent["project_id"] != project_id or parent["status"] != "completed":
                    raise ValueError("追问必须对应同一项目的已完成回答。")
                history = parent["history"] + [{"role": "user", "content": parent["question"]},
                                                {"role": "assistant", "content": parent["answer"][:16000]}]
                history = history[-6:]
            job_id = uuid4().hex
            self.jobs[job_id] = {"id": job_id, "project_id": project_id, "question": question.strip(),
                                 "status": "queued", "events": [], "history": history}
        self.executor.submit(self._run, job_id, project, question.strip(), history)
        return {"id": job_id}

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("问答任务不存在。")
            return copy.deepcopy({k: v for k, v in self.jobs[job_id].items() if k != "history"})

    def _update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def _event(self, job_id, event):
        with self.lock:
            self.jobs[job_id]["events"].append(event)

    def _run(self, job_id, project, question, history):
        self._update(job_id, status="running")
        try:
            result = CodeReviewAgent(project, self.client_factory(), lambda e: self._event(job_id, e)).run(question, history)
            self._update(job_id, status="completed", answer=result["answer"], steps=result["steps"])
        except ModelConfigurationError:
            self._update(job_id, status="failed", message="请在 Hermes 的 .env 中填写有效的 HERMES_BASE_URL、HERMES_API_KEY 和 HERMES_MODEL，然后重启服务。")
        except ModelRequestError as exc:
            self._update(job_id, status="failed", message=str(exc))
        except Exception:
            self._update(job_id, status="failed", message="审查过程中发生错误，请检查项目是否可读后重试。")


class HermesRequestHandler(BaseHTTPRequestHandler):
    registry: ProjectRegistry
    jobs: ReviewJobs
    session_token = secrets.token_urlsafe(32)
    asset_path = Path(__file__).parent / "web_assets" / "index.html"

    def _valid_request(self, api=False):
        # 回环监听不等于防跨站：同时验证 Host、Origin 和每次启动生成的会话令牌。
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin != f"http://{host}"):
            self._send_json(403, {"error": "只接受本机控制台请求。"})
            return False
        if api and not secrets.compare_digest(self.headers.get("X-Hermes-Token", ""), self.session_token):
            self._send_json(403, {"error": "页面会话已过期，请刷新页面。"})
            return False
        return True

    def do_GET(self):
        parsed = urlparse(self.path)
        if not self._valid_request(api=parsed.path.startswith("/api/")):
            return
        try:
            if parsed.path == "/":
                page = self.asset_path.read_text(encoding="utf-8").replace("__HERMES_TOKEN__", self.session_token)
                self._send_bytes(200, page.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/projects":
                self._send_json(200, {"projects": self.registry.list_public()})
            elif parsed.path == "/api/directories":
                self._send_json(200, browse_directories(parse_qs(parsed.query).get("path", [None])[0]))
            elif parsed.path.startswith("/api/reviews/"):
                self._send_json(200, self.jobs.get(parsed.path.rsplit("/", 1)[-1]))
            else:
                self._send_json(404, {"error": "接口不存在。"})
        except (OSError, ValueError, KeyError):
            self._send_json(400, {"error": "目录或任务不可访问，请重新选择。"})

    def do_POST(self):
        if not self._valid_request(api=True):
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("请求必须是 JSON。")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("请求大小无效。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象。")
            if self.path == "/api/projects":
                self._send_json(201, self.registry.add(payload.get("path")))
            elif self.path == "/api/reviews":
                if not isinstance(payload.get("project_id"), str) or (payload.get("parent_id") is not None and not isinstance(payload["parent_id"], str)):
                    raise ValueError("项目或会话标识格式错误。")
                self._send_json(202, self.jobs.create(payload["project_id"], payload.get("question"), payload.get("parent_id")))
            else:
                self._send_json(404, {"error": "接口不存在。"})
        except (ValueError, KeyError) as exc:
            self._send_json(400, {"error": str(exc)})
        except OSError:
            self._send_json(400, {"error": "无法访问或保存此项目，请检查目录权限。"})

    def _send_json(self, status, body):
        self._send_bytes(status, json.dumps(body, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        """不将用户问题或项目路径写入默认 HTTP 日志。"""


def main():
    load_dotenv()
    port = int(os.environ.get("HERMES_WEB_PORT", "8765"))
    HermesRequestHandler.registry = ProjectRegistry(storage=Path.cwd() / ".hermes" / "projects.json")
    HermesRequestHandler.jobs = ReviewJobs(HermesRequestHandler.registry)
    server = ThreadingHTTPServer(("127.0.0.1", port), HermesRequestHandler)
    print(f"Hermes 控制台：http://127.0.0.1:{port}", flush=True)
    print("在页面点击「打开项目」选择文件夹；模型配置读取当前目录 .env。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n控制台已停止。")
    finally:
        server.server_close()
        HermesRequestHandler.jobs.executor.shutdown(wait=False, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
