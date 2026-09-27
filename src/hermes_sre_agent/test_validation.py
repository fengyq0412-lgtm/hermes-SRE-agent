"""测试生成所需的源码事实与验证失败分类；不导入或执行项目代码。"""

import ast
import difflib
import re
from pathlib import Path

from .code_review import SourceTools


TEST_RULES = (
    "测试必须基于当前源码的实际签名、字段名、返回值和异常分支，不得猜测。"
    "async def 必须 await；普通 pytest 测试可用 asyncio.run 调用，使用 asyncio 前需要导入。"
    "异步依赖使用 AsyncMock，同步依赖使用 Mock。若 mock 了上层包装器，不要假定下层函数仍被执行；"
    "优先保留真实包装器，仅 mock 外部 I/O 边界，并核对被 await 的参数。"
    "不得复制被测实现到测试中、用恒真断言、删断言、跳过用例来制造通过。"
    "项目导入会连接数据库或服务时，可在导入前隔离明确的外部适配器模块，仍需测试真实目标函数。"
    "遇到 ModuleNotFoundError 应先沿 traceback 修复导入边界，不要只改断言重复运行。"
    "静态依赖表提供定义位置和同步/异步类型，必要时定点读取，勿逐页遍历整个适配器。"
    "不能用假的 FastAPI/Pydantic 或返回预期值的被测函数替身掩盖环境问题。"
    "不能改业务源码来迎合测试；需求或依赖确实不足时报告具体阻碍。"
)


def import_isolation_plan(root, changes):
    """给出可审核的导入隔离参考，不执行项目、不自动改测试、不自动选择业务边界。"""
    tools = SourceTools(root)
    plans = []
    for path in changes:
        if path.startswith("tests/") or Path(path).name.startswith("test_") or path.endswith("_test.py"):
            continue
        try:
            tree = ast.parse(tools.resolve(path).read_text(encoding="utf-8"))
        except (ValueError, SyntaxError, UnicodeError, OSError):
            continue
        groups = {}
        # 首版仅对顶层模块给出可执行参考；包内相对导入由模型结合实际结构判断。
        if len(Path(path).parts) != 1:
            continue
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                groups.setdefault(node.module, []).extend(alias.name for alias in node.names)
        dependencies = []
        lines = ["import importlib, sys, logging", "from types import ModuleType",
                 "from unittest.mock import Mock, AsyncMock, patch", "import pytest", "",
                 "@pytest.fixture", "def subject():", "    adapters = {}"]
        for module, names in list(groups.items())[:30]:
            dependency = module.replace(".", "/") + ".py"
            if "." in module or dependency in changes:
                continue
            try:
                target = tools.resolve(dependency)
                if target.stat().st_size > 256000:
                    continue
                definitions = {n.name: n for n in ast.parse(target.read_text(encoding="utf-8")).body
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            except (ValueError, SyntaxError, UnicodeError, OSError):
                continue
            # 含类、常量或动态导出时无法安全给出统一替身；尤其不伪造框架模型。
            if not names or any(name not in definitions for name in names):
                continue
            facts = []
            lines.append(f"    adapter = ModuleType({module!r})")
            for name in sorted(set(names)):
                definition = definitions[name]
                asynchronous = isinstance(definition, ast.AsyncFunctionDef)
                facts.append({"name": name, "async": asynchronous, "line": definition.lineno,
                              "signature": ast.unparse(definition.args)[:1000]})
                lines.append(f"    adapter.{name} = {'AsyncMock' if asynchronous else 'Mock'}()")
            dependencies.append({"path": dependency, "symbols": facts})
            lines.append(f"    adapters[{module!r}] = adapter")
        if not dependencies:
            continue
        context = "    with patch.dict(sys.modules, adapters)"
        has_file_handler = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                               and isinstance(n.func.value, ast.Name) and n.func.value.id == "logging"
                               and n.func.attr == "FileHandler" for n in ast.walk(tree))
        if has_file_handler:
            context += ', patch("logging.FileHandler", return_value=logging.NullHandler())'
        lines.extend([context + ":", f"        sys.modules.pop({Path(path).stem!r}, None)",
                      f"        yield importlib.import_module({Path(path).stem!r})"])
        plans.append({"target": path, "local_dependencies": dependencies,
                      "scope": "仅为单元测试参考：先确认这些未修改模块是外部 I/O 适配器，才使用替身。"
                               "真实目标、响应模型和异步包装器必须保留；不证明适配器集成行为。"
                               "读取测试后将适用代码加入可审核的补丁，不能仅描述方案。",
                      "fixture_example": "\n".join(lines) + "\n"})
    return plans


def source_contracts(root, changes):
    """按实际 diff 定位变更函数及本地辅助定义，给测试 Agent 提供直接证据。"""
    tools = SourceTools(root)
    records, remaining = [], 18000
    for path, (before, after) in changes.items():
        if path.startswith("tests/") or path.split("/")[-1].startswith("test_") or path.endswith("_test.py"):
            continue
        source = tools.resolve(path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            records.append({"path": path, "syntax_error": True})
            continue
        changed = set()
        for tag, _, _, start, end in difflib.SequenceMatcher(None, (before or b"").decode().splitlines(),
                                                           after.decode().splitlines()).get_opcodes():
            if tag != "equal":
                changed.update(range(start + 1, max(start + 2, end + 1)))
        definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        targets = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and any(n.lineno <= line <= n.end_lineno for line in changed)]
        # 跟进一层本地调用，避免不知道超时包装器或响应类的实际行为。
        needed = {node.func.id for target in targets for node in ast.walk(target)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        selected = targets + [node for name, node in definitions.items() if name in needed and node not in targets]
        excerpts = []
        for node in selected[:8]:
            snippet = ast.get_source_segment(source, node) or ""
            snippet = snippet[:min(5000, remaining)]
            if not snippet:
                break
            remaining -= len(snippet)
            excerpts.append({"name": node.name, "async": isinstance(node, ast.AsyncFunctionDef),
                             "start_line": node.lineno, "end_line": node.end_lineno, "source": snippet})
        imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        records.append({"path": path, "imports": imports[:30], "definitions": excerpts})
        if remaining <= 0:
            break
    return records


def validation_details(result):
    """将错误归类，日志仅持久化类别与模块名，原始输出保留在测试面板。"""
    status, code = result.get("status"), result.get("exit_code")
    missing = sorted(set(re.findall(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]", result.get("output", ""))))[:10]
    if status == "not_run":
        kind = "tests_not_generated"
    elif status == "finished" and code == 0:
        kind = "passed"
    elif missing:
        kind = "dependency_missing"
    elif status != "finished":
        kind = "sandbox_" + str(status)
    else:
        kind = {1: "test_failure", 2: "collection_error", 5: "no_tests"}.get(code, "test_runner_error")
    summary = {
        "passed": "本次测试已通过。",
        "dependency_missing": "测试导入依赖失败：" + "、".join(missing) +
                              "。请检查导入链；外部 I/O 适配器可在单元测试中隔离，真实业务所需依赖应装入沙箱镜像。",
        "collection_error": "测试收集阶段失败，测试用例尚未正常执行。",
        "test_failure": "测试断言或执行失败，请查看具体失败分支。",
        "no_tests": "没有实际通过的测试，不能据此确认补丁有效。",
        "tests_not_generated": "尚未生成测试文件，本次未启动 Docker 容器；已有业务补丁仍需验证。",
    }.get(kind, "沙箱未完成有效验证，请查看执行输出。")
    return {"failure_kind": kind, "missing_modules": missing, "summary": summary}
