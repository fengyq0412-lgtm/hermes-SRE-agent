# Hermes SRE Agent

面向 AI 与 Web 服务的、**证据优先、审批受控**的 SRE 智能体。

```text
观察 → 诊断 → 制定计划 → 人工审批 → 修复 → 验证
```

Hermes 的目标不是抢先猜出答案，而是把每条结论和真实采集结果关联起来；任何会改变状态的操作，都不能绕过人工审批。本仓库目前包含一条完整、可复现的慢接口故障诊断链路，也能安全读取本机的只读状态，适合作为可演示、可审查的作品集项目。

## 一分钟演示

```bash
# 使用 Conda 建立环境（首次执行）
conda env create -f environment.yml
conda activate hermes-sre-agent

# requirements.txt 会以可编辑模式安装本项目，获得 hermes 命令
pip install -r requirements.txt

# 无需模型密钥的确定性演示
hermes "legal-agent 最近很慢，帮我排查原因" --mode rules --scenario api_slow
```

## Agent 模式：像 Codex 一样从终端自然语言开始

Hermes 默认进入 Agent 模式。你只需启动终端并描述问题；模型会在只读工具白名单中自主选择健康检查、日志、Docker、GPU、Git 变更或代码差异审查工具。模型不能获得任意 Shell、写文件、重启服务或删除资源的权限。

Hermes 使用 OpenAI Chat Completions 兼容协议，因此可以对接任意提供该协议的国内或本地模型服务。将模板复制为根目录 `.env`，然后填写供应商地址、密钥和模型名：

```bash
cp .env.example .env
# 编辑 .env 后，Hermes 会在启动时自动读取它

# 单次自然语言诊断
hermes "昨天部署后 legal-agent 接口从 800ms 变成 4 秒，帮我排查"

# 连续对话式终端
hermes --interactive
```

连接本地兼容服务时，可使用 `http://localhost:端口/v1` 或 `http://127.0.0.1:端口/v1`。远程模型地址必须使用 HTTPS，避免 API 密钥以明文传输。

## 本地控制台：项目选择与代码审查

Hermes 提供本地代码问答工作台：在页面选择项目文件夹，输入一句「帮我检查这个项目有哪些 bug」，即可看到模型的中文回答和工具检查进度。可以继续追问，例如「第二个问题在什么条件下触发？」。

```bash
# 首次使用将 .env.example 复制为 .env，填写模型地址、密钥和模型名
# 从 Hermes 项目目录启动
hermes-web
```

在浏览器打开 `http://127.0.0.1:8765`：

1. 点击左侧「打开项目」，在文件夹选择器中浏览本机目录，也可输入路径后点击「前往」。
2. 进入具体项目文件夹，点击「选择此项目」。无需设置 `HERMES_PROJECT_ROOTS`，旧配置也不再影响网页启动。
3. 输入审查问题，按 Enter 发送。Agent 会选择列目录、搜索代码、读取源码等工具，页面展示检查进度和最终模型回答。
4. 在同一项目继续提问会带上最近三轮问答；「新对话」清除当前项目的对话上下文。切换项目不会混用对话。

项目列表保存在启动目录的 `.hermes/projects.json`，重启后保留；对话与审查结果暂存在内存中，刷新页面或重启服务后不会恢复。网页问答不往被审查项目写报告。选择没有 Git 历史的项目也能直接审查源码。

服务只监听本机回环地址，并验证页面会话令牌。密钥仅在 Python 服务端使用；`.env` 已被 Git 忽略。**点击发送会将相关源码片段发送至你在 `.env` 配置的模型服务**；工具默认排除隐藏文件、`.env`、凭据文件、依赖目录、生成文件和符号链接，但不等同于完整的源码秘密检测。

源码审查单次最多 12 轮，最多列出 500 个文件，每次最多读取 200 行，超出范围会提示截断。因此回答是基于实际读取范围的检查，不代表已覆盖整个项目，也没有运行测试。当前页面显示逐步工具进度，最终回答在模型返回后展示，暂非逐 token 输出。

## 确定性演示模式

未配置模型时仍可运行可复现的规则引擎演示：

```bash
# 单次诊断：运行后输出一份报告并退出
hermes "legal-agent 最近很慢，帮我排查原因" --mode rules --scenario api_slow

# 交互模式：可以连续输入多个故障问题
hermes --interactive --mode rules --scenario api_slow
```

交互模式中输入故障描述即可诊断；输入 `exit` 或 `quit` 退出。每次提问都会创建独立事件编号，并分别保存报告和 trajectory，避免多个事故的证据混在一起。

默认演示场景稳定复现了一个可信的 AI 推理事故：

| 信号 | 证据 |
| --- | --- |
| 用户症状 | API 延迟：800 ms → 4200 ms |
| 健康检查 | HTTP 200，但 `latency_ms=4200` |
| GPU | 77,210 / 81,920 MB 显存，94% 利用率 |
| 日志 | 请求排队与 GPU 显存压力告警 |
| Git / 配置 | 最近发布将 `batch_size: 16 → 64` |

Hermes 会得出有依据的根因和最小回退方案，然后明确停在“等待用户批准”。它不会修改配置，也不会重启容器。

每次运行都会在 `artifacts/` 中产生：

- `*-report.md`：面向人的中文事件报告
- `*-trajectory.json`：工具调用、结果、状态转换、诊断与操作记录，可用于后续评测

可用 `hermes --list-tools` 查看工具白名单；`--scenario api_500` 可查看“证据不足时安全停止”的行为。

## Docker 故障实验环境

PRD 中的演示环境已落地为一个轻量 Docker Compose 服务。它不是生产环境模拟器，而是用于稳定复现 API 慢、API 500 和恢复正常三种情况；状态文件通过 volume 挂载到容器，注入故障不需要重建镜像。

```bash
# 终端 A：启动演示服务
docker compose -f demo/docker-compose.yml up --build

# 终端 B：注入慢接口故障
python demo/inject_fault.py --scenario api_slow

# 确认服务已变慢（约 4.2 秒返回）
curl -i http://localhost:8000/health

# 恢复正常状态
python demo/inject_fault.py --scenario normal
```

故障注入仅改写 `demo/runtime/state.json`，不会执行容器删除、重启、停止进程或任何宿主机变更命令。可用场景为 `normal`、`api_slow`、`api_500`。

Docker 环境启动后，可让 Hermes 读取真实的 HTTP 与 Docker 日志（其余主机采集项若不可用，会在报告中如实说明）：

```bash
hermes "legal-agent 为什么变慢了？" \
  --mode rules \
  --backend live \
  --service legal-agent \
  --container legal-agent \
  --health-url http://localhost:8000/health
```

## 本机只读采集

除了演示场景，还可以通过固定的、无 Shell 拼接的命令读取本机状态：

```bash
hermes "检查本机 legal-agent 是否正常" --mode rules --backend live --service legal-agent
```

本机模式只会尝试采集系统负载、NVIDIA GPU、Docker、HTTP 健康检查和 Git 差异。环境未安装 Docker、没有 NVIDIA GPU、服务不可访问等情况会如实写入报告，且诊断会因关键证据缺失停止；绝不会把失败伪装成成功。

> 本机模式仍然只有只读能力。`restart_service` 与 `edit_config` 只是已登记的危险工具，调用会在工具层被拒绝。

### 本机采集的前置条件

这些不是 `pip` 依赖，而是被诊断对象所在机器应具备的运行能力：

| 能力 | 是否必需 | 说明 |
| --- | --- | --- |
| HTTP 服务与正确的健康检查 URL | 基础诊断必需 | 例如 `http://localhost:8000/health`；`404` 是服务异常证据，不是 Python 安装问题。 |
| Docker Desktop / Docker daemon | 仅诊断容器时需要 | 启动 Docker Desktop 后，用 `docker ps` 确认可连接。 |
| NVIDIA 驱动与 `nvidia-smi` | 仅诊断 GPU 推理时需要 | Apple Silicon / 普通 Mac 没有 NVIDIA GPU，这是正常情况。 |
| 有至少一次提交的 Git 仓库 | 仅分析发布变更时需要 | 首次初始化、尚未提交的仓库没有可比较的历史版本。 |

`wps` 是桌面应用名称，不是 Hermes V0.1 支持的 HTTP / Docker 服务对象。若要诊断本机 WPS，应另行实现 macOS 应用进程、崩溃日志和性能采集适配器；它不应被当成 `http://wps:8000/health` 一类服务。

## 架构

```text
用户故障描述
    ↓
LLMSREAgent：模型选择下一项只读工具
    ↓
ControlledTools：程序校验工具与参数，不提供任意 Shell
    ↓
ScenarioBackend（可复现演示） / LiveReadOnlyBackend（本机只读）
    ↓
证据轨迹 → 模型假设 → 程序验证证据引用 → 报告
    ↓
修复计划（需要审批；当前不会执行）
```

工具边界就是安全边界。`system_metrics`、`gpu_metrics`、`docker_list`、`docker_logs`、`service_health`、`git_recent_changes`、`config_diff` 与 `git_code_review` 是只读工具。`restart_service`、`edit_config` 被定义为危险操作，并且在进入任何后端之前即被阻断。

## 安全设计

- 没有通用的 `run_shell` 工具。
- 每次只读观察及其失败都会被记录到 trajectory。
- 只有多条独立证据互相印证，才会给出高置信度根因。
- 缺少或冲突的证据会产生“未确定”报告，绝不编造结论。
- 当前版本没有任何修复执行路径。

### 代码审查是否需要沙箱？

需要分层处理：**只读差异审查**不需要运行代码，但仍应在只读工作区中进行，限制读取范围、返回长度，并把 Git diff / 日志视为不可信输入，防止提示注入。Hermes 的 `git_code_review` 正是这一层：仅调用禁用外部 diff 与分页器的受限 `git diff`，最多返回 20,000 字符，绝不执行仓库内容。

**运行测试、构建、复现故障或执行脚本**则必须进入独立沙箱：临时 worktree 或容器、无默认网络、无生产密钥、CPU / 内存 / 时长限制，并且需要用户明确授权。它与 SRE 修复执行器应是两个独立权限域。当前版本尚未提供这类执行沙箱，因此页面不会运行任何项目代码。

## 质量检查

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试覆盖：端到端慢接口诊断、模型自主工具选择、伪造证据拦截、危险重启工具拦截、代码审查输入限制和证据不足时的安全停止。

## 路线图

- **V0.1（当前）：** Agent 工具规划、可复现诊断、报告、轨迹与本机只读采集。
- **V0.2：** 一次性审批令牌、白名单配置变更、服务重启与变更后验证。
- **V0.3：** Docker / Linux / HTTP / Git 的更完整实时适配器与评测用例集。
- **V0.4：** 多智能体分诊与事件记忆。
- **V0.5：** Kubernetes、Prometheus、Grafana 集成。

## 项目结构

```text
src/hermes_sre_agent/   Agent 运行时、兼容模型客户端、受控工具、采集后端与报告输出
demo/scenarios/         可移植的故障证据样例
tests/                  工作流与安全保障测试
```
