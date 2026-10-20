// 前端 diff 回归：使用 Node 内置测试工具，不新增运行依赖。
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../src/hermes_sre_agent/web_assets/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
// 简单 DOM 替身只允许文本渲染；任何 innerHTML 写入都会使安全测试失败。
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.text = ''; this.className = ''; this.parent = null; }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
  set innerHTML(_value) { throw new Error('禁止将源码作为 HTML 插入'); }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(...nodes) { for (const node of nodes) { node.remove?.();node.parent = this;this.children.push(node); } }
  replaceChildren(...nodes) { for (const child of this.children)child.parent = null;this.children = [];this.text = '';this.append(...nodes); }
  remove() { if (!this.parent)return;const index = this.parent.children.indexOf(this);if (index >= 0)this.parent.children.splice(index, 1);this.parent = null; }
  replaceWith(node) { if (!this.parent)return;const parent = this.parent,index = parent.children.indexOf(this);node.remove?.();parent.children[index] = node;node.parent = parent;this.parent = null; }
  querySelectorAll(selector) { const found = [];for (const child of this.children) {
    if (selector.startsWith('.') ? child.className.split(' ').includes(selector.slice(1)) : child.tag === selector)found.push(child);
    found.push(...child.querySelectorAll(selector));
  }return found; }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
const context = vm.createContext({ document: { createElement: tag => new Element(tag),
  createTextNode: value => { const node = new Element('#text');node.textContent = value;return node; } } });
vm.runInContext(script.slice(script.indexOf('function element('), script.indexOf('async function api(')) +
                script.slice(script.indexOf('function usageText('), script.indexOf('function statusMeta(')) +
                script.slice(script.indexOf('function statusMeta('), script.indexOf('function freshChat(')) +
                script.slice(script.indexOf('function freshChat('), script.indexOf('async function loadConversationList(')) +
                script.slice(script.indexOf('function parseDiff('), script.indexOf('function renderChat(')), context);
const parse = text => JSON.parse(JSON.stringify(context.parseDiff(text)));
const walk = node => [node, ...node.children.flatMap(walk)];

test('整个页面脚本语法有效，审批与历史记录共用高亮组件', () => {
  assert.doesNotThrow(() => new vm.Script(script));
  assert.equal((script.match(/detail\.append\(renderDiff\(file\)\)/g) || []).length, 2);
  assert.ok(!script.includes("element('pre','',file.diff)"));
});

test('任务状态给出明确下一步，运行中和待审批状态可区分', () => {
  assert.deepEqual(Array.from(context.statusMeta('running')), ['正在分析', 'Agent 正在读取证据、形成结论或验证候选修改。']);
  assert.deepEqual(Array.from(context.statusMeta('awaiting_review')), ['等待你的审核', '候选补丁与测试结果已准备好，批准后才会写回项目。']);
  assert.deepEqual(Array.from(context.statusMeta('unknown')), ['正在提交', '正在创建本轮任务。']);
  assert.ok(html.includes('id="jump-latest"'));
  assert.ok(script.includes("aria-live','polite'"));
});

test('审批与历史方案区分测试通过和改动被验证', () => {
  assert.ok(script.includes('实际测试入口：'));
  assert.ok(script.includes('即使测试退出码为 0，也可能未覆盖业务改动'));
  assert.ok(script.includes('旧版任务即使显示测试通过，也不能据此确认修复有效'));
  assert.ok(script.includes('历史模型报告（缺少改动覆盖证据，默认折叠）'));
  assert.ok(script.includes('修复前对照测试未执行；原缺陷是否被复现仍待验证'));
  assert.ok(script.includes('turn.proposal.can_apply?\'测试已通过且执行到业务改动行'));
});

test('历史会话使用可删除的列表，输入操作区保持稳定对齐', () => {
  assert.ok(html.includes('id="conversations"'));
  assert.ok(!html.includes('id="conversation-select"'));
  assert.ok(html.includes('grid-template-columns:minmax(0,1fr) auto'));
  assert.ok(script.includes("method=body===undefined?'GET':'POST'"));
  assert.ok(script.includes("'DELETE'"));
  assert.ok(script.includes('window.confirm'));
});

test('项目列表可移除，明确保留源码与历史', () => {
  assert.ok(html.includes('project-row'));
  assert.ok(script.includes("'/api/projects/'+encodeURIComponent(project.id)"));
  assert.ok(script.includes("'DELETE'"));
  assert.ok(script.includes('项目文件和历史会话不会删除'));
  assert.ok(script.includes('state.chats.delete(project.id)'));
});

test('旧服务返回 HTML 时会提示重启，而不是暴露 JSON 解析错误', () => {
  assert.ok(script.includes("method==='DELETE'&&r.status===501"));
  assert.ok(script.includes('当前 hermes-web 尚未重启'));
  assert.ok(script.includes('服务返回了非 JSON 响应'));
});

test('运行中可中止，同一会话可继续任务', () => {
  assert.ok(script.includes("'中止任务'"));
  assert.ok(script.includes("'继续任务'"));
  assert.ok(script.includes("'/cancel'"));
  assert.ok(script.includes("parent_id:c.parent,mode"));
});

test('网页运行诊断显式收集目标并将观测源留在只读任务', () => {
  for (const id of ['task-mode', 'diagnostic-service', 'diagnostic-container', 'diagnostic-health',
                    'diagnostic-kube-context', 'diagnostic-prometheus-url', 'diagnostic-grafana-token-env']) {
    assert.ok(html.includes(`id="${id}"`));
  }
  const inputs = {
    'diagnostic-service': 'hermes-web', 'diagnostic-container': 'hermes-web',
    'diagnostic-health': 'http://127.0.0.1:8765/health', 'diagnostic-strategy': 'triage',
    'diagnostic-kube-context': 'prod', 'diagnostic-kube-namespace': 'hermes',
    'diagnostic-kube-selector': 'app=hermes-web', 'diagnostic-prometheus-url': '',
    'diagnostic-prometheus-job': '', 'diagnostic-grafana-url': '', 'diagnostic-grafana-token-env': '',
  };
  context.$ = id => ({value: inputs[id]});
  const target = JSON.parse(JSON.stringify(context.diagnosticTarget()));
  assert.deepEqual(target, {service: 'hermes-web', container: 'hermes-web',
    health_url: 'http://127.0.0.1:8765/health', strategy: 'triage',
    observability: {kube_context: 'prod', kube_namespace: 'hermes', kube_selector: 'app=hermes-web'}});
  assert.ok(script.includes('const mode=requestMode(c)'));
  assert.ok(script.includes('diagnostic_target:target'));
});

test('后台任务不锁定历史查看，但其他会话保持只读', () => {
  const current = {busy: true, active: {conversation_id: 'running'}, conversationId: 'older',
    conversations: [{id: 'running', status: 'running'}]};
  assert.equal(context.projectLocked(current), true);
  assert.equal(context.historyReadOnly(current), true);
  current.conversationId = 'running';
  assert.equal(context.historyReadOnly(current), false);
  current.active = null;current.conversations[0].status = 'awaiting_review';
  assert.equal(context.historyReadOnly(current), false);
  current.conversationId = 'older';
  assert.equal(context.historyReadOnly(current), true);
  assert.ok(script.includes('const locked=c.loading;const readOnly=projectLocked(c)'));
  assert.ok(script.includes('select.disabled=locked'));
  assert.ok(script.includes('remove.disabled=locked||readOnly'));
});

test('历史列表内容未变时保留按钮节点，运行中仍可切换查看', () => {
  vm.runInContext(script.slice(script.indexOf('function renderConversationList()'), script.indexOf('async function deleteConversation(')), context);
  const list = new Element('div');
  const c = {busy: true, loading: false, active: {conversation_id: 'running'}, conversationId: 'older',
    conversations: [{id: 'running', title: '运行中', turn_count: 1, status: 'running'},
      {id: 'older', title: '旧会话', turn_count: 2, status: 'completed'}]};
  context.state = {selected: {id: 'project'}, renderedHistoryKey: null};
  context.chat = () => c;
  context.$ = () => list;
  context.renderConversationList();
  const firstRow = list.children[0], select = list.children[1].children[0];
  assert.equal(select.disabled, false);
  assert.equal(list.children[1].children[1].disabled, true);
  context.renderConversationList();
  assert.strictEqual(list.children[0], firstRow);
  c.conversations[0].status = 'completed';c.busy = false;
  context.renderConversationList();
  assert.notStrictEqual(list.children[0], firstRow);
});

test('发送前明确区分多语言审查和仅 Python 可修改范围', () => {
  assert.ok(html.includes('id="code-scope"'));
  for (const mode of ['code', 'review', 'repair', 'diagnose'])assert.ok(html.includes(`value="${mode}"`));
  const ids = new Map();const get = id => {if (!ids.has(id))ids.set(id, new Element('div'));return ids.get(id);};
  context.$ = get;
  context.setTaskMode('diagnose');
  assert.equal(get('code-scope').hidden, true);
  context.setTaskMode('code');
  assert.equal(get('code-scope').hidden, false);
  assert.match(get('code-scope').textContent, /只能为项目内 Python 源码和测试生成补丁/);
  assert.equal(context.requestMode({}), 'auto');
  context.setTaskMode('review');
  assert.match(get('code-scope').textContent, /不会生成补丁/);
  assert.equal(context.requestMode({}), 'review');
  context.setTaskMode('repair');
  assert.match(get('code-scope').textContent, /仅支持项目内 Python 源码和测试/);
  assert.equal(context.requestMode({}), 'repair');
  assert.equal(context.requestMode({resumeMode: 'review'}), 'review');
});

test('轮询仅更新活动任务节点，保留已展开的详情节点', () => {
  const oldDetail = {className: '', textContent: '查看沙箱测试输出\n结果', open: true,
    querySelector: () => ({textContent: '查看沙箱测试输出'})};
  const nextDetail = {className: '', textContent: oldDetail.textContent, open: false,
    querySelector: () => ({textContent: '查看沙箱测试输出'}), replaceWith(node) { this.replacement = node; }};
  context.preserveDetails({querySelectorAll: () => [oldDetail]}, {querySelectorAll: () => [nextDetail]});
  assert.strictEqual(nextDetail.replacement, oldDetail);

  const title = {textContent: ''}, description = {textContent: ''};
  const status = {className: '', querySelector: selector => selector === 'strong' ? title : description};
  const progress = {textContent: ''}, usage = {hidden: true, textContent: ''}, totalUsage = {textContent: ''};
  const box = {querySelector: selector => ({'.task-state': status, '.live-progress': progress,
    '.live-usage': usage})[selector] || null};
  const c = {conversationId: 'running'};
  context.state = {selected: {id: 'project'}, renderedTurns: new Map([['job', {box}]]), usage: null};
  context.chat = () => c;
  context.$ = () => totalUsage;
  context.renderChat = () => { throw new Error('轮询不应重建整段对话'); };
  context.refreshVisibleTask('project', {id: 'job', conversation_id: 'running', status: 'running',
    events: [{message: '正在采集'}], usage: {requests: 0}});
  assert.strictEqual(context.state.renderedTurns.get('job').box, box);
  assert.equal(title.textContent, '正在分析');
  assert.equal(description.textContent, '正在采集');
  assert.equal(progress.textContent, '正在采集');
  assert.ok(script.includes('if(active){c.active=visible||turn;if(visible)refreshVisibleTask'));
});

test('运行结果更新后，其他消息和测试输出详情节点保持原样', () => {
  vm.runInContext(script.slice(script.indexOf('function renderChat('), script.indexOf('function fileLabel(')), context);
  const ids = new Map();const get = id => {if (!ids.has(id))ids.set(id, new Element('div'));return ids.get(id);};
  context.$ = get;context.document.querySelector = () => get('footnote');
  context.renderConversationList = () => {};
  context.renderMarkdown = (node, text) => { node.textContent = text; };
  context.requestAnimationFrame = () => {};
  context.updateJumpLatest = () => {};
  const turn = {id: 'job', projectId: 'project', conversation_id: 'conv', question: '检查服务',
    status: 'running', mode: 'review', events: [{message: '读取证据'}], trace: [],
    steps: [{tool: 'run_tests', result: {output: 'PASS'}}]};
  const older = {id: 'older', projectId: 'project', conversation_id: 'conv', question: '上一次审查',
    status: 'completed', mode: 'review', answer: '旧结论', events: [], trace: []};
  const c = {turns: [older, turn], conversationId: 'conv', conversations: [{id: 'conv', status: 'running'}],
    busy: true, hydrated: true, loading: false};
  context.chat = () => c;
  context.state = {selected: {id: 'project', name: 'Hermes'}, chats: new Map([['project', c]]),
    renderedKey: null, renderedTurns: new Map(), usage: null, approvalDismissed: new Set(), approvalTurnId: null};
  get('task-mode').value = 'code';get('approval-dialog').open = false;
  context.renderChat();
  const first = context.state.renderedTurns.get('job');
  const olderBox = context.state.renderedTurns.get('older').box;
  const output = first.box.querySelectorAll('details').find(detail =>
    detail.querySelector('summary')?.textContent === '查看沙箱测试输出');
  assert.ok(output);output.open = true;
  const trace = first.box.querySelector('.live-trace');trace.open = true;
  turn.events.push({message: '继续分析'});
  turn.trace.push({seq: 1, summary: '读取指标', time: 'now', stage: '取证', event: 'result'});
  context.refreshVisibleTask('project', turn);
  assert.strictEqual(context.state.renderedTurns.get('job').box, first.box);
  assert.strictEqual(first.box.querySelector('.live-trace'), trace);
  assert.equal(trace.open, true);
  assert.equal(trace.querySelector('.trace-list').children.length, 1);
  turn.pollError = '暂时无法读取任务状态';
  context.refreshVisibleTask('project', turn);
  assert.strictEqual(context.state.renderedTurns.get('job').box, first.box);
  assert.equal(first.box.querySelector('.live-poll-error').textContent, turn.pollError);
  turn.status = 'completed';turn.answer = '已完成';c.busy = false;c.conversations[0].status = 'completed';
  context.renderChat();
  const next = context.state.renderedTurns.get('job');
  assert.strictEqual(context.state.renderedTurns.get('older').box, olderBox);
  assert.strictEqual(next.user, first.user);
  assert.strictEqual(next.box.querySelectorAll('details').find(detail =>
    detail.querySelector('summary')?.textContent === '查看沙箱测试输出'), output);
  assert.equal(output.open, true);
});

test('查看其他历史会话时后台轮询继续，且不会覆盖当前会话父节点', async () => {
  vm.runInContext(script.slice(script.indexOf('async function monitor('), script.indexOf("$('chat-form').onsubmit")), context);
  const active = {id: 'job', conversation_id: 'active', projectId: 'project', status: 'running'};
  const c = {turns: [{id: 'older-job', status: 'completed'}], active, busy: true,
    conversationId: 'older', parent: 'older-job', conversations: [
      {id: 'active', status: 'running'}, {id: 'older', status: 'completed'}]};
  const scheduled = [];let polls = 0;
  context.state = {selected: {id: 'project'}, pollingJobs: new Set(), cancelledJobs: new Set(), usage: null};
  context.chat = () => c;
  context.api = async () => ({id: 'job', conversation_id: 'active', status: ++polls === 1 ? 'running' : 'completed',
    answer: '已完成', session_usage: null});
  context.setTimeout = callback => scheduled.push(callback);
  context.refreshVisibleTask = () => { throw new Error('不应更新正在查看的其他会话'); };
  context.renderChat = () => {};
  context.loadConversationList = async () => {};

  await context.monitor('project', active);
  assert.equal(scheduled.length, 1);
  assert.equal(c.conversations[0].status, 'running');
  await scheduled.shift()();
  assert.equal(c.active, null);
  assert.equal(c.busy, false);
  assert.equal(c.parent, 'older-job');
  assert.equal(c.turns[0].id, 'older-job');
  assert.equal(c.conversations[0].status, 'completed');
  assert.equal(context.state.pollingJobs.size, 0);
});

test('轮询暂时失败时只更新当前任务错误，不重建对话', async () => {
  vm.runInContext(script.slice(script.indexOf('async function monitor('), script.indexOf("$('chat-form').onsubmit")), context);
  const turn = {id: 'job', conversation_id: 'active', projectId: 'project', status: 'running'};
  const c = {turns: [turn], active: turn, busy: true, conversationId: 'active', conversations: []};
  const scheduled = [];let refreshed = 0;
  context.state = {selected: {id: 'project'}, pollingJobs: new Set(), cancelledJobs: new Set()};
  context.chat = () => c;
  context.api = async () => { throw new Error('网络断开'); };
  context.setTimeout = callback => scheduled.push(callback);
  context.refreshVisibleTask = () => { refreshed++; };
  context.renderChat = () => { throw new Error('轮询失败不应重建对话'); };
  await context.monitor('project', turn);
  assert.equal(refreshed, 1);
  assert.match(turn.pollError, /网络断开/);
  assert.equal(scheduled.length, 1);
});

test('多块修改的原行号、新行号和增删统计正确', () => {
  const diff = parse('--- a/main.py\n+++ b/main.py\n@@ -10,3 +10,4 @@\n same\n-old\n+new\n+extra\n tail\n@@ -30 +31 @@\n-before\n+after\n');
  assert.equal(diff.removed, 2);
  assert.equal(diff.added, 3);
  assert.deepEqual(diff.rows.filter(r => r.sign).map(r => [r.kind, r.old, r.new]), [
    ['context', 10, 10], ['removed', 11, ''], ['added', '', 11], ['added', '', 12],
    ['context', 12, 13], ['removed', 30, ''], ['added', '', 31],
  ]);
});

test('新建、纯删除及缺少末尾换行标记不会生成错误行号', () => {
  const created = parse('--- /dev/null\n+++ b/test.py\n@@ -0,0 +1,2 @@\n+first\n+second\n');
  assert.deepEqual(created.rows.filter(r => r.kind === 'added').map(r => [r.old, r.new]), [['', 1], ['', 2]]);
  const removed = parse('@@ -8,2 +7,0 @@\n-old\n-last\n\\ No newline at end of file\n');
  assert.equal(removed.removed, 2);
  assert.equal(removed.added, 0);
  assert.equal(removed.rows.at(-1).kind, 'meta');
  assert.deepEqual(parse('').rows, []);
});

test('CRLF、制表符及与文件头相似的源码保持正确显示', () => {
  const diff = parse('--- a/main.py\r\n+++ b/main.py\r\n@@ -1 +1 @@\r\n--- literal\r\n+++\tliteral\r\n');
  assert.equal(diff.removed, 1);
  assert.equal(diff.added, 1);
  assert.equal(diff.rows.at(-2).text, '-- literal');
  assert.equal(diff.rows.at(-1).text, '++\tliteral');
});

test('红绿行类名、表头、无障碍标签及源码 HTML 转义安全', () => {
  const payload = '<img src=x onerror="alert(1)">';
  const root = context.renderDiff({path: 'main.py', diff: '@@ -1 +1 @@\n-old\n+' + payload + '\n'});
  const nodes = walk(root);
  assert.equal(nodes.filter(n => n.className === 'diff-removed' && n.tag === 'tr').length, 1);
  assert.equal(nodes.filter(n => n.className === 'diff-added' && n.tag === 'tr').length, 1);
  assert.equal(nodes.filter(n => n.tag === 'th').length, 4);
  assert.equal(nodes.filter(n => n.tag === 'img').length, 0);
  assert.ok(nodes.some(n => n.className === 'diff-code' && n.textContent === payload));
  assert.ok(nodes.some(n => n.attributes['aria-label'] === 'main.py 代码差异'));
  assert.ok(nodes.some(n => n.className === 'diff-scroll' && n.tabIndex === 0));
});
