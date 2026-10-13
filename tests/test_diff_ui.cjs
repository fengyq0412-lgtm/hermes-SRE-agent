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
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.text = ''; }
  set textContent(value) { this.text = String(value); }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
  set innerHTML(_value) { throw new Error('禁止将源码作为 HTML 插入'); }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(...nodes) { this.children.push(...nodes); }
}
const context = vm.createContext({ document: { createElement: tag => new Element(tag) } });
vm.runInContext(script.slice(script.indexOf('function element('), script.indexOf('async function api(')) +
                script.slice(script.indexOf('function statusMeta('), script.indexOf('function freshChat(')) +
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
  assert.ok(script.includes("mode=isDiagnosis?'diagnose'"));
  assert.ok(script.includes('diagnostic_target:target'));
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
