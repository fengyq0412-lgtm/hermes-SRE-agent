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
                script.slice(script.indexOf('function parseDiff('), script.indexOf('function renderChat(')), context);
const parse = text => JSON.parse(JSON.stringify(context.parseDiff(text)));
const walk = node => [node, ...node.children.flatMap(walk)];

test('整个页面脚本语法有效，审批与历史记录共用高亮组件', () => {
  assert.doesNotThrow(() => new vm.Script(script));
  assert.equal((script.match(/detail\.append\(renderDiff\(file\)\)/g) || []).length, 2);
  assert.ok(!script.includes("element('pre','',file.diff)"));
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
