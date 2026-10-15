"""在容器临时目录运行测试；原始源码快照保持只读。"""
import os
import shutil
import sys
import json
import threading

import pytest

shutil.copytree("/source", "/tmp/project")
os.chdir("/tmp/project")
sys.path[:0] = ["/tmp/project", "/tmp/project/src"]

targets = json.loads(os.environ.get("HERMES_COVERAGE_TARGETS", "[]"))
tracked = {os.path.abspath(os.path.join("/tmp/project", name)): name for name in targets}
executed = {name: set() for name in targets}


def trace_source(frame, event, _arg):
    if event == "call":
        return trace_source if os.path.abspath(frame.f_code.co_filename) in tracked else None
    if event == "line":
        name = tracked.get(os.path.abspath(frame.f_code.co_filename))
        if name is not None:
            executed[name].add(frame.f_lineno)
    return trace_source


class TestResults:
    """没有实际通过的测试时不允许以成功状态进入写回审批。"""
    passed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1


results = TestResults()
# 仍禁用自动加载其他插件，仅显式开启镜像内固定版本的异步测试插件。
if tracked:
    sys.settrace(trace_source)
    threading.settrace(trace_source)
try:
    status = pytest.main(["-q", "-p", "no:cacheprovider", "-p", "pytest_asyncio.plugin",
                          "-o", "asyncio_default_fixture_loop_scope=function", *sys.argv[1:]], plugins=[results])
finally:
    if tracked:
        sys.settrace(None)
        threading.settrace(None)
print("HERMES_COVERAGE_JSON=" + json.dumps({name: sorted(lines) for name, lines in executed.items()},
                                            ensure_ascii=False, separators=(",", ":")))
if status == 0 and results.passed == 0:
    print("未执行通过任何测试，不能据此确认补丁已验证。")
    status = 5
raise SystemExit(status)
