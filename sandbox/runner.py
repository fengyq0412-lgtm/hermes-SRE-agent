"""在容器临时目录运行测试；原始源码快照保持只读。"""
import os
import shutil
import sys

import pytest

shutil.copytree("/source", "/tmp/project")
os.chdir("/tmp/project")
sys.path[:0] = ["/tmp/project", "/tmp/project/src"]


class TestResults:
    """没有实际通过的测试时不允许以成功状态进入写回审批。"""
    passed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1


results = TestResults()
status = pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[1:]], plugins=[results])
if status == 0 and results.passed == 0:
    print("未执行通过任何测试，不能据此确认补丁已验证。")
    status = 5
raise SystemExit(status)
