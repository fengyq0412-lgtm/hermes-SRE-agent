"""人工运维变更命令；模型工具目录不包含这些操作。"""

import argparse
import getpass
import json
import sys
from pathlib import Path

from .remediation import ChangeManager, RemediationError, _json_object


def main(argv=None, read_input=input, read_token=getpass.getpass):
    parser = argparse.ArgumentParser(description="Hermes V0.2 受控运维变更")
    parser.add_argument("--policy", type=Path, required=True, help="由运维人员维护的变更白名单 JSON")
    parser.add_argument("--state-dir", type=Path, default=Path(".hermes/remediation"), help="私有计划与备份目录")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="生成绑定当前配置与容器身份的变更计划")
    plan.add_argument("--service", required=True)
    plan.add_argument("--changes", required=True, help='JSON 字段值，例如 {"batch_size":16}')
    approve = commands.add_parser("approve", help="人工确认并签发十分钟有效的一次性令牌")
    approve.add_argument("plan_id")
    apply = commands.add_parser("apply", help="消费令牌、写配置、重启并验证")
    apply.add_argument("plan_id")
    show = commands.add_parser("show", help="查看计划、执行状态与验证记录")
    show.add_argument("plan_id")
    args = parser.parse_args(argv)
    try:
        manager = ChangeManager(args.policy, args.state_dir)
        if args.command == "plan":
            result = manager.prepare(args.service, _json_object(args.changes))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("\n下一步：hermes-change --policy <同一策略文件> approve " + result["id"])
        elif args.command == "approve":
            planned = manager.show(args.plan_id)
            print(json.dumps(planned["plan"], ensure_ascii=False, indent=2))
            confirmation = read_input("确认上述变更，请输入目标服务名：").strip()
            token = manager.approve(args.plan_id, confirmation)
            print("一次性令牌（仅显示本次，十分钟有效）：" + token)
        elif args.command == "apply":
            token = read_token("请输入一次性审批令牌：")
            result = manager.apply(args.plan_id, token.strip())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "verified" else 1
        else:
            print(json.dumps(manager.show(args.plan_id), ensure_ascii=False, indent=2))
    except (RemediationError, OSError, ValueError) as exc:
        print(f"运维变更未完成：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
