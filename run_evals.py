"""run_evals.py — 命令行评测入口。

用法：
  python run_evals.py                  # 运行全部任务并与 baseline 比较
  python run_evals.py --update-baseline
  python run_evals.py --tag repo --limit 3 --json
退出码：0=全部通过且无回归；1=有失败/回归；2=运行错误。
"""
import argparse
import json
import sys

import eval_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="只跑指定 tag")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--task-file", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    res = eval_harness.run_evals(
        tags=[args.tag] if args.tag else None,
        limit=args.limit or None,
        update_baseline=args.update_baseline,
        task_file=args.task_file or None,
    )
    if args.json_out:
        try:
            d = os.path.dirname(os.path.abspath(args.json_out))
            os.makedirs(d, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[run_evals] 写 json-out 失败: {e}", file=sys.stderr)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(eval_harness.format_report(res))
    if res.get("error"):
        return 2
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
