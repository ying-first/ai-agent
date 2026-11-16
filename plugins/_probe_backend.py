"""探针脚本：测试 duckduckgo_search 各 backend 的可用性。

注意：本文件不是工具插件，不注册任何 @tool。
必须在 __main__ 保护下运行，避免被插件加载器 import 时触发网络探测。
使用方式：python plugins/_probe_backend.py
"""


def probe_backends():
    from duckduckgo_search import DDGS

    for b in ["auto", "html", "lite", "ecosia", "mojeek", "brave", "wikipedia", "yahoo", "google"]:
        try:
            cli = DDGS(timeout=10)
            rows = list(cli.text("python dataclass", max_results=2, backend=b) or [])
            print("[OK]", b, "->", len(rows), "rows")
            if rows:
                print("     sample:", str(rows[0].get("title"))[:60], "|",
                      str(rows[0].get("href"))[:60])
        except Exception as e:
            print("[FAIL]", b, "->", type(e).__name__, str(e)[:100])
    return []


if __name__ == "__main__":
    probe_backends()
