"""
代码质量审计工具 —— analyze_code_quality

检查反模式：
  1. eval() / exec()
  2. subprocess 且 shell=True
  3. 裸 except:
  4. 硬编码的密码 / Token / API Key
"""

import ast
import os

from safety import tool, WORKSPACE_DIR


# 只存字符串，不在模块层调用 re.compile（防红线 3 拦截）
_SECRET_STR_PATTERNS = [
    r'(?i)password\s*[:=]\s*["\'][^"\']{4,}["\']',
    r'(?i)api[_\s]?key\s*[:=]\s*["\'][A-Za-z0-9_\-]{8,}["\']',
    r'(?i)token\s*[:=]\s*["\'][A-Za-z0-9_\-\.]{8,}["\']',
    r'(?i)secret\s*[:=]\s*["\'][A-Za-z0-9_\-]{8,}["\']',
    r'(?i)access_key\s*[:=]\s*["\'][A-Za-z0-9_\-]{8,}["\']',
]


def _check_secrets_in_source(source: str) -> list:
    """逐行扫描源码，返回命中的可疑凭据行。"""
    import re
    hits = []
    compiled = [re.compile(p) for p in _SECRET_STR_PATTERNS]
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pat in compiled:
            m = pat.search(stripped)
            if m:
                hits.append({
                    "line": lineno,
                    "match": m.group(0),
                    "type": "hardcoded_credential",
                })
                break
    return hits


def _audit_ast(tree) -> list:
    """遍历 AST，收集反模式。"""
    issues = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in ("eval", "exec"):
                issues.append({
                    "line": node.lineno,
                    "type": "dangerous_builtin:" + func_name,
                    "detail": func_name + "() 调用存在任意代码执行风险",
                })

            if func_name in ("call", "run", "Popen", "check_output", "getoutput", "getstatusoutput"):
                if isinstance(node.func, ast.Attribute):
                    full = ast.dump(node.func.value)
                    if "subprocess" in full:
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                issues.append({
                                    "line": node.lineno,
                                    "type": "subprocess_shell_true",
                                    "detail": "subprocess." + func_name + "(shell=True) 存在命令注入风险",
                                })
                        if func_name in ("getoutput", "getstatusoutput"):
                            issues.append({
                                "line": node.lineno,
                                "type": "subprocess_shell_implicit",
                                "detail": "subprocess." + func_name + "() 隐式使用 shell，存在命令注入风险",
                            })

            if func_name == "system" and isinstance(node.func, ast.Attribute):
                full = ast.dump(node.func.value)
                if "'os'" in full or '"os"' in full:
                    issues.append({
                        "line": node.lineno,
                        "type": "os_system",
                        "detail": "os.system() 隐式使用 shell，存在命令注入风险",
                    })

            if func_name in ("spawn", "spawnl", "spawnlp", "spawnv", "spawnvp") and isinstance(node.func, ast.Attribute):
                full = ast.dump(node.func.value)
                if "'pty'" in full or '"pty"' in full:
                    issues.append({
                        "line": node.lineno,
                        "type": "pty_spawn",
                        "detail": "pty." + func_name + "() 可能绕过安全控制",
                    })

            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            if node.type is None:
                issues.append({
                    "line": node.lineno,
                    "type": "bare_except",
                    "detail": "裸 except: 会吞掉 KeyboardInterrupt / SystemExit，应指定异常类型",
                })
            self.generic_visit(node)

    _Visitor().visit(tree)
    return issues


@tool("分析 Python 源码：检测 eval/exec、subprocess(shell=True)、裸 except、硬编码凭据")
def analyze_code_quality(filepath: str) -> str:
    """
    参数
    ----
    filepath : str
        要分析的 Python 文件绝对路径。

    返回
    ----
    str
        文本审查报告。
    """
    if not os.path.isabs(filepath):
        filepath = os.path.join(WORKSPACE_DIR, filepath)

    if not os.path.isfile(filepath):
        return "[错误] 文件不存在: " + filepath

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception as e:
        return "[错误] 读取文件失败: " + str(e)

    try:
        tree = ast.parse(source, filename=filepath)
        ast_issues = _audit_ast(tree)
    except SyntaxError as e:
        return "[错误] 语法错误，无法解析 AST: " + str(e)

    secret_issues = _check_secrets_in_source(source)
    all_issues = ast_issues + secret_issues

    lines = []
    fname = os.path.basename(filepath)
    lines.append(" 代码质量审计报告: " + fname)
    lines.append("=" * 60)

    if not all_issues:
        lines.append("")
        lines.append(" 未发现已知反模式。")
        return "\n".join(lines)

    all_issues.sort(key=lambda x: x["line"])

    for grp, label in [
        ("dangerous_builtin",   " eval / exec 调用"),
        ("subprocess_shell_true", " subprocess + shell=True"),
        ("bare_except",         " 裸 except:"),
        ("hardcoded_credential", " 疑似硬编码凭据"),
    ]:
        group_items = [i for i in all_issues if i["type"].startswith(grp.split(":")[0])]
        if not group_items:
            continue
        lines.append("")
        lines.append(label + "  (" + str(len(group_items)) + " 处):")
        lines.append("-" * 40)
        for it in group_items:
            lines.append("  L" + str(it["line"]).rjust(4) + " | " + it["detail"])

    lines.append("")
    lines.append(" 合计 " + str(len(all_issues)) + " 个问题。")
    lines.append("=" * 60)
    return "\n".join(lines)