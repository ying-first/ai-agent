# -*- coding: utf-8 -*-
"""外部应用程序统一调用插件（Blender / Visual Studio）。

由旧插件 blender_ops.py + vs_ops.py 合并而来：
只暴露 1 个 @tool（run_external_app），减少常驻工具数量。
所有命令统一走 safety 的 execute_approved_command 原语（本插件先完成授权），不使用 subprocess。
"""

import os
import re
import shlex

from safety import human_approve, execute_approved_command, tool

BLENDER_EXE = r"D:\blander\blender.exe"
DEVENV_EXE = r"D:\Visual Studio 2026\Common7\IDE\devenv.exe"

# app 别名 -> (规范名, 可执行文件)
APP_REGISTRY = {
    "blender": ("blender", BLENDER_EXE),
    "vs": ("visual_studio", DEVENV_EXE),
    "vs2026": ("visual_studio", DEVENV_EXE),
    "visualstudio": ("visual_studio", DEVENV_EXE),
    # 归一化已去掉空格/_/-，"visual_studio" 会自动落到 "visualstudio" 这个键
    "devenv": ("visual_studio", DEVENV_EXE),
}


def _normalize_app(app):
    """归一化 app 名：去空白、转小写。"""
    if not app or not isinstance(app, str):
        return ""
    return re.sub(r"[\s_\-]+", "", app.strip().lower())


def _normalize_args(args):
    """把 str / list / tuple / None 统一成命令行参数列表。"""
    if args is None:
        return []
    if isinstance(args, str):
        return shlex.split(args, posix=True) if args.strip() else []
    if isinstance(args, (list, tuple)):
        return [str(a) for a in args]
    return None


def _cmd_preview(exe, exe_args):
    """生成安全的命令预览文本。"""
    return " ".join([f'"{exe}"'] + [f'"{a}"' if " " in a else a for a in exe_args])


def _missing_target_hint(exe_args):
    """检查参数里的关键脚本文件是否存在，返回提示文本或空串。"""
    for flag in ("--python", "-P"):
        if flag in exe_args:
            idx = exe_args.index(flag)
            if idx + 1 < len(exe_args):
                target = exe_args[idx + 1]
                if not os.path.isfile(target):
                    return f"❌ 找不到脚本文件: {target}"
    for arg in exe_args:
        if arg.lower().endswith(".sln") and not os.path.isfile(arg):
            return f"❌ 找不到解决方案文件: {arg}"
    return ""


@tool("统一调用外部程序：run_external_app(app, args)，app 可选 blender / visual_studio（vs），args 为参数列表或字符串")
def run_external_app(app: str, args=None):
    """调用 Blender 无头执行脚本，或用 devenv.exe 编译 VS 解决方案。

    Args:
        app: 目标程序，支持 blender / vs / visual_studio / devenv（不区分大小写）。
        args: 传给该程序的参数。可为 list（推荐，如 ["--background", "--python", "a.py"]）
              或字符串（按空格切分）；Visual Studio 典型用法 ["xxx.sln", "/Build", "Release"]。

    示例:
        run_external_app("blender", ["--background", "--python", "D:/s/xx.py"])
        run_external_app("visual_studio", ["D:/p/test.sln", "/Build", "Release"])
    """
    key = _normalize_app(app)
    if not key:
        return "❌ 参数 app 无效，请传 blender 或 visual_studio。"

    if key not in APP_REGISTRY:
        return (
            f"❌ 不支持的程序: {app}\n"
            f"可用选项: {sorted(set(v[0] for v in APP_REGISTRY.values()))}"
        )

    name, exe = APP_REGISTRY[key]

    exe_args = _normalize_args(args)
    if exe_args is None:
        return "❌ 参数 args 类型无效，请传 list 或 str。"

    if not os.path.isfile(exe):
        return f"❌ 未找到 [{name}] 可执行文件: {exe}"

    hint = _missing_target_hint(exe_args)
    if hint:
        return hint

    # 参数全部校验通过后再请求授权，避免非法调用也弹窗打扰用户
    if not human_approve(f"启动外部程序 [{name}]，参数: {exe_args}"):
        return "❌ 用户未授权，操作已取消。"

    cmd = [exe] + exe_args
    # 本函数已完成授权，直接调用底层执行器，避免二次弹窗
    return execute_approved_command(cmd)
