"""sandbox.py — 子进程执行沙箱（Windows Job Object / POSIX rlimit）。

能保证的：
- 墙钟超时 + 整棵进程树终止；
- Windows Job Object：进程/Job 内存上限、单进程 CPU 时间、活跃进程数上限、
  KILL_ON_JOB_CLOSE（父进程退出时子进程树一并清理）；
- POSIX：RLIMIT_AS / CPU / FSIZE / NPROC + 独立进程组；
- 环境变量白名单 + 密钥剥离（API_KEY/TOKEN/SECRET/PASSWORD/...），
  避免被测脚本直接读到 .env 里的密钥；
- 输出长度裁剪。

不能保证（Windows 本地后端）：
- 文件系统隔离与网络隔离。network="deny" 目前只做代理变量污染 + 标记
  network_enforced=False，不能阻止直接 socket 连接。需要强隔离时请使用
  容器/虚拟机后端（本模块预留了 backend 字段，后续可接 Docker）。
"""
import ctypes
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

PASSTHROUGH_ENV = {
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "TMPDIR", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "OS", "LANG", "LC_ALL", "USERNAME", "USERDOMAIN", "COMPUTERNAME",
    "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
}
_SECRET_RE = re.compile(
    r"(API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION)",
    re.IGNORECASE,
)

# ---------------- Windows Job Object ----------------
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _new_job(memory_mb: int, cpu_s: float, max_processes: int):
    """创建并配置 Job Object，返回 (handle, error)。仅 Windows。"""
    if os.name != "nt":
        return None, "not-windows"
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.SetInformationJobObject.restype = ctypes.c_int
        k32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        k32.AssignProcessToJobObject.restype = ctypes.c_int
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]

        hjob = k32.CreateJobObjectW(None, None)
        if not hjob:
            return None, f"CreateJobObjectW failed: {ctypes.get_last_error()}"

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_mb and memory_mb > 0:
            flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_JOB_MEMORY
            info.ProcessMemoryLimit = int(memory_mb) * 1024 * 1024
            info.JobMemoryLimit = int(memory_mb) * 1024 * 1024
        if cpu_s and cpu_s > 0:
            flags |= _JOB_OBJECT_LIMIT_PROCESS_TIME
            info.BasicLimitInformation.PerProcessUserTimeLimit = int(cpu_s * 10_000_000)
        if max_processes and max_processes > 0:
            flags |= _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = int(max_processes)
        info.BasicLimitInformation.LimitFlags = flags

        ok = k32.SetInformationJobObject(
            hjob, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            err = ctypes.get_last_error()
            k32.CloseHandle(hjob)
            return None, f"SetInformationJobObject failed: {err}"
        return hjob, ""
    except Exception as e:
        return None, f"job-init-error: {e}"


def _assign_job(hjob, proc) -> str:
    if not hjob or os.name != "nt":
        return ""
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.restype = ctypes.c_int
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        handle = int(getattr(proc, "_handle", 0) or 0)
        if not handle:
            return "process-handle-unavailable"
        if not k32.AssignProcessToJobObject(hjob, handle):
            return f"AssignProcessToJobObject failed: {ctypes.get_last_error()}"
        return ""
    except Exception as e:
        return f"job-assign-error: {e}"


def _close_job(hjob):
    if hjob and os.name == "nt":
        try:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(hjob)
        except Exception:
            pass


def _kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _posix_preexec(memory_mb: int, cpu_s: float, max_processes: int):
    if os.name == "posix":
        try:
            import resource
        except Exception:
            return None

        def _set():
            try:
                os.setsid()
            except Exception:
                pass
            try:
                if memory_mb:
                    lim = int(memory_mb) * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
                if cpu_s:
                    cpu = max(1, int(cpu_s))
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                if max_processes:
                    resource.setrlimit(resource.RLIMIT_NPROC,
                                       (int(max_processes), int(max_processes)))
            except Exception:
                pass
        return _set
    return None


def build_env(cfg: dict = None, extra: dict = None) -> dict:
    """环境变量白名单 + 密钥剥离。extra 中的变量视为显式授权，不过滤。"""
    cfg = cfg or {}
    allowed = set(PASSTHROUGH_ENV)
    allowed.update(str(x).upper() for x in (cfg.get("env_passthrough") or []))
    env = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if ku not in allowed:
            continue
        if _SECRET_RE.search(ku):
            continue
        env[k] = v
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # 标记沙箱子进程：safety 不再从 .env 读取真实密钥，MCP 插件不自动拉外部进程
    env["AI_AGENT_SANDBOX"] = "1"
    return env


# ================= Windows AppContainer 真沙箱 =================
_AC_LOCK = threading.RLock()
_AC_STATE = {
    "tried": False, "ok": False, "sid": None, "sid_str": "",
    "error": "", "profile": "", "sandbox_root": "",
}

class _AC_SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("nLength", ctypes.c_ulong),
                ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", ctypes.c_int)]

class _AC_STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p), ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_ulong), ("dwY", ctypes.c_ulong),
        ("dwXSize", ctypes.c_ulong), ("dwYSize", ctypes.c_ulong),
        ("dwXCountChars", ctypes.c_ulong), ("dwYCountChars", ctypes.c_ulong),
        ("dwFillAttribute", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("wShowWindow", ctypes.c_ushort), ("cbReserved2", ctypes.c_ushort),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", ctypes.c_void_p), ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]

class _AC_STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", _AC_STARTUPINFOW),
                ("lpAttributeList", ctypes.c_void_p)]

class _AC_PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", ctypes.c_void_p), ("hThread", ctypes.c_void_p),
                ("dwProcessId", ctypes.c_ulong), ("dwThreadId", ctypes.c_ulong)]

class _AC_SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [("AppContainerSid", ctypes.c_void_p),
                ("Capabilities", ctypes.c_void_p),
                ("CapabilityCount", ctypes.c_ulong),
                ("Reserved", ctypes.c_ulong)]

class _AC_SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", ctypes.c_ulong)]


def _icacls(path: str, *args) -> int:
    try:
        r = subprocess.run(["icacls", path] + list(args),
                           capture_output=True, text=True, errors="replace",
                           timeout=20)
        return r.returncode
    except Exception:
        return -1


def _ac_init(cfg: dict = None) -> dict:
    cfg = cfg or {}
    with _AC_LOCK:
        if _AC_STATE["tried"]:
            return _AC_STATE
        _AC_STATE["tried"] = True
        if os.environ.get("AI_AGENT_DISABLE_APPCONTAINER") == "1":
            _AC_STATE["error"] = "appcontainer-disabled-by-env"
            return _AC_STATE
        if os.name != "nt" or not cfg.get("appcontainer", True):
            _AC_STATE["error"] = "not-windows-or-disabled"
            return _AC_STATE
        try:
            import ctypes.wintypes as w
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            adv = ctypes.WinDLL("advapi32", use_last_error=True)
            userenv = ctypes.WinDLL("userenv", use_last_error=True)
            userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
            userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
                ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
            userenv.CreateAppContainerProfile.restype = ctypes.c_long
            userenv.CreateAppContainerProfile.argtypes = [
                ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]
            adv.ConvertSidToStringSidW.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]

            profile = str(cfg.get("appcontainer_profile") or "AIAgentSandbox")
            sid = ctypes.c_void_p()
            # 必须先 Create：只 Derive 会得到 SID 但不会创建 profile 文件夹，
            # 后续 CreateProcess 会以 ERROR_FILE_NOT_FOUND 失败。
            hr = userenv.CreateAppContainerProfile(
                profile, "AI Agent Sandbox", "untrusted execution",
                None, 0, ctypes.byref(sid))
            if not sid.value:
                hr = userenv.DeriveAppContainerSidFromAppContainerName(
                    profile, ctypes.byref(sid))
            if not sid.value:
                raise OSError(f"CreateAppContainerProfile failed: {hr & 0xFFFFFFFF:#x}")
            sid_str = ctypes.c_wchar_p()
            if not adv.ConvertSidToStringSidW(sid, ctypes.byref(sid_str)):
                raise OSError("ConvertSidToStringSidW failed")

            py_dir = os.path.dirname(os.path.realpath(sys.executable))
            home = os.path.expanduser("~")
            parents = []
            cur = py_dir
            while cur and len(cur) > 3:
                parents.append(cur)
                if os.path.normcase(cur) == os.path.normcase(home):
                    break
                nxt = os.path.dirname(cur)
                if nxt == cur:
                    break
                cur = nxt
            if home not in parents:
                parents.append(home)
            # 只给 traverse(X)，不给列目录/读文件
            for p in parents:
                _icacls(p, "/remove", f"*{sid_str.value}")
                _icacls(p, "/grant:r", f"*{sid_str.value}:(X)")
            _icacls(py_dir, "/remove", f"*{sid_str.value}")
            _icacls(py_dir, "/grant:r", f"*{sid_str.value}:(OI)(CI)RX")

            temp_root = tempfile.gettempdir()
            _icacls(temp_root, "/remove", f"*{sid_str.value}")
            _icacls(temp_root, "/grant:r", f"*{sid_str.value}:(X)")
            sandbox_root = os.path.join(temp_root, "ai_agent_sandbox")
            os.makedirs(sandbox_root, exist_ok=True)
            _icacls(sandbox_root, "/remove", f"*{sid_str.value}")
            _icacls(sandbox_root, "/grant:r", f"*{sid_str.value}:(OI)(CI)M")

            workspace = cfg.get("_workspace") or ""
            if workspace:
                # 真实工作区不给 AppContainer 任何权限，执行走镜像
                _icacls(workspace, "/remove", f"*{sid_str.value}")
                env_file = os.path.join(workspace, ".env")
                if os.path.isfile(env_file):
                    _icacls(env_file, "/remove:d", f"*{sid_str.value}")
                    _icacls(env_file, "/remove", f"*{sid_str.value}")

            _AC_STATE.update(ok=True, sid=sid, sid_str=sid_str.value,
                             profile=profile, sandbox_root=sandbox_root)
        except Exception as e:
            _AC_STATE["error"] = f"{type(e).__name__}: {e}"
        return _AC_STATE


def appcontainer_available(cfg: dict = None) -> bool:
    return bool(_ac_init(cfg).get("ok"))


def appcontainer_status(cfg: dict = None) -> dict:
    st = _ac_init(cfg)
    return {"ok": st.get("ok", False), "profile": st.get("profile"),
            "sid": st.get("sid_str"), "error": st.get("error", "")}


def make_mirror(workspace: str, cfg: dict = None) -> str:
    """把工作区安全文件复制到临时镜像目录；.env/经验库/缓存/构建产物不进入沙箱。"""
    st = _ac_init(cfg)
    if not st.get("ok"):
        return ""
    root = os.path.join(st["sandbox_root"], "run_" + uuid.uuid4().hex[:12])
    excluded_dirs = {
        ".git", "__pycache__", ".experiences", ".traces", ".agent_checkpoints",
        ".agent_index", ".venv", "venv", "node_modules", "bin", "obj", ".vs",
        "logs", ".claude",
    }

    def _ignore(_dir, names):
        out = set()
        for n in names:
            if n in excluded_dirs or n.startswith("."):
                out.add(n)
            elif n == ".env" or n.startswith("_tmp"):
                out.add(n)
            elif ".bak_" in n or n.endswith((".pyc", ".blend")):
                out.add(n)
        return out

    try:
        os.makedirs(root, exist_ok=True)
        shutil.copytree(workspace, root, ignore=_ignore, dirs_exist_ok=True)
        os.makedirs(os.path.join(root, "temp"), exist_ok=True)
        _icacls(root, "/grant:r", f"*{st['sid_str']}:(OI)(CI)M")
        return root
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        return ""


def cleanup_mirror(root: str) -> None:
    if root and os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)


def _ac_build_caps(sid, network: str):
    refs = []
    if str(network).lower() != "allow":
        return _AC_SECURITY_CAPABILITIES(sid, None, 0, 0), refs
    try:
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        adv.ConvertStringSidToSidW.argtypes = [
            ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
        cap_sid = ctypes.c_void_p()
        if not adv.ConvertStringSidToSidW("S-1-15-3-1", ctypes.byref(cap_sid)):
            return _AC_SECURITY_CAPABILITIES(sid, None, 0, 0), refs
        arr = (_AC_SID_AND_ATTRIBUTES * 1)()
        arr[0].Sid = cap_sid
        arr[0].Attributes = 0x4  # SE_GROUP_ENABLED
        refs.extend([cap_sid, arr])
        caps = _AC_SECURITY_CAPABILITIES(
            sid, ctypes.cast(arr, ctypes.c_void_p), 1, 0)
        return caps, refs
    except Exception:
        return _AC_SECURITY_CAPABILITIES(sid, None, 0, 0), refs


def run_appcontainer(cmd, cfg: dict = None, timeout: float = None,
                     cwd: str = None, env_extra: dict = None,
                     purpose: str = "", network: str = "deny",
                     memory_mb: int = None, cpu_time_s: float = None,
                     max_processes: int = None) -> dict:
    """在 AppContainer 中运行命令（仅 Windows）。调用方负责提供镜像 cwd。"""
    cfg = dict(cfg or {})
    st = _ac_init(cfg)
    meta = {"backend": "appcontainer", "purpose": purpose,
            "network_requested": network, "network_enforced": True,
            "filesystem_isolated": True, "job_assigned": False,
            "job_error": "", "mirror": cwd or ""}
    if not st.get("ok"):
        return {"returncode": -1, "stdout": "", "stderr": "", "timeout": False,
                "error": f"AppContainer 不可用: {st.get('error')}", "meta": meta}
    if isinstance(cmd, str):
        cmd = shlex.split(cmd, posix=False)
    cmd = [str(x) for x in (cmd or [])]
    if not cmd:
        return {"returncode": -1, "stdout": "", "stderr": "",
                "timeout": False, "error": "命令为空", "meta": meta}
    if not cwd or not os.path.isdir(cwd):
        return {"returncode": -1, "stdout": "", "stderr": "",
                "timeout": False, "error": "AppContainer 需要有效 cwd（镜像目录）",
                "meta": meta}
    try:
        import ctypes.wintypes as w
        import pywintypes
        import win32file
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": "",
                "timeout": False, "error": f"缺少 pywin32: {e}", "meta": meta}

    timeout = float(timeout if timeout is not None else cfg.get("timeout_s", 120))
    memory_mb = int(memory_mb if memory_mb is not None
                    else (cfg.get("memory_mb", 4096) or 0))
    cpu_s = float(cpu_time_s if cpu_time_s is not None
                  else (cfg.get("cpu_time_s", 300) or 0))
    max_proc = int(max_processes if max_processes is not None
                   else (cfg.get("max_processes", 128) or 0))
    clip = int(cfg.get("clip_output_chars", 20000) or 20000)

    env = build_env(cfg, env_extra)
    temp_dir = os.path.join(cwd, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    env["TEMP"] = env["TMP"] = temp_dir
    env["USERPROFILE"] = env["HOME"] = cwd
    env["AI_AGENT_SANDBOX"] = "1"

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    sa = _AC_SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.bInheritHandle = 1
    k32.CreatePipe.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                               ctypes.POINTER(ctypes.c_void_p),
                               ctypes.POINTER(_AC_SECURITY_ATTRIBUTES), ctypes.c_ulong]
    pipes = {}
    for name in ("out", "err"):
        r = ctypes.c_void_p()
        wr = ctypes.c_void_p()
        if not k32.CreatePipe(ctypes.byref(r), ctypes.byref(wr),
                              ctypes.byref(sa), 0):
            return {"returncode": -1, "stdout": "", "stderr": "",
                    "timeout": False, "error": "CreatePipe failed",
                    "meta": meta}
        k32.SetHandleInformation(r, 1, 0)
        pipes[name] = (r, wr)
    k32.CreateFileW.restype = ctypes.c_void_p
    null_in = k32.CreateFileW("NUL", 0x80000000, 3, ctypes.byref(sa), 3, 0, None)

    size = ctypes.c_size_t(0)
    k32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    attr_buf = ctypes.create_string_buffer(size.value)
    k32.InitializeProcThreadAttributeList(attr_buf, 1, 0, ctypes.byref(size))
    caps, cap_refs = _ac_build_caps(st["sid"], network)
    ok_attr = k32.UpdateProcThreadAttribute(
        attr_buf, 0, 0x00020009, ctypes.byref(caps),
        ctypes.sizeof(caps), None, None)

    si = _AC_STARTUPINFOEXW()
    si.StartupInfo.cb = ctypes.sizeof(_AC_STARTUPINFOEXW)
    si.StartupInfo.dwFlags = 0x100
    si.StartupInfo.hStdInput = null_in
    si.StartupInfo.hStdOutput = pipes["out"][1]
    si.StartupInfo.hStdError = pipes["err"][1]
    si.lpAttributeList = ctypes.cast(attr_buf, ctypes.c_void_p)

    block = "\0".join(f"{k}={env[k]}" for k in sorted(env, key=str.lower)) + "\0\0"
    env_bytes = block.encode("utf-16-le")
    env_buf = ctypes.create_string_buffer(env_bytes, len(env_bytes))
    cmdline = subprocess.list2cmdline(cmd)
    cmd_buf = ctypes.create_unicode_buffer(cmdline)
    pi = _AC_PROCESS_INFORMATION()
    flags = 0x00080000 | 0x08000000 | 0x00000400  # EX_STARTUPINFO|NO_WINDOW|UNICODE_ENV
    ok = k32.CreateProcessW(cmd[0], cmd_buf, None, None, True, flags,
                            env_buf, cwd, ctypes.byref(si), ctypes.byref(pi))
    if not ok:
        err = ctypes.get_last_error()
        for name in ("out", "err"):
            k32.CloseHandle(pipes[name][0])
            k32.CloseHandle(pipes[name][1])
        if null_in and null_in != ctypes.c_void_p(-1).value:
            k32.CloseHandle(null_in)
        return {"returncode": -1, "stdout": "", "stderr": "",
                "timeout": False,
                "error": (f"CreateProcessW失败 err={err}" if ok_attr
                          else "UpdateProcThreadAttribute failed"), "meta": meta}

    for name in ("out", "err"):
        k32.CloseHandle(pipes[name][1])
    if null_in and null_in != ctypes.c_void_p(-1).value:
        k32.CloseHandle(null_in)

    hjob, job_err = _new_job(memory_mb, cpu_s, max_proc)
    if hjob:
        k32.AssignProcessToJobObject.restype = ctypes.c_int
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if k32.AssignProcessToJobObject(hjob, pi.hProcess):
            meta["job_assigned"] = True
        else:
            meta["job_error"] = f"AssignProcessToJobObject failed: {ctypes.get_last_error()}"
    else:
        meta["job_error"] = job_err

    outputs = {"out": bytearray(), "err": bytearray()}

    def _reader(name, handle):
        h = pywintypes.HANDLE(int(handle.value))
        while True:
            try:
                _, data = win32file.ReadFile(h, 65536)
            except Exception:
                break
            if not data:
                break
            outputs[name] += data

    threads = [threading.Thread(target=_reader, args=(n, pipes[n][0]), daemon=True)
               for n in ("out", "err")]
    for t in threads:
        t.start()

    wait_rc = k32.WaitForSingleObject(pi.hProcess, int(timeout * 1000))
    timed_out = wait_rc == 258  # WAIT_TIMEOUT
    if timed_out:
        _kill_tree(type("_P", (), {"poll": lambda self: None,
                                   "pid": pi.dwProcessId,
                                   "kill": lambda self: None})())
        k32.WaitForSingleObject(pi.hProcess, 5000)
    code = ctypes.c_ulong(0)
    k32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    for t in threads:
        t.join(3)
    for name in ("out", "err"):
        k32.CloseHandle(pipes[name][0])
    k32.CloseHandle(pi.hProcess)
    k32.CloseHandle(pi.hThread)
    _close_job(hjob)

    return {
        "returncode": int(code.value),
        "stdout": _clip(outputs["out"].decode("utf-8", "replace"), clip),
        "stderr": _clip(outputs["err"].decode("utf-8", "replace"), clip),
        "timeout": timed_out,
        "error": "",
        "meta": meta,
    }


def _clip(text, limit: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[沙箱输出已截断，原长 {len(s)} 字符]"


def run(cmd, cfg: dict = None, timeout: float = None, cwd: str = None,
        env_extra: dict = None, input_text: str = None, purpose: str = "",
        network: str = None, memory_mb: int = None, cpu_time_s: float = None,
        max_processes: int = None) -> dict:
    """运行子进程。返回 {returncode, stdout, stderr, timeout, error, meta}。

    cmd: list[str]（推荐）或 str（按 shell=False 的 shlex 规则切分）。
    memory_mb/cpu_time_s/max_processes 不为 None 时覆盖 cfg 中的全局限制。
    """
    cfg = dict(cfg or {})
    if isinstance(cmd, str):
        cmd = shlex.split(cmd, posix=(os.name != "nt"))
    cmd = [str(x) for x in (cmd or [])]
    if not cmd:
        return {"returncode": -1, "stdout": "", "stderr": "",
                "timeout": False, "error": "命令为空", "meta": {}}

    timeout = float(timeout if timeout is not None else cfg.get("timeout_s", 120))
    memory_mb = int(memory_mb if memory_mb is not None
                    else (cfg.get("memory_mb", 2048) or 0))
    cpu_s = float(cpu_time_s if cpu_time_s is not None
                  else (cfg.get("cpu_time_s", 60) or 0))
    max_proc = int(max_processes if max_processes is not None
                   else (cfg.get("max_processes", 32) or 0))
    clip = int(cfg.get("clip_output_chars", 20000) or 20000)
    network = str(network if network is not None else cfg.get("network", "allow")).lower()

    env = build_env(cfg, env_extra)
    if network == "deny":
        # 本地后端无法真正断网，只能污染常见代理变量；metadata 会明确标记。
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy"):
            env[k] = "http://127.0.0.1:9"
        env["NO_PROXY"] = ""
        env["no_proxy"] = ""

    meta = {
        "backend": "local-job" if os.name == "nt" else "local-rlimit",
        "purpose": purpose,
        "memory_mb": memory_mb,
        "cpu_time_s": cpu_s,
        "max_processes": max_proc,
        "network_requested": network,
        "network_enforced": False,
        "job_assigned": False,
        "job_error": "",
    }

    if not cfg.get("enabled", True):
        meta["backend"] = "disabled"

    hjob = None
    proc = None
    t0 = time.perf_counter()
    try:
        if cfg.get("enabled", True) and os.name == "nt":
            hjob, job_err = _new_job(memory_mb, cpu_s, max_proc)
            if job_err:
                meta["job_error"] = job_err

        creationflags = 0
        if os.name == "nt" and cfg.get("create_no_window", True):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            cwd=cwd, env=env, creationflags=creationflags,
            preexec_fn=None if os.name == "nt" else _posix_preexec(
                memory_mb if cfg.get("enabled", True) else 0,
                cpu_s if cfg.get("enabled", True) else 0,
                max_proc if cfg.get("enabled", True) else 0),
        )

        if hjob:
            assign_err = _assign_job(hjob, proc)
            meta["job_assigned"] = not bool(assign_err)
            if assign_err:
                meta["job_error"] = assign_err

        timed_out = False
        try:
            stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except Exception:
                stdout, stderr = "", ""

        meta["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "returncode": proc.returncode,
            "stdout": _clip(stdout, clip),
            "stderr": _clip(stderr, clip),
            "timeout": timed_out,
            "error": "",
            "meta": meta,
        }
    except FileNotFoundError as e:
        return {"returncode": -1, "stdout": "", "stderr": "", "timeout": False,
                "error": f"找不到命令: {cmd[0]} ({e})", "meta": meta}
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": "", "timeout": False,
                "error": f"{type(e).__name__}: {e}", "meta": meta}
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
        # KILL_ON_JOB_CLOSE：关闭句柄时清理残留子进程
        _close_job(hjob)
