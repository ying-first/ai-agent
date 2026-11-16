"""gui_frontend.py — 独立图形前端，转发文本框命令到 agent.py:run_agent。"""
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

APP_TITLE = "Agent 命令台（独立前端）"
POLL_MS = 100
MAX_OUTPUT_LINES = 5000
TRIM_CHUNK_LINES = 1000

# 模块级锁：保证同一时刻只有一个 run_agent 在执行。
# watchdog 释放 UI 后用户可能立即发新任务，此时旧线程可能还在跑，
# 用锁把两个 run_agent 串行化，避免 chat_history 竞态。
_RUN_AGENT_LOCK = threading.Lock()


def _get_run_agent():
    from agent import run_agent
    return run_agent


class CommandConsole:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("760x560")
        self.queue = queue.Queue()
        self.busy = False
        # msg_seq：每次发送 / 取消都递增，用于丢弃在途的过期结果。
        self.msg_seq = 0
        # 正在执行的后台线程；取消后需等它退出才能放行新任务，避免并发竞态。
        self.pending_thread = None
        self.cancelling = False
        # output_truncated：保证裁剪提示只插入一次。
        self.output_truncated = False
        
        # streamed_this_turn：本轮任务是否已有任何 chunk 被流式显示。
        # reply 到达时用于判断"正文是否已经在流式阶段画过"，避免重复显示。
        self.streamed_this_turn = False

        out_frame = ttk.Frame(root, padding=(8, 8, 8, 4))
        out_frame.pack(fill=tk.BOTH, expand=True)
        self.output = scrolledtext.ScrolledText(
            out_frame, wrap=tk.WORD, state=tk.DISABLED
        )
        self.output.pack(fill=tk.BOTH, expand=True)

        in_frame = ttk.Frame(root, padding=(8, 4, 8, 8))
        in_frame.pack(fill=tk.X)
        self.input = tk.Text(in_frame, height=4, wrap=tk.WORD)
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input.bind("<Control-Return>", self._on_ctrl_enter)
        self.input.focus_set()

        btn_frame = ttk.Frame(in_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        self.send_btn = ttk.Button(btn_frame, text="发送", command=self.send)
        self.send_btn.pack(fill=tk.X)
        # 取消按钮：无法中断后台线程，但取消后会锁定界面直到旧线程退出。
        self.cancel_btn = ttk.Button(
            btn_frame, text="取消", command=self.cancel, state=tk.DISABLED
        )
        self.cancel_btn.pack(fill=tk.X, pady=(4, 0))
        # 发送图片：选图后交给独立队列做识别，复用忙碌态与取消逻辑。
        self.image_btn = ttk.Button(
            btn_frame, text="发送图片", command=self.send_image
        )
        self.image_btn.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_frame, text="清空输出", command=self.clear_output).pack(
            fill=tk.X, pady=(4, 0)
        )

        self.status = tk.StringVar(value="就绪 · Ctrl+Enter 发送")
        ttk.Label(root, textvariable=self.status, padding=(8, 2)).pack(fill=tk.X)
        # 图片识别走独立队列，避免污染文本对话的 _poll 结果格式。
        self.image_queue = queue.Queue()

        self._say("系统", "前端已启动，输入内容将转发给 agent.py:run_agent。")
        self.root.after(POLL_MS, self._poll)
        self.root.after(POLL_MS, self._poll_image)

    # ---------------- 输出区 ----------------
    def _say(self, who, text):
        if text is None:
            text = "（无内容）"
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, "[{}] {}\n\n".format(who, str(text)))
        self._trim_output()
        self.output.configure(state=tk.DISABLED)
        self.output.see(tk.END)

    def _trim_output(self):
        """裁剪过早输出。调用前 output 必须处于 state=NORMAL。"""
        try:
            line_count = int(self.output.index("end-1c").split(".")[0])
        except (tk.TclError, ValueError):
            return
        if line_count <= MAX_OUTPUT_LINES:
            return
        cut = line_count - MAX_OUTPUT_LINES + TRIM_CHUNK_LINES
        if cut <= 1:
            return
        self.output.delete("1.0", "{}.0".format(cut + 1))
        if not self.output_truncated:
            self.output_truncated = True
            self.output.insert(tk.END, "[系统] 输出过多，已自动裁剪较早内容。\n\n")

    def clear_output(self):
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.configure(state=tk.DISABLED)
        self.output_truncated = False
        self.status.set("输出已清空 · Ctrl+Enter 发送")

    # ---------------- 状态控制 ----------------
    def _set_busy(self, busy):
        self.busy = busy
        self.send_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.cancel_btn.configure(state=tk.NORMAL if busy else tk.DISABLED)
        self.image_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)

    def _on_ctrl_enter(self, event):
        # 返回 "break" 阻断 Tk 类绑定 <Return>，否则发送后会残留一个换行。
        self.send()
        return "break"

    def _finish_cancelled(self):
        """取消后释放 UI 锁。幂等：正常过期结果和 watchdog 谁先到谁释放，
        后到的直接返回（busy 已为 False 或 cancelling 已为 False）。"""
        if not self.cancelling or not self.busy:
            return
        self.pending_thread = None
        self.cancelling = False
        try:
            self._set_busy(False)
            # 注意：不承诺旧线程已死，只说 UI 不再等它
            self.status.set("已取消 · 后台线程可能仍在收尾（输出已丢弃）")
            self.input.focus_set()
        except tk.TclError:
            pass

    # ---------------- 图片识别 ----------------
    def send_image(self):
        """选图并提交识别任务。识别放在后台线程，结果由 _poll_image 取回。"""
        if self.busy:
            self.status.set("正在执行中，请先等待或取消")
            return
        path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            self.status.set("未选择图片")
            return

        question = self.input.get("1.0", tk.END).strip()
        self._say("我", "[图片] %s%s" % (
            path, "｜提问：" + question if question else ""))
        # 输入框内容已作为本次图片提问使用，清空避免误以为下一条还会附带。
        self.input.delete("1.0", tk.END)
        self.msg_seq += 1
        seq = self.msg_seq
        self.cancelling = False
        self.streamed_this_turn = False
        self._set_busy(True)
        self.status.set("正在识别图片…")
        self.pending_thread = threading.Thread(
            target=self._image_worker, args=(seq, path, question), daemon=True
        )
        self.pending_thread.start()

    def _image_worker(self, seq, path, question):
        try:
            import image_tools
            result = image_tools.recognize_image(path, question)
        except Exception as e:
            result = "图片识别失败：%s" % e
        self.image_queue.put((seq, result))

    def _poll_image(self):
        """独立轮询图片队列；过期结果丢弃，并保持轮询链不断。"""
        try:
            try:
                seq, result = self.image_queue.get_nowait()
            except queue.Empty:
                return
            if seq != self.msg_seq:
                # 取消后的过期结果：旧线程已结束，释放锁定。
                self._finish_cancelled()
                return
            self.pending_thread = None
            self._say("助手", result)
            self._set_busy(False)
            self.status.set("图片识别完成 · Ctrl+Enter 发送")
            self.input.focus_set()
        except tk.TclError:
            return
        finally:
            # 与 _poll 对称：无论是否发生异常，都保证轮询链不断。
            try:
                if self.root.winfo_exists():
                    self.root.after(POLL_MS, self._poll_image)
            except tk.TclError:
                pass

    def cancel(self):
        if not self.busy:
            self.status.set("当前没有正在执行的任务")
            return
        if self.cancelling:
            self.status.set("正在取消 · 等待后台线程退出…")
            return
        self.cancelling = True
        self.msg_seq += 1
        self.status.set("正在取消 · 等待后台线程退出…")
        self.input.focus_set()
        # watchdog：8s 后强释放 UI 锁，不等旧线程真退出。
        # 旧线程的 on_text / reply 因 seq != msg_seq 会被 _poll 丢弃，
        # 不会污染新任务；释放的只是 UI 锁，不是杀线程（Python 杀不掉）。
        self.root.after(8000, self._finish_cancelled)

    # ---------------- 发送与后台执行 ----------------
    def send(self):
        if self.busy:
            self.status.set("上一条还在处理中，请稍候…")
            return
        text = self.input.get("1.0", tk.END).strip()
        if not text:
            self.status.set("输入为空 · 请先输入内容")
            return
        self.input.delete("1.0", tk.END)
        self._say("我", text)
        self.msg_seq += 1
        seq = self.msg_seq
        self.cancelling = False
        self.streamed_this_turn = False   # 新任务起点：本轮尚未流式显示任何内容
        self._set_busy(True)
        self.status.set("思考中…（后台执行，界面不阻塞）")
        self.pending_thread = threading.Thread(
            target=self._worker, args=(text, seq), daemon=True
        )
        self.pending_thread.start()

    def _ask_approval_blocking(self, seq, action, timeout=180.0):
        """后台线程发起的权限审批：投递到主线程弹窗并等待结果。"""
        ev = threading.Event()
        ans = {"ok": False}
        self.queue.put((seq, "approve", (action, ev, ans)))
        deadline = time.time() + timeout
        while not ev.wait(0.2):
            if seq != self.msg_seq:
                return False
            if time.time() >= deadline:
                return False
        return bool(ans["ok"])

    def _worker(self, text, seq):
        def on_text(chunk):
            # 过期任务（已取消）不再推流，避免污染新任务的输出
            if seq == self.msg_seq:
                self.queue.put((seq, "stream", chunk))

        def ask_approve_gui(action):
            # delete / execute / network 级工具需要授权时弹出模态确认框
            return self._ask_approval_blocking(seq, action)

        def ask_continue_gui(add):
            # 步数上限时调用：自动追加，但通过 status 通道告知用户。
            # 不直接 set(self.status)：本函数在 run_agent 后台线程里执行，
            # 跨线程写 Tk 变量不安全，统一走 queue 由 _poll（主线程）落地。
            if seq == self.msg_seq:
                self.queue.put((
                    seq, "status",
                    "已达步数上限，自动追加 {} 步…（如需中止请点「取消」）".format(add)
                ))
            return True

        try:
            run_agent = _get_run_agent()
            # 注入权限审批回调：权限分级引擎会通过它弹窗询问 delete/execute/network
            from safety import set_human_approve
            set_human_approve(ask_approve_gui)
            # 模块级锁：watchdog 提前放行后可能出现"旧 run_agent 未退 + 新 run_agent 启动"，
            # 用锁保证同一时刻只有一个 run_agent，避免 chat_history 竞态。
            with _RUN_AGENT_LOCK:
                reply = run_agent(text, on_text=on_text, ask_continue=ask_continue_gui)
        except Exception as e:
            self.queue.put((seq, "error", "{}: {}".format(type(e).__name__, e)))
            return
        self.queue.put((seq, "reply", reply))

    # ---------------- 队列轮询 ----------------
    def _poll(self):
        try:
            while True:
                try:
                    seq, kind, payload = self.queue.get_nowait()
                except queue.Empty:
                    break
                if seq != self.msg_seq:
                    # 过期审批请求必须唤醒后台线程，否则它会一直等到超时
                    if kind == "approve":
                        _, ev, ans = payload
                        ans["ok"] = False
                        ev.set()
                    self._finish_cancelled()
                    continue

                # ===== 权限审批：主线程弹模态框，后台线程等待 =====
                if kind == "approve":
                    action, ev, ans = payload
                    try:
                        ans["ok"] = bool(messagebox.askyesno(
                            "Agent 需要授权", action, parent=self.root))
                    except tk.TclError:
                        ans["ok"] = False
                    finally:
                        ev.set()
                    continue

                # ===== 流式 chunk：不进 try/finally，不碰 busy =====
                if kind == "stream":
                    try:
                        self.output.configure(state=tk.NORMAL)
                        self.output.insert(tk.END, payload)
                        self._trim_output()
                        self.output.configure(state=tk.DISABLED)
                        self.output.see(tk.END)
                        self.streamed_this_turn = True
                    except tk.TclError:
                        return
                    continue

                if kind == "status":
                    # 过程性状态（如"自动追加步数"）：只改状态栏，不释放 busy
                    try:
                        self.status.set(payload)
                    except tk.TclError:
                        return
                    continue

                self.pending_thread = None
                try:
                    if kind == "reply":
                        if self.streamed_this_turn:
                            # 流式阶段已画完整正文，reply 只补空行 + 改状态
                            try:
                                self.output.configure(state=tk.NORMAL)
                                self.output.insert(tk.END, "\n\n")
                                self.output.configure(state=tk.DISABLED)
                                self.output.see(tk.END)
                            except tk.TclError:
                                return
                        else:
                            # 流式阶段无任何内容（如全部轮次 content 都为空），
                            # 走 _say 显示最终的 "（无内容）" 或异常兜底文本
                            self._say("助手", payload)
                        self.status.set("就绪 · Ctrl+Enter 发送")
                    else:
                        self._say("错误", payload)
                        self.status.set("调用出错 · 详见上方")
                except tk.TclError:
                    return
                finally:
                    try:
                        self._set_busy(False)
                    except tk.TclError:
                        return
                try:
                    self.input.focus_set()
                except tk.TclError:
                    return
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(POLL_MS, self._poll)
            except tk.TclError:
                pass


def main():
    root = tk.Tk()
    CommandConsole(root)
    root.mainloop()


if __name__ == "__main__":
    main()
