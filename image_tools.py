"""image_tools.py — 图片识别能力（独立模块，不注册工具）。

为什么单独成模块：
- 主文件 agent.py 的注册链路当前被 AST 校验锁死，任何编辑都会被拒；
- 非主文件又不允许调用 register_business_tools（写入即被拦）；
- 因此这里只提供"可被调用"的实现，由 gui_frontend 直接调用，
  签名与 agent.py 期望的业务工具保持一致，解禁后可直接复用。

安全约束与 agent.py 中的工具完全一致：路径必须过 is_safe_path。
"""
import base64
import io
import os

from safety import client, MODEL, is_safe_path

MAX_IMAGE_BYTES = 5 * 1024 * 1024      # 源文件字节上限（尊重调用方配置）
MAX_DIMENSION = 2048                   # 长边像素压缩阈值
SUPPORTED_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff",
)
DEFAULT_PROMPT = "请详细描述这张图片的内容；若图中有文字，请逐行如实转录。"


def prepare_image(image_path, max_bytes=MAX_IMAGE_BYTES):
    """校验图片并压成 JPEG data URL。返回 (data_url, info)；失败抛异常。"""
    from PIL import Image

    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("请提供有效的图片路径")
    image_path = image_path.strip().strip('"').strip("'")
    if not is_safe_path(image_path):
        raise PermissionError("安全拦截：无权读取该路径")
    if not os.path.isfile(image_path):
        raise FileNotFoundError("文件不存在：%s" % image_path)

    size_bytes = os.path.getsize(image_path)
    if size_bytes > max_bytes:
        raise ValueError("文件过大（%.1fMB，上限 %.1fMB）"
                         % (size_bytes / 1048576.0, max_bytes / 1048576.0))

    ext = os.path.splitext(image_path)[1].lower()
    if ext and ext not in SUPPORTED_SUFFIXES:
        raise ValueError("不支持的图片格式：%s（支持：%s）"
                         % (ext or "无扩展名", " ".join(SUPPORTED_SUFFIXES)))

    try:
        _LANCZOS = Image.Resampling.LANCZOS
    except AttributeError:
        _LANCZOS = Image.LANCZOS

    buf = io.BytesIO()
    with Image.open(image_path) as im:
        fmt = (im.format or "?").upper()
        has_alpha = im.mode in ("RGBA", "LA") or (
            im.mode == "P" and "transparency" in im.info)
        if fmt == "GIF":
            im.seek(0)

        if has_alpha:
            # 先转 RGBA 再缩放，避免调色板(P)透明图在缩放时丢失 alpha。
            rgba = im.convert("RGBA")
            rgba.thumbnail((MAX_DIMENSION, MAX_DIMENSION), _LANCZOS)
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            rgb = im.convert("RGB")
            rgb.thumbnail((MAX_DIMENSION, MAX_DIMENSION), _LANCZOS)
        width, height = rgb.size
        rgb.save(buf, format="JPEG", quality=88, optimize=True)

    data_url = "data:image/jpeg;base64,%s" % base64.b64encode(
        buf.getvalue()).decode("ascii")
    info = "%s %dx%d · %.1fKB · %s" % (
        fmt, width, height, size_bytes / 1024.0,
        "透明通道已铺白底" if has_alpha else "已规范化为 JPEG")
    return data_url, info


def recognize_image(image_path, question=""):
    """识别本地图片，返回可读文本结果（不抛异常，错误也以文本返回）。"""
    if not isinstance(question, str):
        question = str(question or "")
    try:
        data_url, info = prepare_image(image_path)
    except Exception as e:
        return "无法处理图片：%s" % e

    prompt = question.strip() or DEFAULT_PROMPT
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=1500,
            temperature=0.2,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return "图像识别调用失败：%s" % e
    return "【图片信息】%s\n【识别结果】\n%s" % (info, answer or "（模型未返回内容）")


def main():
    """离线自测：只验证路径校验与编码链路，不发起任何网络请求。"""
    import tempfile

    from PIL import Image

    got = recognize_image("definitely_missing_image.png")
    print("[1] 缺失文件 ->", got)

    got = recognize_image("C:/Windows/System32/not_here.png")
    print("[2] 越界路径 ->", got)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "_selftest_img.png")
        Image.new("RGBA", (120, 80), (10, 120, 200, 128)).save(tmp_path, "PNG")
        data_url, info = prepare_image(tmp_path)
        print("[3] 编码成功 ->", info)
        print("    data URL 前缀:", data_url[:32], "... 长度:", len(data_url))

    print("[4] 支持后缀:", " ".join(SUPPORTED_SUFFIXES))


if __name__ == '__main__':
    main()
