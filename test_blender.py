# -*- coding: utf-8 -*-
"""Blender 测试脚本：创建红色立方体并保存为 .blend 文件。

用法：
  1) Blender 内置解释器：
     blender --background --python test_blender.py
  2) 普通 Python / unittest：用例自动 skip（bpy 不可用），不会把测试套件拖红。
"""

import os
import unittest

try:  # 该脚本只能在 Blender 内置解释器中真正执行；普通 Python 下 bpy 不可用。
    import bpy
except ImportError:  # pragma: no cover - 非 Blender 环境
    bpy = None

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "test_cube.blend")


def create_red_cube() -> str:
    """创建红色立方体并保存 .blend，返回输出路径。"""
    # 1. 清空当前场景中的物体
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # 2. 创建默认立方体
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "TestCube"

    # 3. 创建并绑定红色材质
    mat = bpy.data.materials.new(name="RedMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5

    if cube.data.materials:
        cube.data.materials[0] = mat
    else:
        cube.data.materials.append(mat)

    # 4. 保存到指定路径
    out_dir = os.path.dirname(OUT_PATH)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.save_as_mainfile(filepath=OUT_PATH)
    return OUT_PATH


class TestBlenderCube(unittest.TestCase):
    def test_create_red_cube(self):
        if bpy is None:
            self.skipTest("需要 Blender 内置解释器（bpy 不可用）")
        path = create_red_cube()
        self.assertTrue(os.path.isfile(path), f"未生成 .blend 文件: {path}")


def main():
    if bpy is None:
        print("⚠️ 未检测到 bpy，本脚本需在 Blender 解释器中运行：")
        print("   blender --background --python test_blender.py")
        return
    path = create_red_cube()
    print("=" * 50)
    print(f"✅ 已创建红色立方体并保存: {path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
