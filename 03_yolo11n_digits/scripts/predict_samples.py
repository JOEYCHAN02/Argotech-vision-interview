"""使用最佳权重对仓库内示例图片执行推理。"""

from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 获取项目根目录。


def main() -> int:
    """对 samples/images 中的图片批量推理。"""
    model_path = PROJECT_ROOT / "models" / "best.pt"  # 指定最佳模型权重。
    source = PROJECT_ROOT / "samples" / "images"  # 指定演示图片目录。
    if not model_path.is_file():  # 检查权重是否存在。
        raise FileNotFoundError(f"找不到模型权重：{model_path}")  # 缺失时停止程序。
    device = 0 if torch.cuda.is_available() else "cpu"  # 有 CUDA 时使用 GPU，否则使用 CPU 演示。
    model = YOLO(str(model_path))  # 加载训练完成的模型。
    model.predict(  # 批量检测示例图片。
        source=str(source),  # 读取仓库内示例图片。
        imgsz=640,  # 使用与训练一致的输入尺寸。
        conf=0.25,  # 设置预测框的最低置信度。
        device=device,  # 使用可用的 GPU 或 CPU。
        save=True,  # 保存带预测框的结果图。
        project=str(PROJECT_ROOT / "runs"),  # 设置输出根目录。
        name="sample_predictions",  # 设置本次推理目录名。
    )
    return 0  # 返回成功状态码。


if __name__ == "__main__":  # 只在直接运行时执行。
    raise SystemExit(main())  # 启动示例推理。

