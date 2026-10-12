"""使用最佳权重在独立测试集上评估 YOLO11n。"""

from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 获取项目根目录。


def main() -> int:
    """加载最佳模型并评估 test 数据集。"""
    model_path = PROJECT_ROOT / "models" / "best.pt"  # 指定正式训练的最佳权重。
    data_path = PROJECT_ROOT / "data.yaml"  # 指定数据集配置文件。
    if not model_path.is_file():  # 检查模型权重。
        raise FileNotFoundError(f"找不到模型权重：{model_path}")  # 缺失时给出清晰提示。
    device = 0 if torch.cuda.is_available() else "cpu"  # 有 CUDA 时使用 GPU，否则回退到 CPU。
    model = YOLO(str(model_path))  # 加载训练完成的 YOLO11n。
    model.val(  # 在完全独立的测试集上计算指标。
        data=str(data_path),  # 使用项目内数据配置。
        split="test",  # 明确选择测试集而非验证集。
        imgsz=640,  # 使用与训练一致的输入尺寸。
        batch=8,  # 使用与训练一致的批大小。
        device=device,  # 使用可用的 GPU 或 CPU。
        project=str(PROJECT_ROOT / "runs"),  # 保存评估输出。
        name="test_evaluation",  # 设置输出目录名称。
        plots=True,  # 生成 PR 曲线和混淆矩阵。
    )
    return 0  # 返回成功状态码。


if __name__ == "__main__":  # 只在直接运行时执行。
    raise SystemExit(main())  # 启动独立测试集评估。

