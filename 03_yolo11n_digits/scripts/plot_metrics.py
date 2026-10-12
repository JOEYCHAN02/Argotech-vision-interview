"""根据 Ultralytics 的 results.csv 生成适合 GitHub 展示的指标曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 获取项目根目录，避免依赖本机绝对路径。
DEFAULT_CSV = PROJECT_ROOT / "records" / "results.csv"  # 设置正式训练日志的默认位置。
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "metrics"  # 设置指标图片的默认输出目录。


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)  # 创建命令行解析器。
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="训练结果 CSV")  # 允许替换训练日志。
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="曲线输出目录")  # 允许替换输出目录。
    return parser.parse_args()  # 返回解析结果。


def configure_style() -> None:
    """统一所有曲线的显示风格。"""
    plt.rcParams.update(  # 设置适合屏幕和 GitHub 的绘图参数。
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_single_metric(data: pd.DataFrame, column: str, title: str, ylabel: str, output: Path, best_epoch: int) -> None:
    """保存单个随 epoch 变化的指标。"""
    fig, axis = plt.subplots(figsize=(9, 5.2))  # 创建一张独立曲线图。
    axis.plot(data["epoch"], data[column], linewidth=2.2)  # 绘制每一轮的指标值。
    axis.axvline(best_epoch, color="0.4", linestyle="--", label=f"Best epoch: {best_epoch}")  # 标出最佳轮次。
    axis.set(title=title, xlabel="Epoch", ylabel=ylabel, xlim=(1, int(data["epoch"].max())))  # 设置标题和坐标轴。
    axis.set_ylim(0, 1.02)  # Precision、Recall 和 mAP 的理论范围为 0 到 1。
    axis.legend()  # 显示最佳轮次图例。
    fig.tight_layout()  # 自动调整画布边距。
    fig.savefig(output, bbox_inches="tight")  # 保存高分辨率 PNG。
    plt.close(fig)  # 释放绘图资源。


def save_map(data: pd.DataFrame, output: Path, best_epoch: int) -> None:
    """将 mAP50 和 mAP50-95 绘制在同一张图中。"""
    fig, axis = plt.subplots(figsize=(9, 5.2))  # 创建 mAP 图。
    axis.plot(data["epoch"], data["metrics/mAP50(B)"], label="mAP@0.50", linewidth=2.2)  # 绘制宽松 IoU 指标。
    axis.plot(data["epoch"], data["metrics/mAP50-95(B)"], label="mAP@0.50:0.95", linewidth=2.2)  # 绘制严格综合指标。
    axis.axvline(best_epoch, color="0.4", linestyle="--", label=f"Best epoch: {best_epoch}")  # 标出最佳轮次。
    axis.set(title="Mean Average Precision vs. Epoch", xlabel="Epoch", ylabel="mAP", xlim=(1, int(data["epoch"].max())), ylim=(0, 1.02))  # 设置坐标轴。
    axis.legend(loc="lower right")  # 显示图例。
    fig.tight_layout()  # 调整布局。
    fig.savefig(output, bbox_inches="tight")  # 保存图片。
    plt.close(fig)  # 释放资源。


def save_losses(data: pd.DataFrame, output: Path, best_epoch: int) -> None:
    """绘制训练集和验证集的 Box、Classification、DFL loss。"""
    items = (  # 定义三类损失及其 CSV 字段。
        ("Box loss", "train/box_loss", "val/box_loss"),
        ("Classification loss", "train/cls_loss", "val/cls_loss"),
        ("DFL loss", "train/dfl_loss", "val/dfl_loss"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)  # 创建三个并列子图。
    for axis, (title, train_column, val_column) in zip(axes, items):  # 逐类绘制训练和验证损失。
        axis.plot(data["epoch"], data[train_column], label="Train", linewidth=2)  # 绘制训练损失。
        axis.plot(data["epoch"], data[val_column], label="Validation", linewidth=2)  # 绘制验证损失。
        axis.axvline(best_epoch, color="0.4", linestyle="--", linewidth=1)  # 标出最佳轮次。
        axis.set(title=title, xlabel="Epoch", ylabel="Loss", xlim=(1, int(data["epoch"].max())))  # 设置子图信息。
        axis.set_ylim(bottom=0)  # Loss 从 0 开始显示。
        axis.legend()  # 显示训练集和验证集图例。
    fig.suptitle("YOLO11n Training and Validation Losses", fontsize=16)  # 设置总标题。
    fig.tight_layout()  # 调整布局。
    fig.savefig(output, bbox_inches="tight")  # 保存图片。
    plt.close(fig)  # 释放资源。


def save_overview(data: pd.DataFrame, output: Path, best_epoch: int) -> None:
    """生成四项验证指标总览图。"""
    plots = (  # 定义四个总览子图。
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)"),
        ("mAP@0.50", "metrics/mAP50(B)"),
        ("mAP@0.50:0.95", "metrics/mAP50-95(B)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))  # 创建 2×2 总览画布。
    for axis, (title, column) in zip(axes.flat, plots):  # 依次绘制四个指标。
        axis.plot(data["epoch"], data[column], linewidth=2.2)  # 绘制指标曲线。
        axis.axvline(best_epoch, color="0.4", linestyle="--", linewidth=1)  # 标出最佳轮次。
        axis.set(title=title, xlabel="Epoch", ylabel="Score", xlim=(1, int(data["epoch"].max())), ylim=(0, 1.02))  # 设置子图。
    fig.suptitle(f"YOLO11n Validation Metrics (best epoch: {best_epoch})", fontsize=16)  # 设置总标题。
    fig.tight_layout()  # 调整布局。
    fig.savefig(output, bbox_inches="tight")  # 保存图片。
    plt.close(fig)  # 释放资源。


def main() -> int:
    """读取正式训练日志并生成全部指标图片。"""
    args = parse_args()  # 读取命令行参数。
    csv_path = args.csv.resolve()  # 获取明确的 CSV 路径。
    output = args.output.resolve()  # 获取明确的输出目录。
    if not csv_path.is_file():  # 检查训练日志是否存在。
        raise FileNotFoundError(f"找不到训练日志：{csv_path}")  # 缺少日志时给出清晰提示。
    output.mkdir(parents=True, exist_ok=True)  # 创建指标图片目录。
    data = pd.read_csv(csv_path)  # 读取 100 轮训练记录。
    data.columns = [column.strip() for column in data.columns]  # 清理列名可能存在的空格。
    best_epoch = int(data.loc[data["metrics/mAP50-95(B)"].idxmax(), "epoch"])  # 按严格 mAP 选择最佳轮次。
    configure_style()  # 应用统一绘图样式。
    save_single_metric(data, "metrics/precision(B)", "Precision vs. Epoch", "Precision", output / "01_precision_vs_epoch.png", best_epoch)  # 保存 Precision。
    save_single_metric(data, "metrics/recall(B)", "Recall vs. Epoch", "Recall", output / "02_recall_vs_epoch.png", best_epoch)  # 保存 Recall。
    save_map(data, output / "03_map_vs_epoch.png", best_epoch)  # 保存 mAP。
    save_losses(data, output / "04_loss_vs_epoch.png", best_epoch)  # 保存三类 Loss。
    save_overview(data, output / "05_metrics_overview.png", best_epoch)  # 保存指标总览。
    print(f"已生成 5 张训练指标图：{output}")  # 输出完成信息。
    print(f"最佳轮次：{best_epoch}")  # 输出最佳轮次。
    return 0  # 返回成功状态码。


if __name__ == "__main__":  # 仅在直接运行本文件时启动程序。
    raise SystemExit(main())  # 执行主函数并返回状态码。

