"""OpenCV 基础题：读取视频，逐帧动态二值化并展示结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 获取项目根目录，避免依赖电脑上的绝对路径。
DEFAULT_VIDEO = PROJECT_ROOT / "media" / "shapes_test.mp4"  # 设置仓库内测试视频的默认路径。


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description="OpenCV 逐帧 Otsu 二值化演示")  # 创建参数解析器。
    parser.add_argument("--source", type=Path, default=DEFAULT_VIDEO, help="输入视频路径")  # 允许替换测试视频。
    parser.add_argument("--save-video", type=Path, help="可选：保存三画面结果视频")  # 设置可选输出路径。
    parser.add_argument("--no-display", action="store_true", help="不打开窗口，只保存结果")  # 支持无窗口验证。
    return parser.parse_args()  # 返回所有参数。


def add_title(image: cv2.Mat, text: str) -> None:
    """在画面左上角写出区域名称。"""
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 绘制英文标题。


def main() -> int:
    """读取视频并逐帧完成动态二值化。"""
    args = parse_args()  # 读取用户参数。
    source = args.source.resolve()  # 将输入路径转换为明确的绝对路径。
    if not source.is_file():  # 检查输入视频是否存在。
        raise SystemExit(f"找不到输入视频：{source}")  # 给出清晰的中文错误提示。

    capture = cv2.VideoCapture(str(source))  # 使用 OpenCV 打开视频。
    if not capture.isOpened():  # 检查视频是否成功打开。
        raise SystemExit(f"无法打开输入视频：{source}")  # 打开失败时停止程序。

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0  # 读取帧率，读取失败时使用 20 FPS。
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 读取单帧宽度。
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 读取单帧高度。
    writer = None  # 默认不保存输出视频。

    if args.save_video:  # 判断用户是否要求保存结果。
        output = args.save_video.resolve()  # 获取明确的输出路径。
        output.parent.mkdir(parents=True, exist_ok=True)  # 自动创建输出文件夹。
        codec = cv2.VideoWriter_fourcc(*"mp4v")  # 选择通用 MP4 编码器。
        writer = cv2.VideoWriter(str(output), codec, fps, (width * 3, height))  # 创建三画面视频写入器。
        if not writer.isOpened():  # 检查输出视频是否成功创建。
            raise SystemExit(f"无法创建输出视频：{output}")  # 创建失败时停止程序。

    try:  # 确保程序退出时释放所有资源。
        while True:  # 循环处理视频中的每一帧。
            success, frame = capture.read()  # 读取下一帧彩色画面。
            if not success:  # 判断视频是否播放结束。
                break  # 读完后结束循环。

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前彩色帧转换为灰度图。
            threshold, binary = cv2.threshold(  # 对当前帧单独执行 Otsu 动态二值化。
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU  # 阈值写 0，由 Otsu 自动计算实际阈值。
            )  # 得到本帧阈值和黑白二值图。
            result = cv2.bitwise_and(frame, frame, mask=binary)  # 使用二值掩膜保留白色前景区域。

            original_panel = frame.copy()  # 复制原始画面用于添加标题。
            binary_panel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # 将单通道二值图转成三通道方便拼接。
            add_title(original_panel, "Original")  # 标出左侧原始画面。
            add_title(binary_panel, f"Binary  Otsu={threshold:.0f}")  # 显示中间二值图及本帧动态阈值。
            add_title(result, "Foreground result")  # 标出右侧前景结果。
            display = cv2.hconcat((original_panel, binary_panel, result))  # 横向拼接三个画面。

            if writer is not None:  # 判断是否需要保存当前结果。
                writer.write(display)  # 将当前拼接画面写入视频。
            if not args.no_display:  # 判断是否需要打开实时窗口。
                cv2.imshow("Original | Dynamic binary | Result", display)  # 实时显示三个处理阶段。
                wait_time = max(1, round(1000 / fps))  # 根据视频帧率计算播放等待时间。
                if cv2.waitKey(wait_time) & 0xFF == ord("q"):  # 检查用户是否按下 Q 键。
                    break  # 按下 Q 后提前退出。
    finally:  # 无论正常结束还是发生异常都执行清理。
        capture.release()  # 释放输入视频。
        if writer is not None:  # 判断输出视频写入器是否存在。
            writer.release()  # 完成写入并关闭输出视频。
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口。

    print("处理完成。")  # 在终端提示任务完成。
    if args.save_video:  # 判断是否保存了结果视频。
        print(f"结果视频：{args.save_video.resolve()}")  # 输出结果文件位置。
    return 0  # 返回成功状态码。


if __name__ == "__main__":  # 只在直接运行本文件时执行主函数。
    raise SystemExit(main())  # 启动程序并把状态码返回给系统。

