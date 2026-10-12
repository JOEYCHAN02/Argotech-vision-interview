"""OpenCV 进阶版：形态学处理、轮廓检测和几何形状识别。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 获取项目根目录，避免依赖电脑上的绝对路径。
DEFAULT_VIDEO = PROJECT_ROOT / "media" / "shapes_test.mp4"  # 设置仓库内测试视频的默认路径。


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description="OpenCV 进阶形状识别演示")  # 创建命令行参数解析器。
    parser.add_argument("--source", type=Path, default=DEFAULT_VIDEO, help="输入视频路径")  # 设置输入视频。
    parser.add_argument("--threshold", choices=("otsu", "adaptive"), default="otsu", help="二值化方法")  # 选择动态阈值方法。
    parser.add_argument("--save-video", type=Path, help="可选：保存三窗口拼接结果")  # 设置可选输出视频路径。
    parser.add_argument("--no-display", action="store_true", help="只保存视频，不打开窗口")  # 支持无窗口验证。
    return parser.parse_args()  # 返回解析后的参数。


def make_binary(frame: cv2.Mat, method: str) -> cv2.Mat:
    """先去除渐变背景，再使用动态阈值生成目标前景。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将彩色帧转换为灰度图，满足题目要求。
    background = cv2.GaussianBlur(frame, (51, 51), 0)  # 估计平滑背景，减少渐变造成的误检。
    gray_background = cv2.GaussianBlur(gray, (51, 51), 0)  # 对灰度图估计对应的平滑背景。
    color_difference = cv2.absdiff(frame, background)  # 计算当前图像与背景的颜色差异。
    gray_difference = cv2.absdiff(gray, gray_background)  # 计算当前灰度图与背景的亮度差异。
    contrast = np.maximum(np.max(color_difference, axis=2), gray_difference)  # 融合颜色差异和灰度差异。
    if method == "otsu":  # 判断是否使用 Otsu 阈值。
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 根据差异分布自动求阈值。
    else:
        binary = cv2.adaptiveThreshold(  # 在背景差异图上使用局部区域计算阈值。
            contrast,  # 传入颜色差异图。
            255,  # 设置白色前景值。
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 使用高斯加权局部平均。
            cv2.THRESH_BINARY,  # 先得到亮区域掩膜。
            51,  # 设置较大的局部计算窗口，降低背景噪声。
            2,  # 调整局部阈值偏移量。
        )  # 返回自适应阈值结果。
        white_ratio = cv2.countNonZero(binary) / binary.size  # 计算白色区域占整幅图的比例。
        if white_ratio > 0.5:  # 判断自适应阈值是否把背景误判成大面积前景。
            binary = cv2.bitwise_not(binary)  # 反转极性，让目标区域成为前景。
    return binary  # 返回动态阈值结果。


def clean_binary(binary: cv2.Mat) -> cv2.Mat:
    """使用开运算和闭运算去除噪声并连接断裂区域。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 创建椭圆形结构元素。
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算去除小白色噪点。
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算填补小黑洞并连接边缘。
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找目标最外层轮廓。
    filled = np.zeros_like(closed)  # 创建黑色画布作为填充结果。
    valid = [contour for contour in contours if cv2.contourArea(contour) >= 1200]  # 忽略小面积噪点。
    cv2.drawContours(filled, valid, -1, 255, cv2.FILLED)  # 填充有效轮廓，让二值展示呈现实心目标。
    return filled  # 返回去噪并填充后的二值图。


def classify_contour(contour: cv2.Mat) -> tuple[str, tuple[int, int, int, int]] | None:
    """根据轮廓几何特征返回类别和外接框。"""
    area = cv2.contourArea(contour)  # 计算轮廓面积。
    if area < 1200:  # 过滤文字和小噪声轮廓。
        return None  # 面积太小时不作为目标。
    perimeter = cv2.arcLength(contour, True)  # 计算轮廓周长。
    if perimeter <= 0:  # 防止后续计算除以零。
        return None  # 无效轮廓直接忽略。
    approximation = cv2.approxPolyDP(contour, 0.03 * perimeter, True)  # 用多边形近似减少轮廓点数量。
    x, y, width, height = cv2.boundingRect(contour)  # 计算轴对齐外接矩形。
    rotated_box = cv2.minAreaRect(contour)  # 获取旋转外接矩形，计算目标的几何填充率。
    box_width, box_height = rotated_box[1]  # 读取旋转矩形的宽和高。
    rectangle_area = max(box_width * box_height, 1.0)  # 防止退化轮廓造成除零。
    extent = area / rectangle_area  # 三角形填充率约为 0.65，四边形接近 0.9 以上。
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)  # 计算圆度，圆形通常接近 0.8。
    if circularity >= 0.78 and extent <= 0.86:  # 同时满足圆度和填充率时判定为圆形。
        return "circle", (x, y, width, height)  # 返回圆形类别和外接框。
    if extent < 0.72:  # 三角形在旋转外接矩形中的面积比例明显更小。
        return "triangle", (x, y, width, height)  # 返回三角形类别和外接框。
    if len(approximation) == 4:  # 四个顶点对应四边形。
        ratio = min(box_width, box_height) / max(box_width, box_height, 1.0)  # 计算四边形的短边长边比例。
        label = "square" if ratio >= 0.78 else "rectangle"  # 宽高接近正方形，否则判断为长方形。
        return label, (x, y, width, height)  # 返回四边形类别和外接框。
    if extent >= 0.82:  # 四边形边缘轻微锯齿时近似顶点可能多于四个。
        ratio = min(box_width, box_height) / max(box_width, box_height, 1.0)  # 根据旋转框判断正方形或长方形。
        label = "square" if ratio >= 0.78 else "rectangle"  # 宽高接近正方形，否则判断为长方形。
        return label, (x, y, width, height)  # 返回四边形类别和外接框。
    return None  # 其他复杂轮廓不强行分类。


def draw_detections(frame: cv2.Mat, contours: list[cv2.Mat]) -> cv2.Mat:
    """在原始图像上绘制轮廓、框和类别名称。"""
    result = frame.copy()  # 复制原始帧，避免破坏原始画面。
    for contour in contours:  # 逐个处理检测到的轮廓。
        detection = classify_contour(contour)  # 根据几何特征进行分类。
        if detection is None:  # 判断该轮廓是否应该忽略。
            continue  # 跳过小噪声或无法分类的轮廓。
        label, (x, y, width, height) = detection  # 拆出类别名称和外接框。
        if x <= 1 or y <= 1 or x + width >= frame.shape[1] - 1 or y + height >= frame.shape[0] - 1:  # 过滤贴着画面边缘的背景伪轮廓。
            continue  # 边缘伪轮廓通常不是测试目标。
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)  # 在原始图像上绘制绿色轮廓。
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 0, 255), 2)  # 绘制红色识别框。
        cv2.putText(result, label, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 写出类别名称。
    return result  # 返回带识别结果的图像。


def main() -> int:
    """逐帧执行完整的 OpenCV 形状识别流程。"""
    args = parse_args()  # 读取命令行参数。
    if not args.source.is_file():  # 检查视频路径。
        raise SystemExit(f"找不到输入视频：{args.source.resolve()}")  # 提示输入文件不存在。
    capture = cv2.VideoCapture(str(args.source))  # 打开视频文件。
    if not capture.isOpened():  # 检查视频是否打开成功。
        raise SystemExit(f"无法打开视频：{args.source.resolve()}")  # 提示 OpenCV 打开失败。

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0  # 读取帧率。
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 读取宽度。
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 读取高度。
    writer = None  # 初始化输出视频对象。
    if args.save_video:  # 判断是否保存拼接视频。
        args.save_video.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录。
        writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height))  # 创建输出视频。
        if not writer.isOpened():  # 检查输出视频是否创建成功。
            raise SystemExit(f"无法创建输出视频：{args.save_video.resolve()}")  # 提示输出失败。

    try:  # 确保视频和窗口最终被释放。
        while True:  # 循环处理所有视频帧。
            success, frame = capture.read()  # 读取当前彩色帧。
            if not success:  # 判断是否读取结束。
                break  # 视频结束后退出循环。
            binary = make_binary(frame, args.threshold)  # 使用背景差异和动态阈值生成二值图。
            cleaned = clean_binary(binary)  # 对二值图进行去噪和形态学处理。
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找外部轮廓。
            result = draw_detections(frame, contours)  # 在原始帧上绘制轮廓、框和类别。
            cv2.putText(result, "Advanced shape recognition", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 标记进阶结果窗口。
            cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)  # 将清理后的二值图转换为三通道。
            display = cv2.hconcat((frame, cleaned_bgr, result))  # 拼接原始、二值和最终识别结果。
            if writer is not None:  # 判断是否需要保存当前帧。
                writer.write(display)  # 写入输出视频。
            if not args.no_display:  # 判断是否显示窗口。
                cv2.imshow("Advanced OpenCV Demo: Original | Binary | Result", display)  # 显示三个结果区域。
                if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF == ord("q"):  # 等待按键并检测 Q 键。
                    break  # 按 Q 后退出。
    finally:  # 统一释放资源。
        capture.release()  # 释放输入视频。
        if writer is not None:  # 判断输出对象是否存在。
            writer.release()  # 关闭输出视频。
        cv2.destroyAllWindows()  # 关闭 OpenCV 窗口。

    print("进阶版处理完成。")  # 输出完成提示。
    if args.save_video:  # 判断是否保存了结果视频。
        print(f"结果视频：{args.save_video.resolve()}")  # 输出结果路径。
    return 0  # 返回成功状态码。


if __name__ == "__main__":  # 判断文件是否被直接执行。
    raise SystemExit(main())  # 执行主函数。
