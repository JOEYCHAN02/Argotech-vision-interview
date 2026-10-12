# 01 OpenCV 动态二值化与形状识别

本项目包含两份可以独立运行的代码：基础版直接完成题目的四项基础要求，进阶版在基础流程上增加去噪、形态学处理、轮廓检测和几何形状识别。

面试时建议先运行基础版解释核心原理，再运行进阶版展示扩展能力。

## 两个版本

### 1. 基础版 `basic_shape_demo.py`

1. 读取固定测试视频；
2. 将每一帧彩色图像转换为灰度图；
3. 对每一帧重新运行 Otsu，自动计算动态阈值；
4. 显示原始画面、原始二值图和前景结果；
5. 在中间画面实时显示本帧的 Otsu 阈值。

基础版有意保留未经形态学处理的二值图，便于直接观察全局动态阈值的效果和局限。

### 2. 进阶版 `advanced_shape_demo.py`

1. 估计并抑制渐变背景；
2. 使用动态阈值生成前景；
3. 使用开运算、闭运算去噪并连接边缘；
4. 查找图像外部轮廓；
5. 根据顶点数量、圆度、填充率和宽高比识别圆形、三角形、正方形和长方形；
6. 在原图中绘制绿色轮廓、红色识别框和类别名称。

## 文件结构

```text
01_opencv_shape/
├── .gitignore
├── README.md
├── requirements.txt
├── media/
│   └── shapes_test.mp4
├── results/
│   ├── basic_demo.mp4
│   └── advanced_demo.mp4
└── src/
    ├── basic_shape_demo.py
    └── advanced_shape_demo.py
```

## 环境要求

- Windows 10/11、Linux 或 macOS；
- Python 3.10～3.12；
- NumPy 1.26 及以上版本；
- OpenCV 4.10 及以上版本。

## 安装依赖

在本项目目录打开 PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux 或 macOS 激活环境时使用：

```bash
source .venv/bin/activate
```

## 运行基础版

```powershell
python src\basic_shape_demo.py
```

窗口从左到右显示：

- `Original`：原始彩色视频；
- `Binary Otsu=...`：本帧的原始二值图和动态阈值；
- `Foreground result`：二值掩膜作用于原图后的结果。

## 运行进阶版

```powershell
python src\advanced_shape_demo.py
```

窗口从左到右显示原始画面、清理后的二值图和最终识别结果。右侧会显示轮廓、识别框以及 `circle`、`triangle`、`square`、`rectangle` 类别名称。

两版程序均可按键盘 `Q` 提前退出。

## 保存结果视频

```powershell
python src\basic_shape_demo.py --no-display --save-video results\basic_demo.mp4
python src\advanced_shape_demo.py --no-display --save-video results\advanced_demo.mp4
```

也可以替换成自己的视频：

```powershell
python src\basic_shape_demo.py --source "其他视频路径.mp4"
python src\advanced_shape_demo.py --source "其他视频路径.mp4"
```

## 基础版核心原理

```python
threshold, binary = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

这里的 `0` 不是实际使用的固定阈值。加入 `THRESH_OTSU` 后，OpenCV 会根据当前帧的灰度直方图自动计算 `threshold`，因此每一帧都可以使用独立的动态阈值。

由于 Otsu 使用单个全局阈值，渐变光照可能使部分背景被划分为前景。这属于算法特性；进阶版展示了对复杂背景进行进一步处理的一种方法。
