# RapidStamp

在 CPU 上检测并阅读圆章（公章、发票专用章一类标题）。训练用 Python，推理用 C++。第一阶段交付的是一条能跑通的缩小版：合成数据、小模型、ONNX、不依赖 ONNX Runtime 的 C API，以及 NumPy / ONNX / 原生库三种 Python 后端。

生产架构的对比、数据集许可证和为什么不默认用 Ultralytics YOLO，写在 [docs/research.md](docs/research.md)。

## 推荐方案（生产）

1. 整章定位用 **PicoDet-S**（单类，输入 320 或 416）。约 0.99M 参数、320 输入 0.73 GFLOPs，Apache-2.0，面向 CPU 和移动端。RT-DETRv2-S 太大。Ultralytics YOLO11 / v8 / v10 的代码和官方权重是 AGPL-3.0，不适合闭源 SDK。
2. 用印环几何把圆（以后还有椭圆）拟合出来，极坐标展开成水平条。
3. 识别用 **SVTR-LCNet + CTC**（PP-OCRv4 mobile 那一档，约 11 MB）。机构名词表只做后处理。

公开的 ReST（ICDAR 2023 Reading the Seal Title）有 1 万枚真实印章，但没有可再分发的许可证，本仓库不放它的图片。PaddleOCR 的印章检测权重是 Apache-2.0，它的 500 张自建图没有公开。第一阶段的训练数据全部由 `training/rapidstamp_train/synth.py` 合成。

## 这一阶段实际落地的模型

同级联、可以在 CPU 上几分钟训完的缩小版，用来把导出和 SDK 跑通，不是 PicoDet / SVTR 本身。

| 模块 | 内容 | 体积 |
| --- | --- | --- |
| SealDet-Tiny | 深度可分离卷积，输入 128，输出 8×8 网格上的单类框。7253 参数 | 权重包 28 KB，ONNX 31 KB |
| 圆拟合 | 在检测框附近搜索印环相对内圈的对比度，把圆心拉回印泥上 | 无参数 |
| 展开 | 极坐标展开 + 缺口对齐，条带 32×160 | 无参数 |
| SealRec-Tiny | 32×32 字块分类，51 类，21267 参数。解码在 664 条合成标题上取对数概率 | 权重包 85 KB，加上词表的 manifest 后约 102 KB；ONNX 87 KB |

合成验证集（`models/demo/metrics.json`，不能外推到 ReST）：

- 检测头在阈值 0.7：精度 0.741，召回 0.75，匹配框平均 IoU 0.660。这是拟合圆之前的框。
- 真值框上的词表解码：整串 0.917，字符 0.986。
- 端到端（检测 + 圆拟合 + 识别，32 张干净单章）：检出 1.00，整串 0.906。
- NumPy 后端单页约 25 ms（256×256，含圆拟合和词表）。

检测头的框中心经常偏出半径的两成，直接展开会把字切坏。圆拟合把中心误差收到大约 0.03 倍半径，端到端才读得出来。

## Python

```bash
pip install -e python/
python -c "from rapidstamp import SealEngine; print(SealEngine.from_pretrained('numpy'))"
```

```python
from rapidstamp import SealEngine

engine = SealEngine.from_pretrained("numpy")  # native 或 onnx
for seal in engine.infer(rgb_uint8):
    print(seal.text, seal.score, seal.x, seal.y, seal.w, seal.h)
```

`native` 需要先编出 `cpp/build/librapidstamp.so`（或设置 `RAPIDSTAMP_LIB`）。`onnx` 需要安装 `onnxruntime`，并使用 manifest 旁边的 `.onnx`。

命令行：`rapidstamp path/to/page.png --backend numpy`。

## C API

`cpp/include/rapidstamp.h`

```c
rapidstamp_engine* engine = rapidstamp_create("detector.manifest", "recognizer.manifest");
rapidstamp_box boxes[8];
int n = rapidstamp_infer_rgb(engine, rgb, width, height, stride_bytes, boxes, 8);
```

权重是 manifest 描述的算子加上同名的 float32 `.bin`（小端）。C++ 运行时只实现 conv、ReLU、max-pool、全局平均池化和线性层，不链接 ONNX Runtime，这样 SDK 本身保持很小。ONNX 仍然导出，给其他后端用。

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
cpp/build/rapidstamp_demo path/to/page.ppm models/demo/detector.manifest models/demo/recognizer.manifest
```

GitHub Actions（`.github/workflows/ci.yml`）在 Linux、Windows、macOS 上编共享库，另有 Linux arm64、aarch64 交叉编译、macOS universal、Android `arm64-v8a` / `x86_64` 和 iOS arm64 静态库。wheel 用 cibuildwheel（`.github/workflows/wheels.yml`），Python 包本身是纯 NumPy，原生库按需加载。

## 训练

```bash
pip install -r training/requirements.txt
PYTHONPATH=python/src:training python -m rapidstamp_train.train --out models/demo
```

默认在 CPU 上用合成圆章训练演示权重，并写到 `models/demo/` 和 `python/src/rapidstamp/models/`。字体是 Droid Sans Fallback 的子集，见 `assets/fonts/NOTICE.md`。

## 测试

```bash
PYTHONPATH=python/src:training python -m pytest -q
```

## 目录

- `docs/research.md`：候选模型、数据集许可证、推荐方案
- `training/`：合成、训练、导出 ONNX 和 RSP 权重包
- `cpp/`：C API 和解释器
- `python/`：PyPI 包
- `models/demo/`：已训练的演示权重
- `.github/workflows/`：多平台 SDK 与 wheel

## 许可证

Apache-2.0。见 [LICENSE](LICENSE)。
