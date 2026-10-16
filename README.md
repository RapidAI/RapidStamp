# RapidStamp

CPU 上的印章检测与识别。覆盖中文圆形公章、椭圆章、方形名章、财务章、发票章，以及外文公司章、公证章和矩形签章。训练在 PyTorch，部署是两个 ONNX 模型加一份几何后处理，C API 和 Python 包共用同一套权重。

参数量远低于 1B。部署不使用视觉语言模型：Qwen 能读图的官方型号都超过 1B，1B 以内的 Florence-2 / SmolVLM 一类在 CPU 上仍然是生成式运行时。对比和数据许可写在 [docs/research.md](docs/research.md)。

## 模型

| 部分 | 做什么 | 体积 |
| --- | --- | --- |
| SealDet | 320 输入、20×20 单尺度检测，8 类 | 453,197 参数，ONNX 约 1.8MB |
| 几何 | 圆/椭圆极坐标展开，方章按格切开，矩形取水平带 | 无参数 |
| SealRec | 灰度 32×384，CNN-CTC，无 RNN | 1,501,326 参数，ONNX 约 5.8MB |

合成验证集上的 mAP、端到端标题召回和 CPU 延迟在训练后写入 `models/metrics.json`。那是合成 hold-out，不是 ICDAR ReST 的成绩。

## 目录

- `training/`：合成数据、检测和识别训练、ONNX 导出。字体子集在 `training/fonts/`。
- `cpp/`：C API、与 Python 对齐的几何、ONNX Runtime 会话。编译方式见 [docs/sdk.md](docs/sdk.md)。
- `python/`：PyPI 包 `rapidstamp`。
- `.github/workflows/sdk.yml`：Python 测试、Linux / Windows / macOS 的 C++ 构建、Android arm64 交叉编译、cibuildwheel。

## 训练

在有 GPU 的机器上：

```bash
python training/scripts/train_all.py --out /data/rapidstamp --models models
```

默认合成 10000 张训练页和 800 张验证页，检测训 36 个 epoch，识别训 24 个 epoch。`--fast` 只跑几十页、一个 epoch，用来确认合成、训练、导出和评测能串起来。数据、`*.pt` 不进 git；导出的 ONNX 和 `metrics.json` 放在 `models/`。

真实印章包（ReST、PaddleOCR）和 checkpoint 放在训练机的 `/home/mayong`，准备方式见 [docs/data.md](docs/data.md)。`wang-zhix/seal_generate` 没有许可证，源码和它自带的专有字体不进入这个仓库；窄字和底部数字是在自己的绘制代码里做的，说明见 [docs/research.md](docs/research.md)。

## 推理

```bash
pip install -e python
rapidstamp page.jpg --model-dir models
```

C API：

```c
rapidstamp_engine* engine = rapidstamp_create(
    "models/seal_det.onnx", "models/seal_rec.onnx", "models/charset.txt", NULL);
rapidstamp_seal seals[16];
int n = rapidstamp_infer(engine, rgb, width, height, width * 3, seals, 16);
rapidstamp_destroy(engine);
```

桌面、Android 和 iOS 的编译说明在 [docs/sdk.md](docs/sdk.md)。ONNX Runtime 固定 1.22.0，因为这一版同时提供 Linux x64/arm64、Windows x64/arm64 和 macOS arm64/x86_64。

## 许可证

代码是 Apache-2.0。中文字体子集来自文泉驿微米黑，见 `training/fonts/NOTICE`。公开印章数据集的许可和再分发限制见研究报告，图像不放进这个仓库。
