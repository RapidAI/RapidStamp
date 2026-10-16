# 印章检测与识别：架构、数据与推荐方案

本文档对应仓库第一阶段。目标是一条能上 CPU 的印章阅读链路：**小**（权重以 MB 计，而不是上百 MB）、**准**（先把圆章标题读对，再扩到椭圆/方章和真实扫描件）、**快**（单页 CPU 推理以数十毫秒为量级，而不是秒级）。训练用 Python，推理用 C++，交换格式是 ONNX，同时提供不依赖 ONNX Runtime 的小型权重包，方便把 SDK 做到各平台。

下面的体积、FLOPs、COCO mAP 和延迟都来自各论文或官方文档，**不是印章任务上的数字**。印章是大目标、类别少，COCO mAP 偏低的轻量检测器往往够用；反过来，ReST 上 90% 以上的端到端准确率目前是用大模型和大量合成数据做出来的，不能直接当成 CPU 预算。

## 1. 任务怎么拆

一次印章阅读其实是三件不同的事：

1. **印章实例检测**：在合同、发票、扫描页上框出公章、私章、发票专用章。一页通常只有一到数个，目标大、颜色往往偏红或蓝，但灰度复印件、浅印、重叠章会把颜色先验打掉。
2. **文本行几何**：圆章和椭圆章的字沿环排列，每个字的字头朝外；方章是网格；中间常有横排文字（「发票专用章」）或星标。这一步的输出是若干条水平文字条。
3. **文字识别**：在拉直后的条带上读字。公章标题是机构名，字符集大、字体接近印刷体或专用印章字体，但盖章会带来残缺、糊边和旋转。

PaddleOCR 的 `PP-OCRv4_*_seal_det` 做的是第 2 步里的**弯曲文本行检测**（输出多边形），不是第 1 步的整章定位。文档里的版面模型（`PicoDet-S_layout_*`，类别里包含 `seal`）才是整章定位。两套模型不要混用指标。

## 2. 检测模型

| 模型 | 输入 | 参数量 | 计算量 | 公开精度 | 延迟（注明平台） | 权重体积 | 许可证 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP-PicoDet-S | 320 | 0.99M | 0.73 GFLOPs | COCO mAP 27.1（0.5:0.95），mAP50 41.4 | 骁龙 865 四线程 8.13 ms（论文另给单线程约 6.65 ms 的对照列） | 约 4 MB 量级（Paddle 部署包；`PicoDet-S_layout_3cls` 官方列为 4.8 MB） | Apache-2.0（PaddleDetection） |
| PP-PicoDet-S | 416 | 0.99M | 1.24 GFLOPs | COCO mAP 30.6 | 骁龙 865 四线程 12.37 ms | 同上 | Apache-2.0 |
| PP-PicoDet-L | 640 | 3.30M | 8.91 GFLOPs | COCO mAP 40.9 | 骁龙 865 四线程 54.11 ms | 更大 | Apache-2.0 |
| NanoDet-M | 320 | 0.95M | 0.72 GFLOPs | COCO mAP 20.6 | 论文表中 8.71 ms（同 PicoDet 的移动端设置） | 约 1–2 MB | NanoDet 仓库为 Apache-2.0 |
| YOLOv8n | 640 | 3.2M | 8.7 GFLOPs | COCO mAP 37.3 | Ultralytics 文档：ONNX CPU 80.4 ms | FP32 约 6 MB | Ultralytics 代码与官方权重为 AGPL-3.0，闭源嵌入需要商业授权 |
| YOLO11n | 640 | 2.6M | 6.5 GFLOPs | COCO mAP 39.5 | Ultralytics 文档：ONNX CPU 56.1 ms | 同量级 | 同上，AGPL-3.0 |
| YOLOv10n | 640 | 2.3M | 6.7 GFLOPs | COCO mAP 约 38（官方表） | CPU 与 YOLO11n 同级 | 同量级 | Ultralytics 发行版为 AGPL-3.0 |
| RT-DETRv2-S | 640 | 20M | 60 GFLOPs | COCO mAP 48.1 | TensorRT T4 5.03 ms；**官方表没有 CPU 数字**。同尺寸 Transformer 检测器在 CPU 上通常是数百毫秒 | 约 40 MB 以上 | 原版 RT-DETR 为 Apache-2.0；Ultralytics 封装仍受其许可证约束 |
| PP-DocLayout-S / PicoDet-S_layout_3cls | 文档页 | PicoDet-S 量级 | — | 版面 mAP 70.9 / 三类别 88.2（Paddle 自建集，**不是印章单独 AP**） | CPU 17.60 / 6.51 ms（常规 / 高性能，Paddle 文档） | 4.8 MB | Apache-2.0（PaddleOCR） |
| PP-OCRv4_mobile_seal_det | 弯曲文本行 | — | — | 自建 500 张圆章上 Hmean 96.36 | CPU 50.38 / 19.64 ms | 4.7 MB（模块页写作 4.6 MB） | Apache-2.0 |
| PP-OCRv4_server_seal_det | 弯曲文本行 | — | — | 同集 Hmean 98.40 | CPU 545.68 / 439.86 ms | 109 MB | Apache-2.0 |

来源：PicoDet 论文 arXiv:2111.00902 表 1；Ultralytics YOLOv8 检测文档与 [RT-DETRv2 对比页](https://docs.ultralytics.com/compare/rtdetr-vs-yolo11)；PaddleOCR 3.x 印章识别流水线与版面模型表（`seal_recognition`、`seal_text_detection`）。移动端毫秒和桌面 ONNX CPU 毫秒**不能横比**，表里把平台写在延迟列里。

对「整章定位」这件单类、大目标的事，算量排序大致是：

`PicoDet-S (320) < NanoDet-M (320) < YOLO11n 在 320 上的 FLOPs < YOLO11n 在 640 < RT-DETRv2-S`

YOLO11n 在 640 是 6.5 GFLOPs，像素数降到 320 大约是四分之一，仍高于 PicoDet-S 的 0.73 GFLOPs。RT-DETR 的精度更高，但 20M 参数和 60 GFLOPs 对不上「小、CPU 快」。

许可证同样决定默认选型。Ultralytics 的 YOLO11 / YOLOv8 代码、训练入口和官方权重是 AGPL-3.0：闭源 SDK 或只公开部分源码的产品不能直接嵌进去，除非另购商业授权。PaddleDetection 的 PicoDet、NanoDet、PaddleOCR 是 Apache-2.0，适合作为 SDK 的上游。

## 3. 识别模型

圆章文字的主流做法不是直接上场景文字的任意形状检测器，而是先把环拉直，再做普通的序列识别。ICDAR 2023 ReST 的前排方案也是这条路，只是拉直和识别器的重量不同。

| 方案 | 思路 | 体积与速度 | 印章上的公开结果 | 评价 |
| --- | --- | --- | --- | --- |
| 极坐标展开 + CRNN | 以圆心采样角度和半径，把环变成水平条，再用 CNN + BiLSTM + CTC | CRNN 常见为数 MB 到十来 MB，CPU 上一条文本约数毫秒到十几毫秒 | 作为经典基线，ReST 前排没有用纯 CRNN 夺冠 | 好训练、好部署。LSTM 在自研运行时里要额外实现 |
| 极坐标展开 + SVTR / SVTR-LCNet | 拉直后用纯视觉序列模型。PP-OCRv4 mobile 是 PPLCNetV3 + 2 层 SVTR neck + CTC，训练时还有 NRTR 指导头 | `PP-OCRv4_mobile_rec` 约 10.5–11 MB；Paddle 文档 CPU 17.48 / 3.61 ms；技术报告里 OpenVINO 约 9.8 ms、11 MB | 通用中文识别准确率 78.74（Paddle 自建集，不是印章集）。ReST 第二名在拉直后用了 SVTR-Small | **推荐的生产识别器**。没有 RNN，CTC 部署简单，CPU 上够快 |
| 曲线拟合再仿射拼接 | 用最小二乘拟合上下边界，把环切成小块后拼成横条（ReST 第二名，中国移动云） | 几何步骤可忽略，时间花在后面的识别器 | ReST 任务二 91.74% | 比纯极坐标更能照顾椭圆和粗细不均的环，实现更长 |
| 弯曲文本检测 + 识别 | DB/印章专用检测头出多边形，再送识别器。即 PaddleOCR 印章流水线 | mobile 检测 4.7 MB + 识别 10.5 MB；检测 CPU 约 20–50 ms | 检测 Hmean 96.36（内部 500 张圆章） | 不假设「一定是圆」，不规则章更稳，但多一个模型 |
| TrOCR / PARSeq | 编码器-解码器或排列自回归 | TrOCR base 是数百 MB，CPU 上是秒级 | ReST 任务二第一名 91.88%（浦发，圆章先做 PCA 和图像分离，再 TrOCR）；也有队伍直接用 PARSeq | 精度上限高，不符合体积和 CPU 预算 |
| 分割集成 + TrOCR | 多个分割模型投票出标题区域，再识别 | 五个分割网络，远超实时 | ReST 任务一 Hmean@0.7 最高 99.06%；任务二第三名 91.22% | 说明数据增强和合成数据很重要，不适合作为在线模型 |

ReST（arXiv:2304.11966）任务二只有五支队伍超过 90%，第一名 91.88%。前三名都大量使用合成章，而不是只靠 5000 张训练图。第二名的识别器就是 SVTR-Small（输入 48×320）加 DIG，几何上把椭圆/圆章拉直。这是「拉直 + 轻量视觉识别器」能进第一梯队的直接证据；他们的检测骨干是 ViT-Giant，那一部分我们不跟。

PP-OCRv4 识别器的结构取舍也写进了技术报告：Lite neck、SVTR 指导、PP-LCNetV3 把 mobile 识别器维持在约 11 MB、CPU OpenVINO 约 10 ms，同时把精度从 PP-OCRv3 的 71.5% 提到 75% 上下（报告里的消融集），部署文档上的最终模型是 10.5 MB / 78.74%。印章标题比自然场景短，拉直之后用这条 mobile 配方是合适的，不必上 server 版（173 MB）。

## 4. 公开数据与许可证

| 数据 | 内容 | 获取 | 许可证与限制 |
| --- | --- | --- | --- |
| ReST，ICDAR 2023 Reading the Seal Title | 10,000 张真实印章，中文约 99%、英文约 1%。圆/椭圆、方、三角。训练/测试各半。多边形 + UTF-8 文本。任务一是标题文本检测（主指标 Hmean@IoU 0.7），任务二是整图读标题（大小写不敏感的整串准确率） | 注册 [Robust Reading Competition ch=20](https://rrc.cvc.uab.es/?ch=20&com=downloads) 后下载。论文：https://arxiv.org/abs/2304.11966 | **没有 SPDX 开源许可证**。门户只把注册描述成意向，不授予再分发或商用。训练集标注有噪声，测试集人工改过。本仓库不收录图像 |
| PaddleOCR 印章检测训练图 | 官方写明印章文本检测模型用的是**内部** 500 张圆章。另有 demo 包 `ocr_curve_det_dataset_examples.tar`，用于说明标注格式 | 模型权重随 PaddleOCR / PaddleX 发布；完整训练集不公开 | 代码与发布的模型文件为 Apache-2.0。内部图像不能当作可再分发数据集。权重可以按 Apache-2.0 使用，但是否满足本仓库「训练数据进仓库」的要求，要以我们自己的合成数据和获授权的 ReST 为准 |
| Company-Names-Corpus 等语料 | ReST 队伍用来做章面文字的公司名、行政区划 | 各仓库自行声明 | 使用前核对仓库许可证。本仓库的合成器使用内置的小字符表，不拉取这些语料 |
| CHSAD | 605 枚章、约 18,390 对真伪样本，面向鉴伪而不是阅读 | https://github.com/usefulbbs/CHSAD | README 未给出许可证，默认不要用于训练或再分发 |
| Hanyin（Zenodo） | 用于 CycleGAN 的印章图像 | https://doi.org/10.5281/zenodo.6561957 | 记录页的摘要没有写许可证。下载前看 Zenodo 右侧的 License 字段，未写明就不要放进仓库 |
| 互联网爬取的公章 | 若干非官方清单里提到过 | — | 公章图像常含机构名称，涉及商标、隐私和不明确的版权。不作为数据方案 |

结论：第一阶段**不能**把真实印章图片提交进 git。可训练、可再分发的部分是合成器本身。ReST 适合作为后续离线评测集，由使用者自行注册下载，脚本只保留格式说明，不保留文件。

## 5. 合成数据

ReST 前排队伍的共同做法是：真实章不够，就自己印。第一名把训练图扩到约 1.5 万张，并针对重叠、模糊再生成；第三名用改过的 SynthText 在文档背景上生成约 1 万张，语料来自公司名库；识别侧有队伍用百万级横排/弯曲文本。合成时他们提到的因素，本仓库的生成器覆盖其中一部分，其余留作下一阶段：

已实现：

- 圆环边框和沿弧排列的标题，字头朝外，底部留缺口，整章随机旋转
- 机构名模板（城市字 + 字号 + 「有限公司 / 股份公司 / 集团」）以及「发票/合同/财务/集团专用章」
- 红、蓝、紫色印泥，乘到纸面上；纸色噪声、模糊、JPEG、掉墨
- 标签是外接正方形和标题字符串，几何参数和推理用的是同一套常量

下一阶段再加：

- 椭圆（先在 x 方向缩放再做极坐标）、方章 2×2、三角章
- 中心横排第二行、五角星
- 透视、重叠章、表格线穿过章面、灰度复印件
- 更大的字符集和可再分发的印章字体（当前子集来自 Droid Sans Fallback，Apache-2.0，只覆盖演示字符表）
- 用 ReST 训练集做微调，评测集只在本地算指标

## 6. 推荐方案

生产模型按这个级联做，而不是一个大的端到端 Transformer：

1. **整章定位：PicoDet-S，单类，输入 320 或 416。** 0.99M 参数、320 输入 0.73 GFLOPs，Apache-2.0，本来就是为 CPU / 移动端设计的。印章是大目标，不需要 640，也不需要 RT-DETR。版面场景可以直接微调 `PicoDet-S_layout` 里的 seal 类，再收成单类以减小头。Ultralytics YOLO 不作为默认，以免 SDK 带上 AGPL。
2. **圆参数：** 在框内用印泥环的角度直方图找缺口，确定起始角。椭圆再加一个轴比。这一步是几何，耗时可以忽略，而且和 ReST 第二名「先校正再识别」一致。
3. **拉直：** 圆章用极坐标展开（本仓库已实现）。椭圆和字环粗细不稳时，升级为第二名的曲线拟合 + 分块仿射。方章、三角章走仿射到水平行，不进极坐标。
4. **识别：SVTR-LCNet + CTC，高度 32 或 48，宽度 160–320。** 结构对齐 PP-OCRv4 mobile（约 11 MB、CPU 约 10 ms 量级），字符集换成印章常用字。不要用 TrOCR 当在线模型。
5. **可选的机构名词表校正**，只做后处理。ReST 前排用行政区划和公司名纠错，能吃掉残字，但不替代模型。

预算：检测 < 5 MB、识别 < 12 MB，桌面 CPU 上整页（检测一次 + 每章识别一次）目标 < 30 ms。这是设计目标，不是已经在 ReST 上测得的数字。

不把 Paddle 的弯曲文本检测器放进第一版，是因为圆章的几何已经能提供文字条，少一个 5–50 ms 的模型。不规则章和严重残缺的环再加 `PP-OCRv4_mobile_seal_det` 作为第二条路径。

## 7. 本仓库第一阶段实际落地的模型

生产级 PicoDet 和 SVTR 的完整训练（COCO/中文识别预训练、蒸馏、ReST 微调）超出这一阶段能在 CPU 上跑完并提交的范围。仓库里先放一条**同结构、可训练、可导出**的缩小版，用来把数据、ONNX、C API 和轮子跑通：

- **SealDet-Tiny**：深度可分离卷积 + 8×8 网格上的单类框。输入 128×128。约 7k 参数。正样本是落在章内的网格，框用绝对坐标回归。它是 PicoDet / NanoDet 那一类「轻量 CNN + 密集头」的最小可运行形式，不是 PicoDet 本身。
- **几何**：与训练共用的极坐标展开和缺口对齐（`python/src/rapidstamp/geometry.py`，C++ 里有一份对照实现）。检测框的圆心经常偏出半径的两成，直接展开会把字切碎，所以推理时先用印泥外环相对内圈的对比度把圆拟合回去，再展开。这一步对应第 6 节的「圆参数」，只是第一阶段用网格搜索而不是椭圆拟合。
- **SealRec-Tiny**：把拉直条带按弧长切成 32×32 的字块，做字符分类。约 2 万参数，字符表 51 字。标题只来自合成语法（城市 + 字号 + 「有限公司 / 股份公司 / 集团」，或四条固定专用章），解码在这份封闭词表上取字符对数概率之和。裸的长度搜索会在相邻长度上选出重复字，整串准确率明显更低。生产模型应换回第 6 节的 SVTR-LCNet CTC，词表留作后处理。导出格式按「一串 op + 权重」设计，识别头可以替换而不改 C API。
- **运行时**：Python 参考实现、可选 ONNX Runtime，以及不链接 ONNX Runtime 的 C++ 解释器（conv / relu / maxpool / gap / linear）。这样 SDK 不用捆绑几十 MB 的 Runtime，模型和代码都能进移动端。ONNX 仍然导出，作为和其他后端的交换格式。

指标以 `models/demo/metrics.json` 为准，那是在合成验证集上测的，不能外推到 ReST。当前这份演示权重：检测头精度 0.741、召回 0.75、平均 IoU 0.660（圆拟合之前的框）；真值框上词表解码整串 0.917、字符 0.986；干净单章端到端整串 0.906、检出 1.00；NumPy 单页约 25 ms。

## 8. 参考

- Yu et al., ICDAR 2023 Competition on Reading the Seal Title. arXiv:2304.11966. https://rrc.cvc.uab.es/?ch=20
- Yu et al., PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices. arXiv:2111.00902
- Du et al., SVTR: Scene Text Recognition with a Single Visual Model
- PaddleOCR 印章识别：https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/seal_recognition.en.md
- PP-OCRv4 介绍（识别器消融与 11 MB / CPU 延迟）：PaddleOCR `ppocr/blog/PP-OCRv4_introduction`
- Ultralytics YOLO11 / YOLOv8 文档中的 COCO 与 ONNX CPU 表；许可证说明：https://www.ultralytics.com/license
- Li et al., TrOCR. arXiv:2109.10282
- Shi et al., An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition（CRNN）
