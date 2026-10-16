# 印章检测与识别：架构、数据与选型

结论先说。部署模型走**专用轻量 CNN 两段式**，不走小尺寸 VLM。检测用自研的单尺度深度可分离卷积检测头（PicoDet 这一档的体积，代码不依赖 Ultralytics），识别用几何展开加 CNN-CTC。两者参数量合计远低于 500 万，不到 1B 上限的百分之一，导出 ONNX 后用 ONNX Runtime 在 CPU 上跑。C API 只接收 RGB 字节，Android / iOS 可以包同一套库。

≤1B 的视觉语言模型（Florence-2、GOT-OCR2、SmolVLM2、PaddleOCR-VL 等）留作以后的教师模型。Qwen 现成的多模态型号里，能读图的最小官方模型是 Qwen2.5-VL-3B 和 Qwen3.5-2B，都超过 1B；Qwen3-0.6B 是纯文本模型，不能直接做印章识别。

本阶段的权重在合成数据上从零训练。合成验证集上的数字写在 `models/metrics.json`，不能当成 ICDAR ReST 的成绩。

## 1. 要解决的问题

输入是文档页或印章裁图，输出是每个印章的框、类别和文字。

需要覆盖的形态：

| 类别 id | 名称 | 典型外观 | 文字怎么读 |
| --- | --- | --- | --- |
| round_official | 圆形公章 | 双圈、中心五角星、上环机构名，有时下环有 13 位印章编码 | 极坐标展开 |
| oval | 椭圆章 | 椭圆双圈，上环公司名，常带「合同专用章」 | 按椭圆半径展开 |
| square_personal | 方形私章 / 名章 | 方框，2 到 4 个汉字竖排或田字 | 按行切开再拼成横条 |
| finance | 财务专用章 | 椭圆，下环「财务专用章」 | 椭圆展开 |
| invoice | 发票专用章 | 椭圆，下环「发票专用章」，中心税号 | 环文展开，中心另读一行数字 |
| company_latin | 外文公司章 | 圆形，英文或拉丁字母环排 | 极坐标展开 |
| notary | 公证章 | 圆形，「NOTARY PUBLIC」或「××市公证处」 | 极坐标展开 |
| rect_stamp | 矩形签章 | 蓝、黑或红色横排，如 APPROVED、PAID、GEPRÜFT | 取中间水平带 |

难点是环排文字、印章和正文重叠、缺墨、扫描旋转，以及中文和拉丁字母混在一个字符集里。ReST 竞赛报告里，检测可以做到 99% Hmean，端到端标题准确率第一名也只有 91.88%，说明识别比框更难。

## 2. 约束

- 参数量 ≤ 1B。这是硬上限，不是目标。目标是 CPU 上一次页面推理在几十毫秒量级，模型文件几 MB 到几十 MB。
- 训练用 PyTorch，部署用 ONNX。C++ 只依赖 ONNX Runtime，不把 Paddle 或 PyTorch 链进 SDK。
- 许可证要能进闭源 SDK。Ultralytics YOLO 是 AGPL-3.0，官方权重和代码不进这个仓库。PicoDet、RT-DETR 官方实现是 Apache-2.0，可以参考结构，本仓库的网络是自己写的。

## 3. 检测架构

下面的 COCO 数字不能直接当成印章精度。印章是单一大类、形状很显眼，专用数据上的 mAP 会高于 COCO。表的用途是比体积、速度和授权。延迟来自不同硬件，**不能横向当成同一把尺子**，表里写了测试条件。

| 模型 | 输入 | 参数量 | 计算量 | COCO mAP | 速度（原作者条件） | 许可证 | 对 CPU SDK 的判断 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n | 640 | 2.6M | 6.5B FLOPs | 39.5 | ONNX CPU 56.1 ms（Ultralytics 文档） | Ultralytics 实现为 AGPL-3.0 | 精度/速度合适，但 AGPL 不适合直接进 SDK |
| YOLOv8n | 640 | 3.2M | 8.7B FLOPs | 37.3 | ONNX CPU 80.4 ms（同上） | AGPL-3.0 | 同上，且更慢 |
| PicoDet-S | 320 | 0.99M | 0.73 GFLOPs | 27.1（PaddleDetection 表；论文在另一组输入上给 30.6） | 骁龙 865、Paddle Lite FP16，约 6.65 ms / 150 FPS | Apache-2.0 | 移动 CPU 上最合适的公开小检测器 |
| PicoDet-L | 论文设置 | 3.3M | 8.74 GFLOPs | 40.9 | 明显快于同精度的 YOLOv5s（论文） | Apache-2.0 | 若合成模型不够，下一档就加到这个量级 |
| RT-DETR-R18 | 640 | 20M | 60 GFLOPs | 46.5 | T4 TensorRT FP16 217 FPS；CPU 上是秒级 | Apache-2.0 | GPU 实时可以，CPU SDK 不合适 |
| PP-OCRv4 mobile seal det | 长边可变 | 文件约 4.7MB | 未在同一表公布 | 自建 500 张圆章文本检测 Hmean 96.36 | 至强 Gold 6271C CPU 约 50 ms / 高性能 20 ms | 代码 Apache-2.0，权重随 PaddleOCR 发布 | 这是**弯曲文本行检测**，不是整枚印章的实例检测 |

来源：

- Ultralytics YOLO11 与 YOLOv8 对比表：<https://docs.ultralytics.com/compare/yolo11-vs-yolov8>
- Ultralytics 许可证：<https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- PicoDet 论文：Yu 等，*PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices*，arXiv:2111.00902
- PaddleDetection PicoDet 配置表：<https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/picodet/README.md>
- RT-DETR：Zhao 等，CVPR 2024，arXiv:2304.08069；PaddleDetection `configs/rtdetr` 中 R18 为 46.5 AP、20M、60 GFLOPs
- PP-OCRv4 印章文本检测：<https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/seal_recognition.html>

Paddle 的 seal det 解决的是「章上的字在哪里」，输入经常已经是一枚章。本仓库还要在整页文档里找出章，所以检测头是实例检测，类别就是上表的 8 类。文本行的多边形留给展开几何，而不是再做一个 DBNet。

本仓库的 `SealDet`：输入 320，四次 stride 2 后得到 20×20 网格，每个格子回归 `tx, ty, tw, th`、objectness 和 8 类 logits。深度可分离卷积为主，最后一层普通卷积补容量。实测 453,197 个参数，ONNX 约 1.8MB。它比 YOLO11n 更小，结构上靠近 PicoDet-S，但是单尺度，因为印章尺度变化没有 COCO 那么极端，letterbox 到 320 之后一档就够。

## 4. 识别架构

环排文字不要直接丢给通用横排 OCR。ReST 前几名和公开论文都是先把弧拉直。

极坐标展开的公开结果：有工作用 OpenCV 极坐标把圆形公章拉开，字符召回从 27.32% 提到 92.23%（同一篇里的对照，不是公开榜单）。见 *Improvement of Chinese seal detection based on polar coordinate transformation*，SPIE 2023，doi:10.1117/12.3009291。ReST 第二名（中国移动云）对椭圆和圆用最小二乘拟合上下边界，再分段仿射拼成横条；第一名在拉直后用 TrOCR。这些系统为了 91% 的标题准确率用了大模型和集成，不符合体积约束，但几何这一步是对的。

拉直之后的识别器：

| 模型 | 公开精度 | 体积 / 速度 | 部署 |
| --- | --- | --- | --- |
| SVTR-Tiny | 英文 6 个场景平均约 90.9，10 个场景平均 90.13；中文 scene_test 67.90 | 系列里最小的纯视觉模型，无 RNN | ONNX 友好。数字来自 PaddleOCR SVTR 文档 |
| PP-OCRv4 mobile rec | PaddleX 综合识别准确率 78.20（和 SVTR 不是同一套测试集） | 模型文件约 10.6MB；CPU 约 47 ms（至强，Paddle 测试） | 中英数字的工业基线，蒸馏训练重 |
| PP-OCRv4 server rec | 更高，文件约 88MB | 服务端 | 超出「小」的目标 |
| CRNN（CNN + BiLSTM + CTC） | 低于 SVTR / PP-OCRv4 | 小 | LSTM 能导出 ONNX，但 CPU 上不如等宽卷积 |
| TrOCR / PARSeq | ReST 端到端前排在用 | 变压器，大 | 不适合本 SDK |
| 本仓库 SealRec | 见 `models/metrics.json` 的合成条带准确率 | 1,501,326 参数，ONNX 约 5.8MB。CNN + 两层 Conv1d，CTC，无 RNN | 输入灰度 32×384，时间步 96 |

SVTR 文档：<https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/algorithm/text_recognition/algorithm_rec_svtr.en.md>

PP-OCRv4 模型列表：<http://www.paddleocr.ai/main/version2.x/ppocr/model_list.html>

`SealRec` 是 SVTR 的思路（只用视觉模型出序列）缩到能在 CPU 上从零训练的尺寸：高度压到 1 之后用卷积混合相邻步，CTC 解码。字符集在第一阶段是封闭的，由合成语料里实际出现的汉字、数字、拉丁字母和少量西欧变音字母组成，空白符占 index 0。封闭字符集是为了让小模型在合成数据上收敛。下一步把字表扩到常用汉字（大约 GB2312 一级）时，沿用同一个头，只加合成覆盖和真实微调。

圆形和椭圆的展开与渲染共用一套角度：0° 在正右，顺时针增加，条带第 0 行是外半径。训练时按已知旋转把起始角对齐到 140° 附近；推理不知道旋转，就每 15° 搜一次（24 步）并保留置信度最高的解码。方章按 2×1、3×1、4×1、2×2 试切，同样取置信度最高者。矩形章取高度 22%–78% 的水平带。

## 5. 轻量 CNN 与 ≤1B VLM

| | 专用 CNN（本方案） | ≤1B 视觉语言模型 |
| --- | --- | --- |
| 代表 | PicoDet 级检测 + SVTR/CRNN 级 CTC。本仓库 SealDet 453,197 参数，SealRec 1,501,326 参数，合计约 1.95M | Florence-2-base 0.23B、Florence-2-large 0.77B（MIT）；GOT-OCR2.0 约 0.6B；SmolVLM2 256M / 500M（Apache-2.0）；Granite-Docling 258M；PaddleOCR-VL 约 0.9B |
| 超过 1B、不能用 | — | Qwen2.5-VL-3B（约 3B–3.8B，含视觉塔）、Qwen2-VL-2B、Qwen3.5-2B（约 2.3B）。Qwen3-0.6B 只有文本塔 |
| 体积 | ONNX 约 1.8MB（检测）+ 5.8MB（识别） | 几百 MB 到约 2GB（fp16） |
| CPU 速度 | 开发机 ONNX Runtime CPU：检测约 3 ms，识别约 3 ms。圆章按 15° 搜 24 个起始角，一枚大约 70 ms | 即使 0.25B–0.6B，也要跑视觉编码器和自回归解码。公开笔记本测试里 Florence-2-large 在 GPU 上大约 5 秒/页，Qwen2.5-VL-3B 更慢。CPU 会再慢一个数量级以上 |
| 印章环文 | 展开后就是普通横排，小模型专门学这一个分布 | 通用 OCR 对环文没有几何先验。可以微调，但微调后的解码器仍然重 |
| 精度上限 | 域内可以很高；换一套真实扫描会掉，要靠真实数据微调 | 文档 OCR 的零样本能力更强，弯曲印章不是它们的训练分布，零样本不可靠 |
| C++ 部署 | ONNX Runtime 的 C API 成熟，Linux / Windows / macOS / Android / iOS 都有预编译包 | 需要分词器、图像处理器和采样循环。路径是 onnxruntime-genai、llama.cpp 或厂商运行时，和「一个检测头 + 一个 CTC」不是同一套 SDK |

Florence-2 体积和许可证都在界内，GOT-OCR2 也在界内，但它们解决不了「CPU 上快」和「C API 薄」这两条。PaddleOCR-VL 接近 1B，同样不适合放进端侧 SDK。

因此 VLM 不作为部署模型。若以后真实环文上 CNN 到了天花板，可以用一个 ≤1B 的模型做教师，把难样本的转写蒸馏回 CTC，部署物仍然是现在的两个 ONNX。

## 6. 公开数据与许可证

没有一份数据同时覆盖中文公章、发票章和国外邮戳，并且能直接商用再分发。下面按「能不能放进训练」分开写。本仓库**不收录**这些图像，只提供转换脚本。

| 数据 | 内容 | 许可 / 使用限制 | 在本项目里的位置 |
| --- | --- | --- | --- |
| ICDAR 2023 ReST | 10,000 张真实印章，训练/测试各半。约 99% 中文、1% 英文。圆/椭圆、矩形、少量三角形。文字多边形 + 转写。任务 1 用 IoU 0.7 的 Hmean，第一名 99.06%。任务 2 是整图标题准确率，第一名 91.88% | 论文和 RRC 页面都没有给出 SPDX 许可证。下载要在 Robust Reading Competition 注册。**不要再分发图像**。研究使用以竞赛门户条款为准 | `training/scripts/import_rest.py` 把多边形转成外接框。注意 ReST 标的是标题文字区域，不是整枚章的外框，和本检测任务不完全一样 |
| PaddleOCR 印章示例 | `ocr_curve_det_dataset_examples.tar`，演示弯曲文本检测格式。另有内部 500 张圆章评测，**未公开** | 代码 Apache-2.0。示例包的数据条款页面没有单独写清，使用前要再核对下载页 | 只作格式参考，不作为训练集 |
| StaVer | 400 张自动生成发票，打印、盖章、扫描。像素级印章掩膜，含多章、与签名/正文重叠、黑色章。ICDAR 2011，Micenková 与 van Beusekom。作者方法大约 precision 84%、recall 83% | Kaggle 镜像写的是可下载，不是 OSI 许可证。按研究数据对待，**不假定可以商用或再分发** | `training/scripts/import_staver.py` 把掩膜转成框。没有文字转写，只能训练检测 |
| Roboflow「Stamp detection」（用户 stampa） | 约 2.1k 张，类别 signature 与 stamp | **CC BY 4.0** | 国外印章检测的可商用补充。使用时保留署名 |
| Hugging Face `mapo80/stamps` | 把多份印章检测数据合成约 2.7 万张的 YOLO 集 | 集合卡片写 CC BY 4.0，但分项里 StaVer 只是 "free access"，FUNSD / RVL-CDIP 是研究用途。**不能把整包当成 CC BY** | 若用，只取卡片里明确是 CC BY 4.0 的 Roboflow 子集 |
| Chinese-Seal Dataset (CSD) | 1,486 张图、16,688 枚章、33,364 条标注，红/蓝/黑，易/中/难 | SSRN 预印本没有写许可证 | 先确认授权再下载。链接在论文里 |
| DocDiff 印章子集 | 1,597 张红色章及掩膜，作者说明大部分裁自 ReST | 继承 ReST 的再分发限制 | 不入库 |
| FUNSD | 表格理解，不是印章集。有时被拿来当文档背景 | 非商业研究与教学 | 不作为商业训练背景 |
| RVL-CDIP / IIT-CDIP | 烟草诉讼档案扫描件 | 没有一份简单的商业许可证 | 不作为背景 |
| Company-Names-Corpus | ReST 参赛队用来做公司名语料，不是图像 | 以该 GitHub 仓库许可证为准 | 本仓库用自编的城市、字号、后缀表，不拷贝该语料 |

ReST 论文：Yu 等，*ICDAR 2023 Competition on Reading the Seal Title*，arXiv:2304.11966。门户：<https://rrc.cvc.uab.es/?ch=20>。

StaVer：Micenková, van Beusekom, *Stamp Detection in Color Document Images*, ICDAR 2011。Kaggle 镜像：<https://www.kaggle.com/datasets/rtatman/stamp-verification-staver-dataset>。

Roboflow 集：<https://universe.roboflow.com/stampa/stamp-detection-4jhgg>，页面标注 CC BY 4.0。

CSD：Xu 等，SSRN 5079459。

## 7. 合成数据

ReST 前三名都在真实 5,000 张之外合成了章：CCB 用公司名语料、随机字体、颜色、透明度和断裂边框；中国移动云合成了圆、椭圆、矩形、三角；中科院团队用改过的 SynthText，背景用 WTW，大约 10,000 张。端到端第一名还用了大约 600 万条文本训练 TrOCR。合成不是凑数，是这个任务上公开有效的做法。

本仓库的生成器在 `training/rapidstamp_train/synthesize.py`，不下载外部图像。

- **版式**：上表 8 类。圆章可 360° 旋转，椭圆 ±4°，方章和矩形章 ±5°。旋转后的圆章用同一套极角公式对齐，单元测试检查展开峰值和旋转相关。
- **文字**：自编中文公司名（城市 + 字号 + 「有限公司 / 股份有限公司 / 集团有限公司」）、财务/发票/合同/业务/报关专用章、13 位编码、18 位税号、英文公司名、公证用语，以及带变音的 `GEPRÜFT`、`ESPAÑA`、`MÜNCHEN`。字符必须落在封闭字表里，生成时会检查。
- **绘制**：每个字沿弧刚性旋转，字顶朝外。双圈、公章五角星、发票章中心税号。墨色以朱红为主，外文章和矩形章加入蓝、黑、紫。墨迹有颗粒、模糊和随机断笔。章内铺一层浅色底，减轻正文透出，同时页面上仍有灰色正文和红色横线做干扰。
- **页面**：约 8% 空页，其余 1–3 枚章，JPEG 质量 60–95。检测标签是紧外接框和类别。识别标签是从页面裁图上、用和推理相同的函数切出来的横条，并加模糊、亮度、JPEG。另外混入约一条/页的纯横排文本，让 CTC 先学会字表。
- **字体**：`training/fonts/RapidStampCJK.ttf` 是文泉驿微米黑的子集，只留字表里的字形，Apache-2.0（Google 与文泉驿，见 `training/fonts/NOTICE`）。拉丁字母和变音用系统里的 DejaVu Sans Bold。两份字体都不从网络临时下载。
- **真实数据接入**：`import_rest.py`、`import_staver.py`。拿到授权数据后，检测框可以直接并进 `det/*/labels`。ReST 的多边形要先当成文本行，不要和整章框混成一个类。

## 8. 推荐方案

**部署：SealDet + 几何展开 + SealRec，ONNX Runtime，C API。**

理由：

1. 印章检测不需要 DETR 的全局注意力。PicoDet 这一档在移动 CPU 上已经证明能跑到百帧，印章的类间差异（圆、椭圆、方、横条、颜色）比 COCO 的 80 类更容易。
2. 环文的精度主要来自展开，而不是来自一个能看圆的识别器。展开是确定的几何，放在 C++ 里不占参数，也便于和渲染器对齐。
3. CTC 小模型的 CPU 延迟和工程面都适合多平台 SDK。VLM 在 1B 以下也仍然是生成式运行时。
4. AGPL 的 YOLO 权重不进仓库。需要更大检测器时，把通道加到 PicoDet-L（约 3.3M）仍然远低于 1B，并且可以继续用现在的 ONNX 接口（输出张量布局不变的前提下只替换文件）。

训练顺序：合成数据预训练 → 在有授权的 ReST / CC BY 印章框上微调检测 → 用 ReST 标题转写和更多合成环文微调识别 → INT8 量化。VLM 只在第三步之后、CNN 仍有系统性环文错误时考虑做教师。

## 9. 本仓库里已经落地的部分

- `training/`：合成、训练、ONNX 导出、ReST 与 StaVer 的转换脚本。
- `cpp/`：C API、与 Python 对齐的几何、ONNX Runtime 会话。CI 编 Linux x64、Linux arm64、Windows x64、macOS arm64，并交叉编译 Android arm64。
- `python/`：PyPI 包 `rapidstamp`，cibuildwheel 打 Linux / Windows / macOS 轮子。
- 权重和合成验证指标在训练后写入 `models/`。那份数字是合成 hold-out，不是 ReST。

## 10. 已知缺口

- 字表是封闭的。真实公司名里的生僻字会无法编码。
- 三角章在 ReST 里很少，这一阶段没有单独的类。
- 方章的行列是在推理时搜索的，复杂篆体、阴文不会好。
- 椭圆大角度旋转后，外接框不再对齐椭圆轴，展开会糊。合成里把椭圆旋转限制在 ±4°。
- 发票税号是第二条读取，和环文分开计。
- 没有在 ReST 测试集上出分。那是下一阶段，而且图像不能提交进这个仓库。
