# 印章检测与识别：架构、数据与选型

结论先说。部署模型走**专用轻量 CNN 两段式**，不走小尺寸 VLM。检测是 YOLOX-Nano（Megvii，Apache-2.0）在印章数据上的微调，识别用几何展开加 CNN-CTC。检测 898,119 参数，识别 6,169,425 参数（2,817 类，含空白），合计约 707 万，仍远低于 1B。导出 ONNX 后用 ONNX Runtime 在 CPU 上跑。C API 只接收 RGB 字节，Android / iOS 可以包同一套库。Ultralytics YOLO 是 AGPL-3.0，不进这个仓库。

≤1B 的视觉语言模型（Florence-2、GOT-OCR2、SmolVLM2、PaddleOCR-VL 等）留作以后的教师模型。Qwen 现成的多模态型号里，能读图的最小官方模型是 Qwen2.5-VL-3B 和 Qwen3.5-2B，都超过 1B；Qwen3-0.6B 是纯文本模型，不能直接做印章识别。

当前权重和混合 hold-out 写在 `models/` 与 `models/metrics.json`。那不是 ICDAR ReST 的官方 test。

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
- 许可证要能进闭源 SDK。Ultralytics YOLO 是 AGPL-3.0，官方权重和代码不进这个仓库。检测器选用 Megvii YOLOX-Nano，Apache-2.0，结构按权重对齐重写在 `training/rapidstamp_train/yolox_nano.py`，COCO 预训练权重只放在训练机。PicoDet、RT-DETR 官方实现也是 Apache-2.0。

## 3. 检测架构

下面的 COCO 数字不能直接当成印章精度。印章是单一大类、形状很显眼，专用数据上的 mAP 会高于 COCO。表的用途是比体积、速度和授权。延迟来自不同硬件，**不能横向当成同一把尺子**，表里写了测试条件。

| 模型 | 输入 | 参数量 | 计算量 | COCO mAP | 速度（原作者条件） | 许可证 | 对 CPU SDK 的判断 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n | 640 | 2.6M | 6.5B FLOPs | 39.5 | ONNX CPU 56.1 ms（Ultralytics 文档） | Ultralytics 实现为 AGPL-3.0 | 精度/速度合适，但 AGPL 不适合直接进 SDK |
| YOLOv8n | 640 | 3.2M | 8.7B FLOPs | 37.3 | ONNX CPU 80.4 ms（同上） | AGPL-3.0 | 同上，且更慢 |
| PicoDet-S | 320 | 0.99M | 0.73 GFLOPs | 27.1（PaddleDetection 表；论文在另一组输入上给 30.6） | 骁龙 865、Paddle Lite FP16，约 6.65 ms / 150 FPS | Apache-2.0 | 移动 CPU 上最合适的公开小检测器 |
| PicoDet-L | 论文设置 | 3.3M | 8.74 GFLOPs | 40.9 | 明显快于同精度的 YOLOv5s（论文） | Apache-2.0 | 若合成模型不够，下一档就加到这个量级 |
| RT-DETR-R18 | 640 | 20M | 60 GFLOPs | 46.5 | T4 TensorRT FP16 217 FPS；CPU 上是秒级 | Apache-2.0 | GPU 实时可以，CPU SDK 不合适 |
| YOLOX-Nano | 320（本仓库） | 898,119 | 未在此复测 COCO FLOPs | 印章 hold-out 见第 9 节 | 混合 hold-out 整页 188 ms；真实 52 页 268 ms | Apache-2.0（Megvii） | 采用。COCO 权重微调，图里已经解出框 |
| PP-OCRv4 mobile seal det | 长边可变 | 文件约 4.7MB | 未在同一表公布 | 自建 500 张圆章文本检测 Hmean 96.36 | 至强 Gold 6271C CPU 约 50 ms / 高性能 20 ms | 代码 Apache-2.0，权重随 PaddleOCR 发布 | 这是**弯曲文本行检测**，不是整枚印章的实例检测 |

来源：

- Ultralytics YOLO11 与 YOLOv8 对比表：<https://docs.ultralytics.com/compare/yolo11-vs-yolov8>
- Ultralytics 许可证：<https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- PicoDet 论文：Yu 等，*PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices*，arXiv:2111.00902
- PaddleDetection PicoDet 配置表：<https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/picodet/README.md>
- RT-DETR：Zhao 等，CVPR 2024，arXiv:2304.08069；PaddleDetection `configs/rtdetr` 中 R18 为 46.5 AP、20M、60 GFLOPs
- PP-OCRv4 印章文本检测：<https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/seal_recognition.html>
- YOLOX：Ge 等，*YOLOX: Exceeding YOLO Series in 2021*，arXiv:2107.08430；Nano 权重 <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth>

Paddle 的 seal det 解决的是「章上的字在哪里」，输入经常已经是一枚章。本仓库还要在整页文档里找出章，所以检测头是实例检测，类别就是上表的 8 类。文本行的多边形留给展开几何，而不是再做一个 DBNet。

本仓库的 `SealDet` 是 YOLOX-Nano：depth 0.33、width 0.25、depthwise，输入 320。stride 8/16/32 共 2,100 个锚点。训练输出原始 logits；导出的 ONNX 在图内解成归一化 xyxy、objectness 概率和各类概率，Python 与 C++ 只做阈值和按类 NMS。COCO 分类头换成 8 类，其余张量从 Megvii 的 `yolox_nano.pth`（tag 0.1.1rc0，Focus stem）加载。实测 898,119 个参数，ONNX 约 3.5MB。版权声明见仓库根目录 `NOTICE`。

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
| 本仓库 SealRec | 见 `models/metrics.json` | 6,169,425 参数，2,817 类（含空白），ONNX 约 24MB。CNN + Conv1d，CTC，无 RNN | 输入灰度 32×384，时间步 96 |

SVTR 文档：<https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/algorithm/text_recognition/algorithm_rec_svtr.en.md>

PP-OCRv4 模型列表：<http://www.paddleocr.ai/main/version2.x/ppocr/model_list.html>

`SealRec` 是 SVTR 的思路（只用视觉模型出序列）缩到能在 CPU 上从零训练的尺寸：高度压到 1 之后用卷积混合相邻步，CTC 解码。空白符占 index 0。字表从合成语料的封闭集开始，再并入 ReST 和 Paddle 标签里出现的字，现在是 2,816 字。分类头按字符对齐从旧模型扩出来，还没有覆盖整张 GB2312。

圆形和椭圆的展开与渲染共用一套角度：0° 在正右，顺时针增加，条带第 0 行是外半径。训练时按已知旋转把起始角对齐到 140° 附近。推理对圆章和椭圆做 8 个起始角（每 45° 一个），再加规范的 140°。墨迹环上若能找到不短于 40° 的空白，再加一个由空白估出的起始角。含汉字的读法按发出字符的平均概率排序，纯拉丁和数字按整段概率质量排序。差距小于 0.012 时偏向更长、且以汉字开头的读法。不再对最高分做 ±12° 细搜：15° 网格会把旋转后的残段也送进候选，那些残段经常压过完整标题。方章按 2×1、3×1、4×1、2×2 试切，分数里减去 0.05×|字数−格数|，再用同一套近分规则。矩形章取高度 22%–78% 的水平带。

## 5. 轻量 CNN 与 ≤1B VLM

| | 专用 CNN（本方案） | ≤1B 视觉语言模型 |
| --- | --- | --- |
| 代表 | YOLOX-Nano + CNN-CTC。本仓库 SealDet 898,119 参数，SealRec 6,169,425 参数，合计约 707 万 | Florence-2-base 0.23B、Florence-2-large 0.77B（MIT）；GOT-OCR2.0 约 0.6B；SmolVLM2 256M / 500M（Apache-2.0）；Granite-Docling 258M；PaddleOCR-VL 约 0.9B |
| 超过 1B、不能用 | — | Qwen2.5-VL-3B（约 3B–3.8B，含视觉塔）、Qwen2-VL-2B、Qwen3.5-2B（约 2.3B）。Qwen3-0.6B 只有文本塔 |
| 体积 | ONNX 约 3.5MB（检测）+ 24MB（识别） | 几百 MB 到约 2GB（fp16） |
| CPU 速度 | 训练机上整页约 188 ms。圆章 8 个起始角，一次批量识别，不再细搜 | 训练机 CPU 上实测：Florence-2-base 每条 2.4 秒，SmolVLM-256M 每条 6.5 秒。都不是几百毫秒 |
| 印章环文 | 展开后就是普通横排，小模型专门学这一个分布 | 通用 OCR 对环文没有几何先验。可以微调，但微调后的解码器仍然重 |
| 精度上限 | 域内可以很高；换一套真实扫描会掉，要靠真实数据微调 | 文档 OCR 的零样本能力更强，弯曲印章不是它们的训练分布，零样本不可靠 |
| C++ 部署 | ONNX Runtime 的 C API 成熟，Linux / Windows / macOS / Android / iOS 都有预编译包 | 需要分词器、图像处理器和采样循环。路径是 onnxruntime-genai、llama.cpp 或厂商运行时，和「一个检测头 + 一个 CTC」不是同一套 SDK |

Florence-2-base（231,414,016 参数，MIT）和 SmolVLM-256M（256,484,928 参数，Apache-2.0）都在 1B 以内。Qwen 能读图的官方型号最小是 Qwen2.5-VL-3B，超过 1B，没有拿来评。PaddleOCR-VL-0.9B 是 Apache-2.0，但是自回归文档模型，官方路径是 transformers / vLLM，不是这一套 CTC ONNX。

同一台训练机的 CPU 上，Florence-2-base 用 `<OCR>` 读 4 条合成方章和 4 条 Paddle 圆章，整串 0/8，平均 2362 ms。SmolVLM-256M 读 8 条方章加 8 条 Paddle 圆章，整串 0/16，平均 6539 ms，输出是和章面无关的句子。两条都不满足「准确率明显更高、每枚章几百毫秒、C++ 可部署」这三条，所以不作为默认识别器。SmolVLM 在 llama.cpp 里有转换脚本，Florence-2 是编码器-解码器，没有一条和现有 ONNX Runtime C API 一样薄的路径。轻量 CTC 继续当默认识别，也是快速模式。

因此 VLM 不作为部署模型。若以后真实环文上 CTC 到了天花板，可以用一个 ≤1B 的模型做教师，把难样本的转写蒸馏回 CTC，部署物仍然是检测加 CTC 这两个 ONNX。

## 6. 公开数据与许可证

没有一份数据同时覆盖中文公章、发票章和国外邮戳，并且能直接商用再分发。下面按「能不能放进训练」分开写。本仓库**不收录**这些图像，只提供转换脚本。

| 数据 | 内容 | 许可 / 使用限制 | 在本项目里的位置 |
| --- | --- | --- | --- |
| ICDAR 2023 ReST | 这份压缩包里是 5,000 张带 `gt_{id}` 标题多边形和转写的训练图，另有 5,000 张没有标注的测试图。图像是已经裁好的章，不是整页。多边形点很密，包住的是标题文字，不是整枚章的外轮廓。训练转写里出现 1,827 个不同字符 | 包内没有 LICENSE / README。论文和 RRC 页面也没有 SPDX。**图像不进 git、不再分发**。使用以竞赛注册条款为准。测试图没有转写，所以不能拿来当有监督测试；验证集是从 5,000 张有标注图里按 id 留出的 10% | `training/scripts/import_rest.py` 把裁图贴回合成页面，检测框是贴上去的整枚裁图，识别条来自同一套极坐标展开 |
| PaddleOCR 印章数据 | 用户提供的 `dataset.zip`，约 2.3GB、10008 个文件，内含 `dataset/train/train.txt`。Paddle 公开文档里的印章示例包只是弯曲文本检测格式演示，另有内部 500 张圆章评测未公开 | PaddleOCR 代码是 Apache-2.0。这份数据包里有没有单独的数据许可证，要等压缩包到位后看包内说明。在看到许可证之前，**不假定可以再分发**，图像不进 git | `training/scripts/import_paddle.py`。训练机路径是 `/home/mayong/datasets/paddle_seal/paddle_seal.zip` |
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
- **真实数据接入**：`import_rest.py`、`import_paddle.py`、`import_staver.py`，再用 `finetune_real.py` 和合成页混在一起微调。ReST 的多边形是标题区域，不拿它的外接框当印章框；检测框是贴回页面的整枚裁图。
- **外部生成器**：<https://github.com/wang-zhix/seal_generate>（它自己写的上游是 <https://github.com/zh3389/seal_generate>）。仓库根目录没有 LICENSE，不能把源码再分发进这个 Apache-2.0 仓库，也不能把它当成可安装依赖。`fonts/` 里是 SimSun、华文宋和方正字体，这些是专有字体，不下载、不随训练环境分发，也不使用它的 `dicts/hospital.txt`。版式上借用的只有两件公开可见的做法，并且是在本仓库自己的绘制函数里实现的：中文笔画横向压到约 0.7 宽，圆章更常带一圈底部数字。

训练机上空间够用的位置是 `/home/mayong`（单独一块盘）。数据、混合集、微调和导出的权重都放在那里。说明见 [data.md](data.md)。

## 8. 推荐方案

**部署：SealDet + 几何展开 + SealRec，ONNX Runtime，C API。**

理由：

1. 印章检测不需要 DETR 的全局注意力。PicoDet 这一档在移动 CPU 上已经证明能跑到百帧，印章的类间差异（圆、椭圆、方、横条、颜色）比 COCO 的 80 类更容易。
2. 环文的精度主要来自展开，而不是来自一个能看圆的识别器。展开是确定的几何，放在 C++ 里不占参数，也便于和渲染器对齐。
3. CTC 小模型的 CPU 延迟和工程面都适合多平台 SDK。VLM 在 1B 以下也仍然是生成式运行时。
4. AGPL 的 Ultralytics YOLO 不进仓库。检测用 Apache-2.0 的 YOLOX-Nano，COCO 预训练后再在印章页上微调。ONNX 在图内把框解出来，换文件时 Python 和 C++ 都按「已解码的 xyxy + 概率」读。

训练顺序：合成数据预训练 → 在有授权的 ReST / CC BY 印章框上微调检测 → 用 ReST 标题转写和更多合成环文微调识别 → INT8 量化。VLM 只在第三步之后、CNN 仍有系统性环文错误时考虑做教师。

## 9. 训练机上的实测

权重在 `models/`。检测换成 YOLOX-Nano：898,119 参数，ONNX 3,671,258 字节（约 3.5MB），md5 `aab5e8629155d0abfdd7da5bb6368028`。初始化是 Megvii COCO 的 `yolox_nano.pth`，再在合成页、ReST、Paddle 圆章和网页训练页上微调 8 个 epoch。训练图里 85% 会画红字、蓝字、红线、表格和小 logo，另外每个 epoch 加 4,000 张只有这些干扰、没有章的空白页。识别没有换：仍是 Paddle 微调后的 6,169,425 参数、2,817 类，ONNX 约 24MB。圆章仍是 8 个起始角。

同一份混合 hold-out（1,277 页、1,555 枚章，不是 RRC 官方 test）上，检测 mAP50 从 0.804 到 0.835，检测召回从 0.997 到 1.000。标题整串从 0.513 到 0.523。整页从 217 ms 到 188 ms。

| | 检测召回 @0.5 | 标题整串命中 | 说明 |
| --- | --- | --- | --- |
| 合成验证页 | 1.000 | 0.648 | 上一轮 0.645。圆章对齐条带仍是 0.764（146/191） |
| ReST 留出的 477 张 | 1.000 | 0.239 | 上一轮 0.216。对齐条带仍是 0.237 |
| 混合，检测 mAP50 | 0.835 | 0.523 | 识别条带整串仍是 0.633。自由环文验证条带仍是 0.906 |

分项整页整串：椭圆 0.346（上一轮 0.315），财务章 0.641（0.547），发票章 0.726（0.757），外文公司章 0.843（0.898），公证章 0.779（持平），矩形章 0.924（0.830），圆章 0.324（0.339），方章仍是 0。发票中心数字从 0.089 降到 0.041（5/123），新框把中心行切偏了。Paddle 验证条带整串仍是 0.113（318/2819），公司名仍是 0.465（1310/2819）。自由姓名格验证条带仍是 0.375（150/400）。合成方章条带仍是 0/137。

方章另训过一版识别，没有发布。训练条带是从字表里可绘制的 2,745 个汉字随机抽的姓名，2×1、3×1、2×2、4×1，阳刻和阴刻各约占一半，笔画用 MaxFilter 加粗，不是单独的篆体字体。12,000 条训练、姓名重复 6 次，从已发布的识别器接着训 6 个 epoch。合成方章条带仍是 0/137（「周明蒋芳」读成「蒋明蒋磊」这一类姓氏替换）。另抽的 200 条同分布条带整串 0.045（9/200）。Paddle 公司名从 0.465 掉到 0.319（900/2819），整串从 0.113 掉到 0.079。混合标题整串 0.514，低于配旧识别器的 0.523。所以 `models/` 里的 `seal_rec.onnx` 没换。

网页真实测试集仍是固定的 52 页、58 个框。类无关 precision@0.5 从 0.033 到 0.271（23/85），召回从 0.379 到 0.397（23/58），mAP50 从 0.032 到 0.097，预测框从 671 个到 85 个。整页从 1577 ms 到 268 ms。这 52 页的文字标注几乎是空的，整串不以它们为准。更早的从零训练检测器在同一套上是 precision 0.024、召回 0.345、845 个框、2192 ms。

## 10. 本仓库里已经落地的部分

- `training/`：合成、训练、ONNX 导出、ReST 与 StaVer 的转换脚本。
- `cpp/`：C API、与 Python 对齐的几何、ONNX Runtime 会话。CI 编 Linux x64、Linux arm64、Windows x64、macOS arm64，并交叉编译 Android arm64。
- `python/`：PyPI 包 `rapidstamp`。包是纯 Python，CI 在 Linux、Windows、macOS 上用 `python -m build` 打 `py3-none-any` 轮子并安装。
- 权重和混合 hold-out 指标在 `models/`。检测是 COCO 预训练的 YOLOX-Nano，再在合成、ReST、Paddle 和网页训练页上微调；识别在自由环文上训练后，又用 Paddle 圆章和自由姓名格微调。
- 网页印章图的采集脚本在 `training/scripts/collect_web_seals.py`，来源 URL 在 `training/manifests/web_seals_sources.jsonl`，固定划分在 `training/manifests/web_seals_split.jsonl`。图片只在训练机上。

## 11. 已知缺口

- 字表有 2,816 字。扩展子集字体覆盖原来的 1,921 字；Paddle 新进来的字合成时不一定画得出来，训练用的是真实图。ReST 和 Paddle 里的生僻字仍会被读成形近字。
- 三角章在 ReST 里很少，这一阶段没有单独的类。
- 方章的行列是在推理时搜索的。随机姓名、阴阳刻和加粗笔画没有把合成方章条带从 0/137 拉起来，模型仍会换成另一个常见姓名。没有找到可以再分发的篆体字体，MaxFilter 只是笔画变粗，不是篆书。
- 椭圆大角度旋转后，外接框不再对齐椭圆轴，展开会糊。合成里把椭圆旋转限制在 ±4°。
- 发票税号是第二条读取，和环文分开计。
- 这份 ReST 压缩包的测试图没有转写，所以没有官方测试集分数。留出的 10% 是有标注训练图的 hold-out，不是 RRC 的 test split。
- `training/fonts/RapidStampCJK-ext.ttf` 覆盖原来的 1,921 字。字表后半段的字合成时画不出来。
- 自由姓名格验证条带整串 0.375，合成方章条带和整页方章仍是 0。开放字表的姓名章实验见第 9 节，没有替换识别器。

## 12. 来也文章里能用的部分

来源：来也技术团队，刘凤楠，2022-07-17，《来也智能文档处理系统中的印章识别实践》，<https://laiye.com/news/post/2613.html>。

这篇文章**没有**把极坐标展开写成他们的方法。环文只出现在他们否掉两段式识别的理由里：先检测字行再拉直，误差会叠上去。不要把这篇当成极坐标展开的出处。本仓库的圆章和椭圆仍然用极坐标把环文拉成横条，再送给识别器。

他们实际落地的是：

- **检测**：改过的 YOLOv5，形状和颜色做成多标签，损失是 `BCEWithLogitsLoss`，NMS 用 `combined_non_max_suppression`。YOLOv5 这条实现链是 AGPL，不进本仓库。检测仍用 Apache-2.0 的 YOLOX-Nano。
- **识别**：比较过 Mask-TextSpotter（要字符级掩膜，中文太重）、ABCNet（BezierAlign，网里仍是两段）、PGNet（字符图，标注别扭），最后用 TrOCR 端到端读整枚章。TrOCR 超出 CPU SDK 的体积，不作为默认识别器。
- **数据**：两万张以上的真实标注章，三十万张以上用模板和脚本合成的章。识别增强包括随机缩放、旋转、缺墨纹理、HSV，以及用随机像素填 alpha 来模拟正文压在章上。
- **方章顺序**：两条阅读顺序，用特殊换行记。「$$」是从上到下，「**」是从右到左。文中的例子是「兴业银行股份有限公司$$投资银行部」和「柴姣**之印」。传统名章是先右列后左列、列内从上到下。
- **解码**：束搜索比贪心的 F1 只高 0.4%，他们上线贪心。本仓库的 CTC 也是贪心。
- **指标**：十三个真实场景上的「章在不在」F1 和「文字项」F1。他们报告超过当时几家通用 OCR。

方章的「$$ / **」不放进 CTC。本仓库合成方章的标签是行优先、从左到右拼出来的横条，条带整串按这个顺序计。另做一版从右到左的切条，不会改变这 137 条已经切好的条带分数，所以没有改默认搜索。

文中 20 张图收在训练机 `/home/mayong/datasets/web_seals/laiye2613/`，来源页和 CDN 地址写在同目录的 `laiye_manifest.jsonl`。按像素看，这些是约 55 像素的图标、流程图和架构图。没有一张能在不臆造文字的情况下框出一枚可校对的章。它们没有并进固定的 52 页测试集，也没有当成训练页。测试集划分不改。

识别器实验用的是 PaddleOCR 的 PP-OCRv5 mobile rec（Apache-2.0，推理 ONNX 约 16MB，18,383 字，输入名 `x`，BGR、高 48、`(x/255-0.5)/0.5`，CTC 空白在 index 0）。零样本直接读现有条带，不把红墨转成黑字（转了更差）：

| 条带 | 已发布 CTC | PP-OCRv5 零样本 BGR | 每条 |
| --- | --- | --- | --- |
| 合成方章 137 | 0/137 | 0.788（108/137），「周明蒋芳」读对 | 约 36 ms |
| 合成圆章 191 | 0.764（146/191） | 0.309（59/191）。数字之间会插入引号，公司名末字也会掉 | 约 36 ms |
| 自由环文 800 | 0.906 | 0.368（294/800） | 约 36 ms |
| 自由姓名格 400 | 0.375 | 0.225（90/400） | 约 36 ms |
| 姓名章留出 200 | 另训那版是 0.045 | 0.045（9/200） | 约 36 ms |
| Paddle 验证 2819 | 整串 0.113，公司名 0.465 | 整串 2/2819，公司名同样是 2/2819。红章展开条上会把「有限」读成「有阳」 | 约 36 ms |

零样本把合成方章从 0 拉起来了，但真实圆章公司名从 0.465 掉到大约 0，所以没有换成发布权重。墨色转黑的预处理在方章上是 0.190，在 Paddle 上是 0。随后用官方 `PP-OCRv5_mobile_rec_pretrained.pdparams` 在 Paddle 训练条（重复 4 次）、合成条、自由环文、自由姓名格和姓名章上微调，字表仍是官方 18,383 字，条带高 48 宽 576，最长标签 48（Paddle 标签最长 40，官方默认 25 会丢掉六成）。Paddle 验证整串在第 800 步到 0.113，第 1200 步 0.115，第 4 个 epoch 的最好验证整串是 0.141，编辑距离 0.685。这仍是 Paddle 验证集上的整串，不是公司名（第一个数字前截断）指标。最终条带分数以导出的 ONNX 为准；只有公司名不掉、圆章和环文不明显变差时才替换 `models/seal_rec.onnx`，原来的灰度 CTC 留作 `seal_rec_fast.onnx`。

用户交来的裁图是铺满画面的整枚章。已发布检测器在 63 张测试裁图上召回是 0：预测框是章内部的碎片，和整枚章的 IoU 到不了 0.5。训练里因此加了大尺度裁剪，并把 156 张训练裁图按四种尺度贴到页面上。测试裁图不进训练。这批图上的文字草稿（已发布 CTC 和零样本 PP-OCRv5）整串没有一张一致，目视也不能稳定读出环文，所以没有写成金标准文字。
