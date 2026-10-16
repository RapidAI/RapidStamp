# 训练数据放在哪

图像和 checkpoint 不进 git。训练机上 `/home` 和 `/home/mayong` 不是同一块盘，大文件只放 `/home/mayong`。

| 路径 | 内容 |
| --- | --- |
| `/home/mayong/datasets/rest/rest_icdar2023.zip` | ICDAR 2023 ReST。包内没有许可证文件，不要再分发 |
| `/home/mayong/datasets/paddle_seal/paddle_seal.zip` | PaddleOCR 印章包。上传目标就是这个路径 |
| `/home/mayong/datasets/web_seals/` | 搜索引擎采到的真实盖章图。只在训练机上，不进 git |
| `/home/mayong/rapidstamp/synth` | 本仓库生成器写出的合成页 |
| `/home/mayong/rapidstamp/real/rest` | `import_rest.py` 的输出 |
| `/home/mayong/rapidstamp/real/paddle` | `import_paddle.py` 的输出 |
| `/home/mayong/rapidstamp/mix` | 合成和真实数据的符号链接，以及并集字表 |
| `/home/mayong/rapidstamp/free_rings` | 从整张字表抽的环文条带，用来减弱模板词偏置 |
| `/home/mayong/rapidstamp/mix_ring` | 识别重训用的混合条带。ReST 训练条带重复 4 次 |
| `/home/mayong/rapidstamp/checkpoints` | 微调权重 |
| `/home/mayong/rapidstamp/models` | 导出的 ONNX |

已有的 PyTorch 环境留在原处，不再往小盘装一份 CUDA 版。新的数据目录不要建到 `/home/znsoft`。

准备和微调：

```bash
python training/scripts/import_rest.py \
    --src /home/mayong/datasets/rest/rest_icdar2023.zip \
    --dst /home/mayong/rapidstamp/real/rest

python training/scripts/import_paddle.py \
    --src /home/mayong/datasets/paddle_seal/paddle_seal.zip \
    --dst /home/mayong/rapidstamp/real/paddle

python training/scripts/finetune_real.py \
    --synth /home/mayong/rapidstamp/synth \
    --rest /home/mayong/rapidstamp/real/rest \
    --paddle /home/mayong/rapidstamp/real/paddle \
    --init-det /path/to/best_det.pt \
    --init-rec /path/to/best_rec.pt
```

Paddle 的 zip 已经在上面的路径里，`import_paddle.py` 写出了训练 7,181 条、验证 2,819 条。识别微调用的是 `train_ring_rec.py --paddle`，不是 `finetune_real.py`。

环文整串识别另训一个更宽的识别器，检测权重不动：

```bash
python training/scripts/synth_free_rings.py \
    --charset models/charset.txt \
    --out /home/mayong/rapidstamp/free_rings

python training/scripts/train_ring_rec.py \
    --synth /home/mayong/rapidstamp/synth \
    --rest /home/mayong/rapidstamp/real/rest \
    --free /home/mayong/rapidstamp/free_rings \
    --det /home/mayong/rapidstamp/checkpoints/finetune/best_det.pt \
    --charset models/charset.txt
```

`train_ring_rec.py --paddle` 会把 Paddle 条带、自由环文和自由姓名格并进同一轮识别训练。2026-10-16 这一轮用了已经导入的 Paddle 目录，字表扩到 2,816。指标见 `models/metrics.json`。

检测微调是 `training/scripts/train_yolox_det.py`。它在训练机下载 Megvii 的 `yolox_nano.pth` 到 `/home/mayong/rapidstamp/weights/`（`*.pth` 不进 git），把合成、ReST、Paddle 和网页训练页链到 `mix_yolox`，并在图上画红蓝字、线条、表格和 logo。`*.pt` 同样不进 git。

PP-OCRv5 mobile rec 的条带打分是 `training/scripts/eval_ppocr_strips.py`，微调列表是 `training/scripts/prepare_ppocr_finetune.py`，配置是 `training/configs/ppocrv5_mobile_seal.yml`。预训练 `pdparams` 和 Paddle 环境放在 `/home/mayong`，不进 git。

方形名章条带用 `training/scripts/synth_name_seals.py` 从现有字表里抽可绘制汉字，写出阴阳刻条带。2026-10-16 在 `/home/mayong/rapidstamp/name_seals` 生成了 12,000 条训练条带。接着训出的识别器没有替换 `models/seal_rec.onnx`，原因见研究报告第 9 节。

网页印章图用搜索引擎采集，图片留在 `/home/mayong/datasets/web_seals/`，来源 URL 写在 `training/manifests/web_seals_sources.jsonl`：

```bash
python training/scripts/collect_web_seals.py \
    --out /home/mayong/datasets/web_seals \
    --manifest training/manifests/web_seals_sources.jsonl
```

脚本会问 Bing、百度和 360 图片。Google 图片搜索会 302 跳转，这里不抓。这台机器上百度直接拒绝爬虫。中文查询和 “official round stamp on document” 上 Bing 回的是空页；英文的 notary、corporate seal 查询 Bing 有结果。2026-10-16 这一批留下 401 张：360 图片搜索 350 张，Bing 51 张，百度 0 张。来源页的版权仍归原站，这些图只做内部训练和评测。清单里的 `model_text` 是现有模型在原图上的自动结果。这一版平均每张标出十几个框，环文经常塌成同一小撮字，不能直接当评测或训练标签。`spot_check` 只覆盖抽看过的 9 张。

固定划分写在 `training/manifests/web_seals_split.jsonl`，不随新增文件重抽：sha256 前 8 位转成整数后模 10，小于 3 为测试，其余可训练；拼图只进训练。文章页和商品图整类排除，不用作空负样本，因为页面上往往还有章，标成空会教检测器把章压掉。只有审成「无关」的页是空负样本。框是红蓝墨连通域，再对上审阅时给的大致区域；彩色页上没有红蓝墨的框丢掉。当前清单是训练 137 页、测试 52 页、排除 191 页、没有可用框而跳过 21 页。测试集上的文字几乎是空的，整串命中不以这 52 页为准。图片在 `/home/mayong/rapidstamp/real/web`（训练）和 `/home/mayong/rapidstamp/real/web_test`（测试，目录名是 `det/val`，这样评测脚本不用改）。这 52 页不再重抽。

用户另交的一批裁图在 `/home/mayong/datasets/user_seals/`（`images/` 与 `compound-img/`，229 张 png，压缩包里没有转写）。导入脚本是 `training/scripts/import_user_seals.py`。框是红、蓝或灰扫描上的深色墨外接矩形，每张图一枚章。形状按框的长短边和四角是否还有墨来定；这一批四角是纸、长短边接近，类别都是 `round_official`。没有可靠转写，金标准文字留空，不用模型输出回填。同一拍摄（文件名去掉末尾 `_N`）或像素完全相同的文件合成一组，按组名的 sha256 前 8 位模 10，小于 3 为测试，其余训练。像素完全相同的副本只保留一份，避免测试图再进训练。清单是 `training/manifests/user_seals_split.jsonl`。图片在 `/home/mayong/rapidstamp/real/user_seals`（训练）和 `/home/mayong/rapidstamp/real/user_seals_test`（测试）。这批的精确率、召回和 mAP 单独报告，不并进上面那 52 页。2026-10-16 用已发布检测器（得分阈值 0.25）在 63 张测试裁图上：精确率 0（0/2）、召回 0（0/63）、mAP50 0、约 24 ms/张。把阈值降到 0.05 后仍没有一个预测框和金标准的 IoU 达到 0.5，打出来的是章内部的小碎片，而不是铺满画面的整枚章。

训练侧补了两种大尺度。`DetDataset` 在训练时有一半概率把画面裁到某一枚章的周围，使章的长边占裁图的 0.62 到 0.98。另外只拿那 156 张训练裁图，按长边占页面的 0.22、0.45、0.72、0.96 贴到 640 像素的纸上，写到 `/home/mayong/rapidstamp/real/user_pages`（624 页）。63 张测试裁图不贴、不进训练。脚本是 `training/scripts/paste_user_seal_pages.py`。

文字仍没有金标准。已发布 CTC 直接读这 63 张裁图，零样本 PP-OCRv5 也读了同一批极坐标条。两条草稿整串一致的是 0 张，最长公共前缀平均不到 1 个字。展开条的目视核对也不能稳定读出环文，所以没有把草稿写进标签。`training/scripts/read_seal_crops.py` 只产生草稿。

ReST 的测试图没有转写，脚本不使用 `test_images.zip`。有标注的 5,000 张里大约 10% 按 id 留作验证。检测框是贴回文档页的整枚裁图；标题多边形只说明字在哪，不当作印章外框。
