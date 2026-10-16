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

网页印章图用搜索引擎采集，图片留在 `/home/mayong/datasets/web_seals/`，来源 URL 写在 `training/manifests/web_seals_sources.jsonl`：

```bash
python training/scripts/collect_web_seals.py \
    --out /home/mayong/datasets/web_seals \
    --manifest training/manifests/web_seals_sources.jsonl
```

脚本会问 Bing、百度和 360 图片。Google 图片搜索会 302 跳转，这里不抓。这台机器上百度直接拒绝爬虫。中文查询和 “official round stamp on document” 上 Bing 回的是空页；英文的 notary、corporate seal 查询 Bing 有结果。2026-10-16 这一批留下 401 张：360 图片搜索 350 张，Bing 51 张，百度 0 张。来源页的版权仍归原站，这些图只做内部训练和评测。清单里的 `model_text` 是现有模型的预标注，`spot_check` 只覆盖抽看过的几张。

ReST 的测试图没有转写，脚本不使用 `test_images.zip`。有标注的 5,000 张里大约 10% 按 id 留作验证。检测框是贴回文档页的整枚裁图；标题多边形只说明字在哪，不当作印章外框。
