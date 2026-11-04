# RapidStamp 300M 学生模型 VLEnc300M（vlenc300m_30k_20261104）

状态：15k 训练完成（2026-11-03 20:59 UTC）。**30k 续训完成（2026-11-04 16:07 Asia/Shanghai）**。最新权重 `/home/rapidstamp_data/checkpoints/vlenc300m_30k_20261104/`（best.pt = last.pt = step 30000）。15k 保留在 `/home/rapidstamp_data/checkpoints/vlenc300m_20261103/`。**未部署**，生产代码与模型路由未改动。本文档供复现与后续决策。

## 1. 结论速览

金标 gold_v2 环文本 exact（越高越好），顺序 single227 / unseen59 / orig63 / long45：

| 模型 | 参数 | single227 | unseen59 | orig63 | long45 | GPU 延迟/crop |
|---|---|---|---|---|---|---|
| **VLEnc300M step30000（最新 best.pt）** | 302M | **168** | 46 | **54** | **20** | ~58–64 ms（同结构，沿用 15k 实测） |
| VLEnc300M step15000 | 302M | 167 | 45 | 54 | 18 | 64 ms fp32 / 58 ms bf16 |
| VL-FT v1 teacher（paddleocr_vl16_ft_20261101） | ~0.9B | 159 | **48** | 51 | 18 | ~535 ms |
| PaddleOCR-VL-1.6 原始 zero-shot | ~0.9B | 144 | 39 | 50 | 18 | — |
| SVTRv2 distill3（28M） | 28M | 144 | 41 | 47 | 17 | ~32 ms（同结构 distill1 实测 fp32） |

- 30k 相对 15k：+1 / +1 / 0 / +2。val_real 574/600，15k 为 570/600。
- 30k 相对 VL-FT v1 teacher：+9 / −2 / +3 / +2（unseen59 46 vs 48）。
- 30k 相对 distill3：+24 / +5 / +7 / +3。
- 15k 相对 teacher：single227 +8、orig63 +3、long45 持平、unseen59 −3（45 vs 48）。15k 相对 distill3：+23 / +4 / +7 / +1。
- 30k 的 1-NED 约 0.9419 / 0.9642 / 0.9883 / 0.8568。15k 为 0.9393 / 0.9484 / 0.9883 / 0.8658；15k 的 val_real 1-NED 0.9871（570/600 exact）。orig63 的 1-NED 相同。long45 的 exact 升到 20，1-NED 低于 15k。
- multi8 仍为 0/8（多行/多环样本，结构上只输出一条序列，见已知问题）。
- 续训中途 step 24000 的金标 exact 为 170/47/55/20（val_real 573/600），高于 step 30000 的金标。best 只按 val_real 选，24000 没有单独 ckpt。30k 的 val_real 574 更高，所以 best.pt = last.pt = step 30000。

## 1.1 续训到 30k（vlenc300m_30k_20261104）

- 完成：2026-11-04 16:07 Asia/Shanghai。从 15k 的 step 15000 接着计数，训到 step 30000。
- 起点目录保持不动：`/home/rapidstamp_data/checkpoints/vlenc300m_20261103/`（best.pt = last.pt = step 15000，md5 dc714f81a089aae9582314b642351a76）。
- 新目录：`/home/rapidstamp_data/checkpoints/vlenc300m_30k_20261104/`（best.pt = last.pt = step 30000）。
- 学习率是新日程，其余与第 6 节相同。warmup 500 步升到 15k 峰值的 0.3 倍（视觉 3e-5 → 9e-6，合并/解码 3e-4 → 9e-5），然后 cosine 到 step 30000。15k 日程是 warmup 1000、余弦下限 2%。30k 是否仍用 2% 下限，本次没有另记。
- 数据、硬标签 CE、bs 32、gc 0、bf16、AdamW、best 按 val_real、金标只记录，都与 15k 相同。脚本仍是 `/home/rapidstamp_data/vlenc300m_20261103/train_vlenc300m.py`。
- 结构未改。延迟沿用第 9 节，未重跑 `bench_latency.py`。GPU 约 58–64 ms（bf16 57.5 / fp32 63.7）。
- 仓库里没有 `docs/recipes/`。15k 的复现副本在服务器 `datasets_final_20261103/recipes/vlenc300m_20261103/`（脚本、命令、本文、eval、latency、data_manifest.md5）。30k 复用这份副本的脚本和数据清单；差别是起点、输出目录，以及上面的学习率日程。

复现这次 30k，或从 30k 再往后续：`--out` 用 30k 目录，15k 目录保持不动。`--resume 1` 读的是 `--out` 里的 `resume.pt`。跨目录从 15k 接着训时，先把 15k 的权重放进新目录再开训。15k 的命令行没有学习率参数，峰值写在脚本里；复现 30k 时让脚本里的峰值是视觉 9e-6、合并/解码 9e-5（原先的 0.3 倍），warmup 500，cosine 走到 step 30000。

```bash
cd /home/rapidstamp_data/vlenc300m_20261103
# 复现 30k：脚本内峰值已改为 0.3× 之后再跑。--out 是新目录。
/home/znsoft/miniconda3/bin/python train_vlenc300m.py \
  --out /home/rapidstamp_data/checkpoints/vlenc300m_30k_20261104 \
  --steps 30000 --warmup 500 --bs 32 --gc 0 --resume 1
# 再从 30k 往后续：同一 --out，--resume 1，把 --steps 提高到新目标。
```

15k 从头训练的命令仍是第 7 节。

## 2. 结构

- 视觉编码器：VL-FT v1 teacher（`checkpoints/paddleocr_vl16_ft_20261101/best`）的 SigLIP 视觉塔 `visual.vision_model.*`，**截取前 18/27 层**；hidden 1152、16 头、patch 14、2D RoPE，位置嵌入按 32×32 网格插值。
- 输入 448×448 RGB，归一化 (x−0.5)/0.5 → 32×32=1024 patch。
- 2×2 patch 合并：LayerNorm(4608) + Linear(4608→512) + 可学位置嵌入（256 token）。
- 解码器：NRTR 式自回归 Transformer 解码器，4 层、d512、8 头、FFN 2048、pre-norm、GELU、dropout 0.1；可学位置 (maxlen+2)。
- 词表 6627（与 SVTRv2 distill3 `vocab.json` 相同，md5 e1f751ad9f221fd9fab0bb17101472c7），EOS=0，BOS=C−2，PAD=C−1；maxlen 40；推理为 greedy。
- 参数 301.96M（视觉 275.83M + 合并/解码 26.13M），`arch.json` 见 recipe。
- 编码前向为自写 batched 实现（SDPA），已用 `verify_encoder.py` 与 HF `PaddleOCRVisionTransformer` 对齐（27 层全量 maxdiff 0.0），patch 顺序 row-major。

## 3. 初始化

- 视觉 18 层 + embeddings + post_layernorm 直接载入 teacher safetensors（OCR 预训练 + 印章 FT），是此前 ViT-B 从零训练失败后的关键改动。
- 合并层、解码器随机初始化（位置嵌入 std 0.02）。

## 4. 训练方法

- 硬标签蒸馏：teacher 伪标签 + 人工/投票真实标签 + 合成数据混合训练，CE（label smoothing 0.1，忽略 PAD），teacher forcing。
- soft-label（logit 级）KD 在 28M 上无增益（softkd 142/38/47/18 vs distill3 144/41/47/17），故 300M 仅用硬标签。
- 增强：RandomAffine(±10°, 平移 4%, 缩放 0.88–1.08, shear 5, p0.7)、ColorJitter(0.4,0.4,0.4,0.05, p0.6)、灰度 p0.15、高斯模糊 p0.2、降采样再放大 p0.15。
- 选点：独立 val_real 600 条（`md5(path)` 排序取自 train_clean_v3 train split + ReST train GT，排除金标文本，从训练中剔除，`val_real.jsonl`）。**金标只记录、不用于选点**。15k 结束时 best 与 last 均为 step 15000。30k 续训后 best 与 last 均为 step 30000，见第 1.1 节。

## 5. 数据（与 svtrv2_distill3 同源，金标文本全部过滤，长度 ≤40）

| 名称 | 文件 | md5 | 采用行数 × 重复 |
|---|---|---|---|
| real_v3 | `label_clean/vote3_full_20261102/train_clean_v3.jsonl`（根 `web_seals_cn_all/crops/`） | 37b4f9170da93df28193fb27c1f3900a | 13,077 × 2 |
| teacher PL + ReST + AI Studio | `label_clean/distill3_20261103/pseudo_v3_combined.jsonl` | a2318069f8f153db13f574652c12ef55 | 31,154 × 2 |
| supp | `labels/supplement_nogold_vote_fix_reviewed_20261101.jsonl` | 7b0340ef0510af677e23558cdf724277 | 84 × 4 |
| synth_gs2 | `synth_corpus/gold_style_v2_20261030/labels.jsonl` | fb738ef8cdebbb4535de8b6621aa1737 | 6,000 × 1 |
| synth_gs2_long | `synth_corpus/gold_style_v2_long_20261030/labels.jsonl` | 4b07133f1a69c1e4c5e8f8055ba46311 | 14,000 × 1 |
| synth_long_v3 | `synth_corpus/long_v3_20261031/labels.jsonl` | 55ebff499c4e25102a311634db30ca30 | 12,000 × 1 |
| synth_doc | `synth_corpus/vitb_pretrain_2m/shards/54816.jsonl`（前 24 万行 seed1 打乱取 12 万） | 09a69dcc6823ff1e05cb3b5db6d7cd1f | 119,997 × 1 |
| 金标（仅评测） | `gold_v2/labels.jsonl` | 2094e8c558ca08c9cb0469ab5e92bbc6 | 235 |

路径均相对 `/home/rapidstamp_data/`。完整清单见 `datasets_final_20261103/MANIFEST.md5` 与 `recipes/svtrv2_distill3_20261103/data_manifest.md5`。

## 6. 超参

15k 那次：bs 32（无梯度检查点），15,000 步，warmup 1,000 + cosine（下限 2%），AdamW：视觉 lr 3e-5、合并/解码 lr 3e-4，wd 0.05（1D/pos/emb/norm 参数 wd 0），grad clip 1.0，bf16 autocast，num_workers 10，每 3,000 步评测。显存峰值 31.4 GB，1.61 s/step（RTX 6000 Ada 48GB，约 6.7 h）。30k 续训只改了步数和学习率日程（第 1.1 节），优化器其余项与这里相同。

## 7. 命令

```bash
cd /home/rapidstamp_data/vlenc300m_20261103
# 训练（排队脚本：等 GPU 空闲 ≥33GB，失败自动 --resume 重试 3 次，run.log 以 ALL DONE 结尾）
setsid nohup bash run_vlenc300m.sh &
# 等价于
/home/znsoft/miniconda3/bin/python train_vlenc300m.py --out /home/rapidstamp_data/checkpoints/vlenc300m_20261103 \
  --steps 15000 --warmup 1000 --bs 32 --gc 0 --resume 1
# 延迟
/home/znsoft/miniconda3/bin/python bench_latency.py   # 写 checkpoints/vlenc300m_20261103/latency.json
```

环境：`/home/znsoft/miniconda3/bin/python`（torch 2.13，transformers 5.19，含 `paddleocr_vl` 模型代码），评测函数 `vl16_20261101/vlcommon.py`（score_gold / ned1）。30k 续训命令见第 1.1 节。

## 8. 评测曲线（gold exact，val_real exact/600）

| step | single227 | unseen59 | orig63 | long45 | multi8 | val_real |
|---|---|---|---|---|---|---|
| 3000 | 86 | 22 | 35 | 1 | 0 | 439 |
| 6000 | 128 | 34 | 46 | 7 | 0 | 523 |
| 9000 | 152 | 45 | 51 | 13 | 0 | 558 |
| 12000 | 161 | 43 | 54 | 19 | 0 | 569 |
| 15000 | **167** | 45 | 54 | 18 | 0 | **570** |

15k 结束时 gold single227 仍在上升（161→167），val_real 569→570 已趋平。每步预测见 `checkpoints/vlenc300m_20261103/eval_step*.json`。

30k 续训从 step 15000 接着计数。本文只记下 step 24000（金标峰值，未单独存盘）和 step 30000（best = last）。18000 / 21000 / 27000 没有写入本文。

| step | single227 | unseen59 | orig63 | long45 | multi8 | val_real |
|---|---|---|---|---|---|---|
| 24000 | 170 | 47 | 55 | 20 | — | 573 |
| 30000 | **168** | 46 | 54 | **20** | 0 | **574** |

step 24000 的金标高于 step 30000，val_real 573 低于 574。best 按 val_real 选，所以磁盘上的 best.pt 是 step 30000，与 last.pt 相同。

## 9. 延迟（`bench_latency.py`，与 distill3 同协议：gold 前 40 张 crop，bs1，3 次 warmup，中位数 / p95）

硬件：NVIDIA RTX 6000 Ada 48GB；CPU Intel i7-8700K（6C12T）。端到端 = 编码 + greedy 自回归解码（maxlen 40，遇 EOS 停），448×448，bs1。

| 设备 | 端到端 中位 / p95 (ms) | 仅编码器 中位 / p95 (ms) |
|---|---|---|
| GPU bf16 autocast | 57.5 / 65.7 | — |
| GPU fp32 | 63.7 / 71.0 | 34.9 / 35.3 |
| CPU 8 线程 fp32 | 1828.4 / 1848.9 | 1696.4 / 1696.9 |
| CPU 1 线程 fp32 | 5734.7 / 5814.2 | — |

对比：VL-FT teacher GPU 约 535 ms/crop（约 8× 慢）；28M SVTRv2 GPU fp32 约 32 ms、CPU 8 线程约 155 ms。300M 在 GPU 上可用，CPU 上约 1.8 s/crop，仅适合离线。结果 `checkpoints/vlenc300m_20261103/latency.json`。30k 与 15k 结构相同，延迟不变，上表即 30k 的延迟（GPU 约 58–64 ms）。30k 没有另跑 `bench_latency.py`。

## 10. 产物

- `checkpoints/vlenc300m_30k_20261104/`：最新权重。best.pt = last.pt = step 30000。15k 目录没有被这次续训覆盖。
- `checkpoints/vlenc300m_20261103/`：15k。best.pt（1.21 GB，step15000，md5 dc714f81a089aae9582314b642351a76）、last.pt、resume.pt（含优化器，3.6 GB）、arch.json、vocab.json、val_real.jsonl、eval_step*.json、train.log、latency.json。
- 复现副本：`datasets_final_20261103/recipes/vlenc300m_20261103/`。30k 复用这份脚本和数据清单，见第 1.1 节。仓库没有 `docs/recipes/`。

## 11. 已知问题

- unseen59 仍低于 teacher：30k 为 46/59，15k 为 45/59，teacher 为 48/59。样本小，需在更大新数据上确认泛化。
- multi8 = 0/8：单序列解码，无法处理多环/多行，需要上游拆分或多序列头。30k 没有改变这一点。
- 自回归 greedy，延迟随输出长度线性增加。long45 在 30k 为 20/45（15k 为 18/45），仍是短板。30k 的 long45 1-NED 约 0.8568，低于 15k 的 0.8658。
- 模型为 PyTorch 自定义类，尚未导出 ONNX / 未量化；CPU 延迟偏高，不适合纯 CPU 部署。
- 视觉层截取 18/27 层是一次性选择，未做层数消融。
- val_real 来自投票/人工标签，可能残留少量噪声。

## 12. 下一步（均需用户批准）

1. 与生产 A（parseq_reader_large_clean_csd + ≥16 字路由）在同一金标与线上样本上做端到端对比，决定是否作为 GPU 端候选。
2. 续训到 30k 已完成（2026-11-04，第 1.1 节）。相对 15k，金标只多了 +1/+1/0/+2，val_real 574 vs 570。step 24000 的金标更高（170/47/55/20）但未单独存盘。24 层视觉塔仍未做。
3. ONNX 导出 + fp16/INT8，评估 KV cache 解码加速。
4. 用本模型重新打伪标签，蒸馏下一代 28M 学生。
