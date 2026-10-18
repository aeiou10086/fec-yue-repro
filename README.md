# YUE — Fusion Equilibrium Challenge (NeurIPS 2026) 复现包

Team: **YUE** · Competition: https://www.codabench.org/competitions/17739/
Starter kit base: github.com/Sophelio/fusion-equilibrium-challenge-starter (MIT, unmodified parts not re-distributed here; only our additions)

## 方法概述

- **输入**: 21 路 DIII-D 线圈电流指令 + 修正后的等离子体电流（`data_fixes.fix_d3d_ip_times`，官方勘误 #6）。不使用任何磁场诊断测量、不读取任何 `efit_*` 目标字段或 `magnetics_dsep`。
- **ψ 模型**: 输入标准化 → PCA(32) 压缩 65×65 通量图 → MLP(256,128) 回归 PCA 系数（系数亦标准化）。
- **标量头**: q95 = ExtraTrees；betaN = MLP。两标量与 ψ 独立预测。
- **拆分**: 按完整放电（shot）划分，160 train / 12 validation / 24 audit，seed=42，文件名+SHA256 记录于 `weights/manifest.json`。
- **验证**: 官方 `local_score.py`（metric v3.2.0）在 12 个 held-out 放电上 S=0.920899；float16 存储 S=0.920873（Δ=-0.000026）。24 个独立审计放电支持标量头选择。
- **存储**: ψ float16（官方推荐，相对精度），标量 float32。

## 环境重建

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
```

依赖基于 starter kit 的 `uv.lock` 提取（见 env/requirements.txt）。

## 训练入口（从零重建模型）

```bash
# 前置: 下载 DIII-D 训练数据到 ../hf_dataset/data/diii_d_train/*.parquet
# (数据: huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge, CC BY 4.0)
cd src
python train_offline.py --train-shots 160 --valid-shots 12          # → PCA+MLP ψ + ExtraTrees 标量
python improve_scalars.py                                            # → betaN MLP 标量头
python audit_scalars.py                                              # → 24-shot 独立审计
# 最终模型 (v1.1): 259-shot z-floor PCA 配方 + hybrid 标量头 = weights/final_zfloor_hybrid.pkl
# (via Release v1.1; 训练脚本 src/train_full271_zfloor.py; local 12-shot S=0.9361)
```

## 推理入口（从权重重新生成本次提交的 NPZ）

```bash
# 前置: 缓存公测输入 (HTTP Range 仅取输入列; 或用官方 streaming)
python src/prepare_public_inputs.py            # → hf_dataset/input_cache/diii_d_public_test/*.npz
python src/build_public_npz.py --dtype float16 # → artifacts/development_candidate/diii_d_public_test.npz
```

## 数据来源声明（per-machine）

- **DIII-D**: 仅 `Sophelio/fusion-equilibrium-challenge` 数据集（CC BY 4.0），revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`。未使用其他托卡马克档案、外部数据集或预训练模型。
- **MAST**: 本提交未包含 MAST 预测（仅参加 Challenge 1）。未使用任何 MAST 平衡重建数据。
- AI 编程辅助（Claude Code / pi）已用于代码开发，方法如本文件所述。

## 硬件与耗时声明

- Apple M-series MacBook, 16GB RAM, 8 逻辑核心, **纯 CPU, 无 GPU**
- ψ-MLP 训练（38k 帧）: ~10 分钟; 标量 MLP: 4.9 秒; 端到端含数据解析: ~4 分钟; 峰值 RSS ~1.9GB
- 符合 compute-light badge 标准（单 CPU 2 小时内端到端）
权重从git移除，改由GitHub Release分发 (v1.0=mlp_hybrid冻结版, v1.1=final_zfloor_hybrid 最终提交版)
