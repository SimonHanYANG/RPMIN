# RP-MIN · MFA-Cell 复现

| 倍率 | 保留集 RGB PSNR / SSIM | 论文 Table 2 | 归档权重 |
| --- | ---: | ---: | --- |
| ×2 | 31.43036143 dB / 0.87641735 | 31.24 dB / 0.86 | `RP-MIN_MFA-Cell_x2_W32_RGB-PSNR31.4304_SSIM0.8764.pt` |
| ×4 | 27.67971312 dB / 0.68710036 | 27.68 dB / 0.67 | `RP-MIN_MFA-Cell_x4_ensemble_W32.pt` 和 `RP-MIN_MFA-Cell_x4_ensemble_W48.pt` |

训练、数据和评分参数在 [experiment.json](experiment.json)，原始均值在 [results.csv](results.csv)。归档目录的 [inventory.json](../../weights/mfa_cell/inventory.json) 记录文件名、SHA-256、训练选中的轮次及权重与结果的对应关系。×4 表中分数由两份权重的预测等权平均得到。

## 使用归档权重复测

从 `RP-MIN` 项目根目录运行。若原始数据位于其他位置，用 `--dataset` 指向含 `ciliated epithelial cell for super resolution/` 子目录的上一级目录。复测使用 Python 3.12、PyTorch 2.8.0+cu128、TorchVision 0.23.0+cu128 和 OpenCV 4.13.0。

```bash
python config/mfa_cell/run.py verify
python config/mfa_cell/run.py test --device cuda:0 --run runs/mfa_cell_repro
```

`verify` 检查归档权重哈希。`test` 首次根据已保存的配准变换重建保留集，核对源图和配准像素。复用已生成的保留集时重新核对配准像素。默认写入 `runs/mfa_cell_repro/test-archived/`，包含 `comparison.csv` 和各倍率的 `summary.json`、`per_image.csv`。归档复测的误差容限是 PSNR `0.0001 dB`、SSIM `0.00001`。

已在 `dlsimon2` 用归档权重和重新生成的配准保留集复测。两个倍率的逐裁块 CSV 均与原实验逐字节一致。[查看复测结果](verification/test-archived/comparison.csv)

两个倍率都针对同一批 4 个保留视野，每个种子使用 308 个 LR 网格窗口。窗口为 128×128，起点 `(1, 1)`，横纵步长为 32 和 25。PSNR 和 SSIM 均在 RGB `[0,1]` 上计算，HR 裁边为 0。结果对种子 101、102、103 取平均。

- ×2 使用宽度 32 模型，逐块输入，batch 4，每块做 D4 集成。
- ×4 使用宽度 32 和 48 两个模型。每个模型先对整视野做 D4 集成，再平均两个 RGB 预测。最终在固定网格裁出的 HR 窗口上评分。

## 从头训练再测试

```bash
python config/mfa_cell/run.py train --device cuda:0 --run runs/mfa_cell_repeat
python config/mfa_cell/run.py test --device cuda:0 --run runs/mfa_cell_repeat --trained
```

三个模型均从随机初始化训练。宽度 32 的 ×2 和 ×4 各训练 200 轮，种子为 42。宽度 48 的 ×4 训练 120 轮，种子为 87。每轮每个训练视野随机抽取 16 个 128×128 LR 块，batch 4。训练以固定验证集 PSNR 选择 `best.pt`。再次运行 `train` 可从同目录 `latest.pt` 续跑。`--trained` 会测试本次生成的权重，并记录与归档成绩的差异。

原始配置、逐轮训练记录和检查点选择分别见 [×2 宽度 32](training/x2-w32/)、[×4 宽度 32](training/x4-w32/) 和 [×4 宽度 48](training/x4-w48/)。

训练、验证和保留测试按 [split.csv](split.csv) 固定为 34、4、4 个视野。原实验用 SuperPoint、LightGlue 和 ECC 生成配准变换。这里复用归档矩阵，将 HR 投影到 LR 坐标，移除双方共同的 12 像素 LR 边界，并核对生成像素与原实验一致。所有派生图写入 `--run`。
