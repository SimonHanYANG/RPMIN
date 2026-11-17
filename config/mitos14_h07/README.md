# RP-MIN · MITOS-14 H10 + H07 复现

| 倍率 | H10_00 Y-PSNR | Y-SSIM | 最终微调权重 |
| --- | ---: | ---: | --- |
| ×2 | 28.27777273 dB | 0.88740333 | `RP-MIN_MITOS14_x2_Y-PSNR28.2778_SSIM0.8874.pt` |
| ×4 | 27.28777164 dB | 0.78834452 | `RP-MIN_MITOS14_x4_Y-PSNR27.2878_SSIM0.7883.pt` |

训练参数、数据划分、倍率对应的锁定权重和评分参数保存在 `experiment.json`。`results.csv` 保存原始结果的完整精度。`.pt` 文件名里的成绩对应整套推理配置。该配置读取两个倍率的检查点，执行跨倍率融合、相机校正和最终高频融合。

## 用原始训练完成的权重复测

在 `dlsimon2` 运行以下命令。`--out` 指向新目录，每个倍率会输出 `summary.json` 和 `per_image.csv`。复测检查测试配对清单及锁定权重的 SHA-256，同时核对评分参数和表中数值。

```bash
cd /root/autodl-tmp/sr-baselines
project=RPMIN-e79d2b3
cfg="$project/config/mitos14_h07/run.py"
CUDA_VISIBLE_DEVICES=0 torch-env/bin/python "$cfg" test \
  --archived-weights "$project/weights/mitos14_h07" \
  --out runs/mitos14-h07-retest
```

测试使用 `H10_00` 上每倍率同一批 132 个 128×128 LR 图块。每块读取 16 像素 LR 邻域，执行 D4 集成后裁回原评分区域。BT.601 Y 的数字范围为 16–235，PSNR 峰值为 255，HR 评分裁边为 0。最终模型的高频融合权重为 ×2 `0.25`、×4 `0.5`。结果对应 Y 通道。

两个倍率的 `summary.json` 和 `per_image.csv` 均与原实验逐字节一致。两份已归档最终微调权重都是在第二阶段第 400 步由验证集选出的检查点。[查看归档复测记录](verification/archived/)

## 重跑训练与测试

复现从已归档的上一轮 MITOS-14 `context-trained/best.pt` 起步。第一阶段在 H10_01–03 和 H07_00–01 上混合训练 3,600 步，起始学习率 `6e-5`。第二阶段从混合权重继续在 H10_01–03 上适配 1,000 步，起始学习率 `3e-5`。训练用种子 207，移位采样种子 208。两个阶段按 H10_04 验证 PSNR 保存最佳检查点，最终频段融合也在 H10_04 上选定。H10_00 只在测试命令中评分。三个划分均使用 H10 的不同视野，因此保留集衡量同一张切片上的未见视野。

```bash
cd /root/autodl-tmp/sr-baselines
cfg=RPMIN-e79d2b3/config/mitos14_h07/run.py
CUDA_VISIBLE_DEVICES=0 torch-env/bin/python "$cfg" train --run runs/mitos14-h07-repeat
CUDA_VISIBLE_DEVICES=0 torch-env/bin/python "$cfg" test \
  --run runs/mitos14-h07-repeat --out runs/mitos14-h07-repeat-test
```

该流程读取原实验保留的配准图像。H10 位于 `runs/mitos14-h10-train/x{2,4}/references/`，H07 位于 `runs/mitos14-rpmin-h07/x{2,4}/h07-references/`。`pipeline/download_mitos_h07.py` 和 `pipeline/prepare_mitos_h07.py` 保存了本轮下载与拼接方式，H07 拼接后通过项目 `align-references` 命令逐倍率配准，参数为 `--splits train --margin 12 --refine`，`--weights-cache` 取配置中的 `lightglue_cache`。原环境为 Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128 和 RTX 5090，项目代码对应 `experiment.json` 中的提交。

在该服务器完整执行一次重训和测试后，×2 为 `28.27770956 dB / 0.88741152`，×4 为 `27.28773444 dB / 0.78838094`。两个倍率与原实验的 PSNR 差均小于 `0.0001 dB`，SSIM 差均小于 `0.00004`。[查看重训日志和测试结果](verification/retrained/)

## 权重归档

项目下的 `weights/mitos14_h07/` 包含两份命名后的最终权重、前序检查点、辅助校正参数、验证决策、原始测试明细和 `inventory.json`。`inventory.json` 逐项记录 SHA-256 和原路径。归档仅包含模型及复现记录，复跑还需要服务器上的原配准图像。新建归档时可运行 `torch-env/bin/python "$cfg" archive --out <新目录>`。
