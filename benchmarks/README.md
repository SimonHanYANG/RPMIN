# dlsimon2 的 MFA-Cell ×2 和 ×4 对比实验

归档表格包括 [MFA-Cell 双倍率 RGB](mfa_cell_rgb.csv)、[MITOS-14 H10 基础 RGB](mitos14_h10_rgb.csv) 和 [MITOS-14 H10 相机校正 RGB](mitos14_h10_camera_rgb.csv)。[Figure 5 定性对比](figure5/) 保存了视野划分、裁剪坐标、配准映射和模型指纹。

下文命令使用原服务器工作区。复跑需要该工作区的配准图、依赖环境、上游仓库及初始化权重。将本目录下的 `*.py`、`*.sh`、`figure5_rois.json`，以及 `config/mitos14_h07/pipeline/*.py` 置于 `sr-baselines/` 根目录。三个上游仓库使用下表提交，运行 `compatibility.py` 应用实验使用的适配。

服务器工作目录为 `/root/autodl-tmp/sr-baselines`。`RPMIN-e79d2b3` 来自 [RP-MIN GitHub main](https://github.com/SimonHanYANG/RPMIN/commit/e79d2b3ce3ba85f8b8b0e47b6b9c867567bafc1b)。短预算横向对比的 RP-MIN 行使用宽度 32 的 `RP-MIN-fresh/runs/x4-polish/pixel-refine/best.pt`，逐块单次前向，实测 **27.4757 dB / 0.6861**。该权重 SHA-256 为 `fb48c80af1dfbff230929a851e8835733e1e78741dd6bbb06913f0dba9c7cb87`。

所有方法使用同一组已配准参考图和保留测试集。训练、验证、测试分别包含 34、4、4 个视野。推理按固定网格独立处理 308 个 128×128 LR 图块，步长为横向 32、纵向 25，起点为 1。`rpmin.objective.image_metrics` 计算 RGB PSNR 和 SSIM。HR 裁边为 0，推理单次前向，RP-MIN 的推理种子为 101。`comparison.csv` 和各子目录的 `per_image.csv` 记录汇总和逐块数据。

| 方法 | 源码提交 | 权重和训练口径 |
| --- | --- | --- |
| [AESOP RRDB128](https://github.com/2minkyulee/AESOP-SR) | `3d6fe1d` | 官方 `AESOP_RRDB128_290K.pth` 直接测试，另用相同初始权重做 204 次显微域 L1 微调 |
| [RRDB P+SeD](https://github.com/lbc12345/SeD) | `444a8f2` | 官方 `RRDB+P+SeD.pt` 直接测试，另用相同初始权重做 204 次显微域 L1 微调 |
| [NeXtSRGAN](https://github.com/PomKlementieff/NeXtSRGAN) | `b1f5095` | 显微配对数据上从零开始做 800 次 L1 更新和 200 次 GAN 更新，按验证 PSNR 在第 100 和 200 个 GAN 更新保存的权重中选择 |
| NeXtSRGAN（RRDB 初始化） | `b1f5095` | 从通用 ESRGAN PSNR 权重初始化对应卷积层，做 204 次 L1 和 100 次 GAN 更新，检查点在第 50 和 100 个 GAN 更新之间按验证 PSNR 选择。[初始化权重](https://github.com/cszn/KAIR/releases/download/v1.0/RRDB.pth) |
| NeXtSRGAN（权重插值） | `b1f5095` | 在 RRDB 初始化实验的 L1 与 GAN 两套权重之间做线性插值。沿用上游 `net_interp.py` 的系数网格，按验证 PSNR 在 GAN 比例大于 0 的候选中选取。 |

两组迁移微调均使用 LR 裁块 32、批量 8、Adam 学习率 `2e-5`、种子 42。此阶段只更新生成器。NeXtSRGAN 使用 Keras 3 的 PyTorch CUDA 后端运行官方 RRDB 生成器、判别器、VGG 感知损失及相对 GAN 损失。服务器上的 TensorFlow 2.20 CUDA 试跑遇到 RTX 5090 的 `CUDA_ERROR_INVALID_PTX`。兼容改动可在各源码仓库的 `git diff` 查看。结果属于不同初始化和训练预算下的对照。

## ×2 适配与双倍率结果

三个对照网络的 ×2 版本保留 ×4 生成器的卷积层，关闭第二次最近邻放大，用 MFA-Cell ×2 配对图做适配。AESOP 与 SeD 仍从各自官方权重开始，做 204 步 L1 微调。NeXtSRGAN 沿用 ESRGAN PSNR 权重初始化、204 步 L1、100 步 GAN，并在 ×2 验证集选择权重插值系数。测试对每个倍率使用 4 个保留视野的 308 个配准图块。`runs/comparison_x2_x4.csv` 汇总四个网络，RP-MIN 行采用项目已核对的 D4 配置。`runs/x2-pilot/rpmin/summary.json` 和 `runs/x4-pilot/rpmin/summary.json` 也记录了 RP-MIN 的逐块单次推理成绩，和三个对照网络的推理设置一致。

```bash
cd /root/autodl-tmp/sr-baselines
CUDA_VISIBLE_DEVICES=0 RUN=runs/x2-repeat bash run_mfa_x2.sh torch
CUDA_VISIBLE_DEVICES=0 RUN=runs/x2-repeat bash run_mfa_x2.sh next
```

## RP-MIN 项目记录与论文表 2

| 倍率 | RP-MIN 测试配置 | PSNR（dB） | SSIM |
| --- | --- | ---: | ---: |
| ×2 | 宽度 32 基础模型，逐块输入，D4，3 个种子取均值 | 31.4304 | 0.8764 |
| ×4 | 宽度 32 和 48 等权集成，逐块输入，D4，3 个种子取均值 | 27.6562 | 0.6870 |
| ×4 | 宽度 32 和 48 等权集成，整视野输入后按网格评分，D4，3 个种子取均值 | **27.6797** | **0.6871** |
| ×4 | 宽度 32 `pixel-refine`，整视野输入后按网格评分，D4，3 个种子取均值 | 27.6330 | 0.6881 |

×4 的 0.6871 是 27.6797 dB 对应的 SSIM。若单独按 SSIM 排序，宽度 32 `pixel-refine` 的整图 D4 记录已有 0.6881。论文表 2 的 MFA-Cell 行是 ×2 为 31.24 dB / 0.86，×4 为 27.68 dB / 0.67。横向对照的 `comparison.csv` 使用逐块单次前向口径。上表的项目记录来自 `RP-MIN-fresh/runs/full/test/x2/aligned/selected/summary.json`、`runs/x4-polish/width48-frozen/comparison.json` 和 `runs/x4-polish/frozen/comparison.json`。

## MITOS-14 H10 训练对比

项目数据有 5 个同属 H10 的 10×／20×／40× 视野。用种子 42 固定整视野划分，两个倍率均以 `H10_01`、`H10_02`、`H10_03` 训练，`H10_04` 验证，`H10_00` 测试。实验评估同源未见视野，病例级样本量为 1。配准逐视野独立执行，配准报告记录接受或回退的变换。每个倍率使用 132 个互不重叠的 128×128 LR 图块，逐块单次前向，统一计算 RGB PSNR 和 SSIM。

- RP-MIN 随机初始化，在 MITOS-14 训练 68 轮，共 816 步，按验证 PSNR 选择检查点。
- AESOP 和 SeD 分别从作者发布的通用权重初始化，在 MITOS-14 各做 204 步 L1 生成器微调。
- NeXtSRGAN 从通用 ESRGAN PSNR 权重初始化，在 MITOS-14 训练 204 步 L1 和 100 步 GAN。检查点和权重插值系数在验证视野上选取。

四个网络的训练均使用 H10 训练视野。×2 对照的上采样适配沿用 MFA-Cell 实验。输出在 `runs/mitos14-h10-train/`，包含 `comparison.csv`、每个倍率的 `split.csv`、配准报告、检查点和逐块评分。[MITOS-ATYPIA-14 原始数据说明](https://mitos-atypia-14.grand-challenge.org/Dataset/)

```bash
cd /root/autodl-tmp/sr-baselines
CUDA_VISIBLE_DEVICES=0 RUN_ROOT=runs/mitos14-h10-repeat bash run_mitos_train.sh 2
CUDA_VISIBLE_DEVICES=0 RUN_ROOT=runs/mitos14-h10-repeat bash run_mitos_train.sh 4
torch-env/bin/python summarize_mitos_train.py runs/mitos14-h10-repeat
```

### MITOS-14 的 RP-MIN 优化

沿用 `H10_01–03` 的训练图对、`H10_04` 验证和 `H10_00` 的 132 个测试图块。宽度 32 的 RP-MIN 从本次 MITOS 基础权重继续训练 96 轮，共 3,456 步，并启用相位细节分支。宽度 48 的细节版从随机权重训练 120 轮，共 2,880 步。训练配置和权重分别保存在 `runs/mitos14-rpmin-opt/x{2,4}/detail` 与 `width48`。

验证比较单模型、双模型等权集成、反射填充和 D4。在验证 SSIM 至少达到基础单模型的候选中选 PSNR 最高的配置。也比较了用训练图块拟合的相位残差校正。`locked/selection.json` 保存全部验证分数、最终配置及权重指纹。`locked/test/summary.json` 和 `per_image.csv` 保存锁定后的测试结果。更新表复用各对照网络已经完成的评分。

两个倍率均选中宽度 32 和 48 的等权预测集成及相位残差校正，逐块输入且无需反射填充或 D4。单次前向对照均由验证集选中宽度 48，测试 ×2 为 26.3236 dB / 0.8534，×4 为 25.3450 dB / 0.7436，记录在 `locked/single-forward/summary.json`。

```bash
cd /root/autodl-tmp/sr-baselines
CUDA_VISIBLE_DEVICES=0 bash run_mitos_rpmin_select.sh 2
CUDA_VISIBLE_DEVICES=0 bash run_mitos_rpmin_select.sh 4
torch-env/bin/python summarize_mitos_train.py runs/mitos14-h10-train --rpmin-opt runs/mitos14-rpmin-opt
```

后续在同一划分测试了两种神经残差细化，共 3,200 步和 2,400 步。验证 SSIM 有小幅提升，PSNR 最高仍为上一轮配置。最终验证选中跨倍率融合，在 ×2 预测中混入 ×4 降采样预测的零均值差分，系数 0.25，×4 则用 ×2 的差分修正低频，系数 0.5。测试得到 ×2 **26.4588 dB / 0.8520**，×4 **25.5007 dB / 0.7404**。四个网络的更新表位于 `runs/mitos14-rpmin-cross-scale/comparison.csv`。

第三轮用三个训练视野拟合 5×5 多通道成像核。以宽度 48 检查点为初值，做 1,600 步合成与真实样本交替训练，再用真实配对图微调 2,800 步。两个倍率均由 `H10_04` 验证选中混入 25% 新模型输出。保留图块的 RGB 结果为 ×2 **26.4715 dB / 0.8535**、×4 **25.5129 dB / 0.7432**。RGB 对照表位于 `runs/mitos14-rpmin-camera/comparison.csv`。

另用 BT.601 Y 单独评价同一组 132 个图块。以 255 为 PSNR 峰值，Y 的数字范围为 16–235，裁边为 0。采用亮度损失微调 1,600 步，验证选中 ×2 的跨倍率融合模型，×4 混入 25% 亮度微调模型。Y 测试为 ×2 **28.1939 dB / 0.8856**、×4 **27.2234 dB / 0.7860**。稿件没有注明 RGB 或 Y；Y 结果单列于 `runs/mitos14-rpmin-luma/comparison.csv`。RGB 表仍统一使用 RGB。

后续亮度优化在相同的 128×128 LR 评分区域外读取 16 像素的真实上下文，做 D4 集成后裁回原来的评分区域。×4 验证选中 0.5 强度的相位亮度残差，系数由三个训练视野拟合。保留原扫描方向的 2,400 步亮度微调，以及按局部梯度方向分组的残差校正，均未获得更优的验证 PSNR。锁定后的 `H10_00` 亮度结果为 ×2 **28.2438 dB / 0.8859**、×4 **27.2774 dB / 0.7863**。评分区域和 Y 的峰值、裁边均沿用上一轮。结果与配置在 `runs/mitos14-rpmin-context/`。

接着在训练阶段给原生 LR 图块加入 16 像素真实邻域，以中心 HR 区域计算亮度损失，微调 2,000 步。验证集选中 ×2 混入 50%、×4 混入 25% 的新模型输出，再使用训练视野拟合的相机核做低频数据一致性校正，两个倍率的更新强度均为 0.6。仍在原 `H10_00` 评分图块上计算 BT.601 Y，×2 **28.2705 dB / 0.8869**，×4 **27.2787 dB / 0.7865**。分相位拟合的相机核对训练图对的误差改善有限，独立保存在 `phase-camera.json`。本轮配置、测试记录和汇总在 `runs/mitos14-rpmin-consistency/`。

继续验证了 MFA-Cell 权重初始化与新增官方 MITOS-14 训练视野。MFA-Cell 初始化后在 H10 微调 3,600 步，验证集未选中该候选。官方 Hamamatsu 档案提供 ×10、×20、×40 原始帧，已核对官方 `H10_01` ×10 帧与项目图像像素一致。新增 `H07_00` 和 `H07_01` 两个独立训练视野，按图像相关性验证子视野拼接，配准记录保存在 `runs/mitos14-rpmin-h07/`。[官方 MITOS-ATYPIA-14 下载页](https://mitos-atypia-14.grand-challenge.org/Donwload/)

联合 H10 和 H07 微调 3,600 步后，仅用原 H10 训练视野再适配 1,000 步。×2 和 ×4 都在验证集选中再适配模型的高频分量，融合权重分别为 0.25 和 0.5。按原有 BT.601 Y 口径在 `H10_00` 的同一批 132 个图块上复测，×2 **28.2778 dB / 0.8874**，×4 **27.2878 dB / 0.7883**。验证决策与汇总在 `runs/mitos14-rpmin-h07/`，锁定后的测试记录在 `runs/mitos14-rpmin-h07-readapt/`。

## 服务器复跑

```bash
cd /root/autodl-tmp/sr-baselines
refs=/root/autodl-tmp/rpmin-project/RP-MIN-fresh/runs/full/references/x4
out=runs/x4-repeat
export OMP_NUM_THREADS=2
PYTHONPATH=RPMIN-e79d2b3 torch-env/bin/python -m rpmin --device cuda evaluate \
  --manifest "$refs/test/aligned.csv" --split test \
  --checkpoint /root/autodl-tmp/rpmin-project/RP-MIN-fresh/runs/x4-polish/pixel-refine/best.pt \
  --lr-patch 128 --stride-x 32 --stride-y 25 --origin 1 --batch 4 --seeds 101 --out "$out/rpmin"
PYTHONPATH=RPMIN-e79d2b3 torch-env/bin/python -m rpmin --device cuda evaluate \
  --manifest "$refs/test/aligned.csv" --split test --bicubic --scale 4 \
  --lr-patch 128 --stride-x 32 --stride-y 25 --origin 1 --batch 4 --seeds 101 --out "$out/bicubic"
torch-env/bin/python compare_x4.py AESOP --manifest "$refs/test/aligned.csv" --split test --out "$out/aesop"
torch-env/bin/python compare_x4.py SeD --manifest "$refs/test/aligned.csv" --split test --out "$out/sed"
for model in AESOP SeD; do
  torch-env/bin/python compare_x4.py "$model" --manifest "$refs/train-val/aligned.csv" \
    --split val --out "$out/$model-transfer" --fit-transfer
  torch-env/bin/python compare_x4.py "$model" --manifest "$refs/test/aligned.csv" \
    --split test --out "$out/$model-transfer" --adapted "$out/$model-transfer/adapted.pt"
done
torch-env/bin/python compare_x4.py prepare --manifest "$refs/train-val/aligned.csv" --split val --out "$out/inputs"
torch-env/bin/python compare_x4.py prepare --manifest "$refs/test/aligned.csv" --split test --out "$out/inputs"
RUN="$out" bash run_next_x4.sh train
RUN="$out" bash run_next_x4.sh evaluate
transfer="$out-next-transfer"
mkdir -p "$transfer"
ln -s "$(realpath "$out/inputs")" "$transfer/inputs"
RUN="$transfer" PRETRAIN=204 GAN=100 INIT_RRDB=weights/rrdb-official-psnr.pth bash run_next_x4.sh train
RUN="$transfer" bash run_next_x4.sh evaluate
RUN="$transfer" bash next_interpolate_x4.sh
torch-env/bin/python summarize_x4.py "$out" --next-transfer "$transfer"
```
