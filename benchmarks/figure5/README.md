# Figure 5 · ×4 定性对比

六行沿用论文图中的区域，新增 NeXtSRGAN、AESOP 和 SeD 三列。[manifest.json](manifest.json) 记录每行视野、划分、LR 裁剪、HR 区域映射和检查点 SHA-256。第 1、2 行属于验证集，第 3、5、6 行属于测试集，第 4 行属于训练集。

在原对照实验工作区生成六行推理素材。

```bash
CUDA_VISIBLE_DEVICES=0 torch-env/bin/python export_figure5.py figure5_rois.json runs/figure5-x4-replay
```

复排时，将素材目录命名为 `server_results/`，将论文 Figure 5 的内嵌图保存为 `Figure5_original.png`，与 `make_figure5.py` 放在同一目录。使用 Pillow 运行 `python make_figure5.py`，生成 300 dpi 的 `Figure5_extended.png` 与拆分面板。
