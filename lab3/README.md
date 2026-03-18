# Lab 3 Starter: U-Net + ViT with Dual-Domain Loss

This folder contains a baseline setup for **Lab 3**:
- model comparison between `U-Net` and `ViT`
- integration of a **Dual-Domain Loss** (spatial + frequency)
- shared training/evaluation loop for fair comparison

## Folder structure expected

Your dataset root should look like this:

```
data/
  train/
    compressed/*.jp2
    ground_truth/*.(png|jpg|jpeg|bmp|tif|tiff)
  validate/
    compressed/*.jp2
    ground_truth/*.(png|jpg|jpeg|bmp|tif|tiff)
  test/
    compressed/*.jp2
    ground_truth/*
```

Matching is done by filename stem. Example:
- `compressed/123.jp2`
- `ground_truth/123.png`

## Install dependencies

```bash
pip install -r lab3/requirements.txt
```

## Quick training runs

From project root:

```bash
python lab3/train.py --data_root ../data --model unet --epochs 10 --batch_size 8 --amp
python lab3/train.py --data_root ../data --model vit --epochs 10 --batch_size 8 --amp
```

If your shell is already inside `CMPUT414W26-sun`, `--data_root ../data` points to the sibling `data` folder in your workspace.

## Core Lab 3 knobs

Dual-domain objective:

`L_total = alpha * L_spatial + beta * L_frequency`

Tune with:
- `--alpha`
- `--beta`
- `--phase_weight`

Example:

```bash
python lab3/train.py --data_root ../data --model unet --alpha 1.0 --beta 0.15 --phase_weight 0.7 --amp
```

## Output artifacts

Per model, training saves:
- CSV log: `outputs/{model}_log.csv`
- best checkpoint by validation PSNR: `outputs/{model}_best.pt`

## Suggested Lab 3 plan

1. Train U-Net and ViT with only spatial supervision (`--beta 0.0`) for baseline.
2. Re-train both with dual-domain supervision (`--beta 0.05`, `0.1`, `0.2`).
3. Compare PSNR + SSIM curves and qualitative edge/ringing restoration.
4. Pick one architecture to carry into Lab 4 hyperparameter optimization.
