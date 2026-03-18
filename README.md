# HCLBind Implementation

## 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate hclbind-env

```

## 2. Usage
Download model weights (https://drive.google.com/drive/folders/1tKDuAs9hobOurKvSOOGlTucNMtDhu9uh?usp=sharing), extract them to `outputs/final_ckpt` and `outputs/pretrain_ckpt`.

```bash
export ROOTDIR=/PATH-TO/HCLBind/
cd $ROOTDIR
```

### Pretrain

```bash
python pretrain.py --config config/pretrain.yaml
```

### Finetune
```bash
python finetune.py --config config/finetune.yaml --ckpt outputs/final_ckpt
```
