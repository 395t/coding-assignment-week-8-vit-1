# Instructions to Train DeiT

This code was adapted from the published [DeiT repository](https://github.com/facebookresearch/deit). Most changes involve pulling in other datasets to fine-tune DeiT on.
Main dependencies include PyTorch 1.8 and timm==0.3.2.
You can train and evaluate DeiT models using the main.py script.

## Training a Model on CIFAR-10

Training DeiT-tiny on CIFAR-10
```
python main.py --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --data-set CIFAR-10 --model deit_tiny_patch16_224 --batch-size 128 --epochs 20 --output_dir tiny_CIFAR10
```

Training DeiT-tiny-distilled on Caltech101
```
python main.py --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth --data-set CALTECH101 --model deit_tiny_distilled_patch16_224 --batch-size 128 --epochs 20 --output_dir tiny_CALTECH101_distilled
```