# Vision Transformers

## Summary


### DeiT
The Data-efficient image Transformers (DeiT) paper introduces improvements upon the original Vision Transformer (ViT) architecture by leveraging knowledge distillation to reach high performance using a smaller dataset. In our experiments we compare performance between the DeiT models and their distilled counterparts to observe the importance of the knowledge distillation introduced in the paper. For all experiments we use the DeiT-tiny architecture (5-6M params) and train for 20 epochs on an NVIDIA 2060ti. Below we show results for training these models across the CIFAR-10, STL-10, and Caltech101 datasets.

#### Results
![DeiT Table](imgs/DeiTTable.png) 
DeiT Performance | '
:-|-:
![DeiT Accuracy](imgs/DeiTLoss.png)  | ![DeiT Loss](imgs/DeiTLoss.png) 

![DeiT Model Accuracies](imgs/DeiTModelAccuraciesBar.png) 

When performing the fine-tuning experiments in the paper, the authors fine-tuned for 300 epochs compared to the 20 epochs performed here. Despite training for such a short number of epochs, the DeiT models were able to reach high accuracies across all tasks. The distilled models consistently perform better than the base models, showing accuracies around .3-.5% better across the board. Perhaps this gap would be wider had we trained using one of the larger DeiT architectures or fine-tuned over more epochs.

## References

[DeiT Repository (FB Research)](https://github.com/facebookresearch/deit)

# LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference

## Summary


### LeViT
LeVit paper utilises recent finding in attention-based architectures which are competitive on highly parallel processing hardware. LeVit outperforms existing convnets and vision transformers on the basis of speed/accuracy tradeoff. We finetuned LeVit for 20 epochs by changing various parameters like learning rate, dropout, warm restarts, batch size, etc for 20 epochs. We also trained LeVit for 70 epochs with the same parameters used for ImageNet without fine tuning. 
Below we show results for training these models across the CIFAR-10, STL-10, and Caltech101 datasets. 

### LeViT Implementation
Refer the readme for Levit implementation. 

#### Results

LeVit training loss for 20 epochs for Cifar-10 | LeVit training loss for 70 epochs for Cifar-10 '
:-|-:
![LeVit training loss for 20 epochs](imgs/cifar10-Levit-trainingLoss.png)  | ![LeVit training loss for 70 epochs](imgs/cifar10-Levit-train-loss-70epoch.png)

LeVit test accuracy for 20 epochs for Cifar-10 | LeVit test accuracy for 70 epochs for Cifar-10 '
:-|-:
![LeVit accuracy for 20 epochs](imgs/cifar10-Levit-test-acc.png)  | ![LeVit accuracy for 70 epochs](imgs/cifar10-Levit-test-acc-70epoch.png)

LeVit training loss for 20 epochs for STL10 | LeVit training loss for 70 epochs for STL10   '
:-|-:
![LeVit training loss for 20 epochs](imgs/stl10-Levit-trainingLoss.png)  | ![LeVit training loss for 70 epochs](imgs/stl10-Levit-train-loss-70epoch.png) 

LeVit test accuracy for 20 epochs for STL10 | LeVit test accuracy for 70 epochs for STL10   '
:-|-:
![LeVit test accuracy for 20 epochs](imgs/stl10-Levit-test-acc.png)  | ![LeVit test accuracy for 70 epochs](imgs/stl10-Levit-test-acc-70epoch.png)

LeVit training loss for 20 epochs for Caltech101 | LeVit training loss for 70 epochs for Caltech101 '
:-|-:
![LeVit training loss for 20 epochs](imgs/caltech-Levit-trainingLoss.png)  | ![LeVit training loss for 70 epochs](imgs/caltech101-Levit-train-loss-70epoch.png)

LeVit test accuracy for 20 epochs for Caltech101 | LeVit test accuracy for 70 epochs for Caltech101 '
:-|-:
![LeVit test accuracy for 20 epochs](imgs/caltech-Levit-test-acc.png)  | ![LeVit test accuracy for 70 epochs](imgs/caltech101-Levit-test-acc-70epoch.png) 

LeVit training loss for 20 epochs comparison between datasets | LeVit training loss for 70 epochs comparison between datasets '
:-|-:
![LeVit training loss for 20 epochs](imgs/trainingLoss-bw-datasets-Levit256.png)  | ![LeVit training loss for 70 epochs](imgs/trainingLoss-bw-datasets-Levit384.png) 

LeVit test accuracy for 20 epochs comparison between datasets | LeVit test accuracy for 70 epochs comparison between datasets   '
:-|-:
![LeVit test accuracy for 20 epochs](imgs/testAcc-bw-datasets-Levit256.png)  | ![LeVit test accuracy for 70 epochs](imgs/testAcc-bw-datasets-Levit384.png)

## Observations

1) With increasing the epochs without much finetuning, the performance of the all the LeVit models on all the datasets have improved significantly. 
2) On all the datasets, the training loss of LeVit-384 is minimum.
3) On all the datasets, the test accuracy of LeVit-256 is highest within 20 epochs but test accuracy of LeVit-384 is highest after the initial 30-40 epochs.
4) Cifar-10 reaches an accuracy of 70 percent after fine-tuning for 20 epochs whereas Cifar-10 reaches an accuracy of 82 percent for 70 epochs with default parameters.
5) STL10 reaches an accuracy of 51 percent after fine-tuning for 20 epochs whereas Cifar-10 reaches an accuracy of 68 percent for 79 epochs with default parameters. At around 40th epoch, there is a jump in the accuracy for SL10.
6) Caltech-101 reaches an accuracy of 42.7 percent after fine-tuning for 20 epochs whereas Cifar-10 reaches an accuracy of 60.5 percent for 70 epochs with default parameters. At around the 25th epoch, there is a significant jump in the accuracy for Caltech-101.

## References

[Pytorch image models](https://github.com/rwightman/pytorch-image-models)
