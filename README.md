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
