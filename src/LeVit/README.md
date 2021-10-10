The model is based on [paper](https://arxiv.org/abs/2104.01136)

To run LeVit, pytorch 1.7+ is required. 

The implemetation is heavily based on [Pytorch image models](https://github.com/rwightman/pytorch-image-models) with changes to train.py and levit.py. 

The major changes is to make LeVit model flexible for training on any dataset.

Steps to run the model are : 

`git clone https://github.com/rwightman/pytorch-image-models.git`

Replace levit.py and train.py from this repository to pytorch-image-models repository.

To train a levit-256 on Cifar-10, locally distributed, 1 GPU, one process per GPU w/ cosine schedule, random-erasing prob of 10%, no droput and per-pixel random value:

`./distributed_train.sh 1 /data/cifar10/ --model levit_256 --sched cosine --epochs 10 --warmup-epochs 10 --lr 0.01 --reprob 0.1 --dropout 0.0 --remode pixel --batch-size 256 --amp -j 1`

model values for levit are [levit_128, levit_128s, levit_192, levit_256, levit_384]

For adding more hyperparameters, please refer train.py .


