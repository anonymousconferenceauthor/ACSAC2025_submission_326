#!/bin/bash

# EPOCHS=100
# BACKBONE=resnet10
# DATASET=TinyImageNet

# python3 pre_train.py --dataset $DATASET --backbone $BACKBONE --task "random" --num_epochs $EPOCHS
# for K in 16 32 64 128; do
#   python3 main.py --method "disentangle_mu" --task "random" --unlearn_k $K --dataset $DATASET --batch_size 128 --backbone $BACKBONE
# done


EPOCHS=100
BACKBONE=resnet10
DATASET=CIFAR20

python3 pre_train.py --dataset $DATASET --backbone $BACKBONE --task "random" --num_epochs $EPOCHS
for S in 2 17 71; do
  python3 main.py --method "disentangle_mu" --task "sub_class" --s_cls $S --dataset $DATASET --batch_size 128 --backbone $BACKBONE
done