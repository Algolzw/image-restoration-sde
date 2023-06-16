#!/bin/bash

#############################################################
### training ###

# for single GPU
python train.py -opt=options/train/train_haze.yml
# python train.py -opt=options/train/train_bokeh.yml

# for multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6512 train.py -opt=options/train/train_haze.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6622 train.py -opt=options/train/train_bokeh.yml --launcher pytorch


#############################################################