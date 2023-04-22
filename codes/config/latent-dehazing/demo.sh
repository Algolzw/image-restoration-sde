#!/bin/bash

#############################################################
### training ###

# for single GPU
python train.py -opt=options/dehazing/train/nasde.yml

# for multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6112 train.py -opt=options/dehazing/train/nasde.yml --launcher pytorch


#############################################################

### testing ###
# python test.py -opt=options/dehazing/test/nasde.yml

#############################################################
