#!/bin/bash

#############################################################
### training ###

# for single GPU
python train.py -opt=options/derain/train/train_sde_derain.yml

# for multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6512 train.py -opt=options/derain/train/train_sde_derain.yml --launcher pytorch


#############################################################

### testing ###
# python test.py -opt=options/derain/test/test_sde_derain.yml

#############################################################