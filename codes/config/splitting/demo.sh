#!/bin/bash

#############################################################
### training ###

# for single GPU
# python train.py -opt=options/train/ir-sde.yml
# python train.py -opt=options/train/refusion.yml

# for multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6512 train.py -opt=options/train/ir-sde.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6512 train.py -opt=options/train/refusion.yml --launcher pytorch

#############################################################

### testing ###
python test.py -opt=options/test/ir-sde.yml
# python test.py -opt=options/test/refusion.yml

#############################################################