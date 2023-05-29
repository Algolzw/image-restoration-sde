#!/bin/bash

#############################################################
### training ###

# for single GPU
python train.py -opt=options/ssr/train/refusion.yml

# for multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6524 train.py -opt=options/ssr/train/refusion.yml --launcher pytorch


#############################################################

### testing with GT ###
# python test.py -opt=options/ssr/test/refusion.yml

### inference only ###
# python inference.py -opt=options/ssr/test/refusion.yml

#############################################################
