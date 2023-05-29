#!/bin/bash

#############################################################
### training ###

# for single GPU
python train.py -opt=options/bokeh/train/refusion.yml

# for multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6212 train.py -opt=options/bokeh/train/refusion.yml --launcher pytorch


#############################################################

### testing ###
python test.py -opt=options/bokeh/test/refusion.yml

#############################################################
