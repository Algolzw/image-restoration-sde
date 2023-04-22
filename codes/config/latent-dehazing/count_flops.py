import argparse

import torch
from torchsummaryX import summary

import options as option
from models import create_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/dehazing/test/nasde.yml",
    help="Path to option YMAL file of Predictor.",
)
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)
model = create_model(opt)

test_tensor = torch.randn(1, 8, 750, 500).cuda()
summary(model.model, x=test_tensor, cond=test_tensor, time=1)

test_tensor = torch.randn(1, 3, 6000, 4000).cuda()
summary(model.latent_model, x=test_tensor)
