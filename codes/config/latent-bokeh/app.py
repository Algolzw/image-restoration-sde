import gradio as gr
import cv2
import argparse
import sys
import numpy as np
import torch
from pathlib import Path

import options as option
from models import create_model
sys.path.insert(0, "../../")
import utils as util

# options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='options/bokeh/test/refusion.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

def deraining(image):
    image = image[:, :, [2, 1, 0]] / 255.

    src_lens = torch.tensor(float(18))
    tgt_lens = torch.tensor(float(160))
    disparity = torch.tensor(float(35))

    image = torch.tensor(image).float().cuda()
    image = torch.permute(image, (2, 0, 1))

    latent_LQ, hidden = model.encode(torch.unsqueeze(image, 0))
    noisy_state = sde.noise_state(latent_LQ)

    model.feed_data(noisy_state, latent_LQ, src_lens=src_lens, tgt_lens=tgt_lens, disparity=disparity, GT=None)
    model.test(sde, hidden=hidden, save_states=False)
    visuals = model.get_current_visuals(need_GT=False)
    output = util.tensor2img(visuals["Output"].squeeze())
    return output

interface = gr.Interface(fn=deraining, inputs="image", outputs="image", title="Image Deraining using IR-SDE")
interface.launch(share=True)

