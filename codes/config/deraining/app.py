import gradio as gr
import cv2
import argparse
import sys
import numpy as np
import torch

import options as option
from models import create_model
sys.path.insert(0, "../../")
import utils as util

# options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='options/test/ir-sde.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

def deraining(image):
    image = image[:, :, [2, 1, 0]] / 255.
    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    noisy_tensor = sde.noise_state(LQ_tensor)
    model.feed_data(noisy_tensor, LQ_tensor)
    model.test(sde)
    visuals = model.get_current_visuals(need_GT=False)
    output = util.tensor2img(visuals["Output"].squeeze())
    return output

interface = gr.Interface(fn=deraining, inputs="image", outputs="image", title="Image Deraining using IR-SDE")
interface.launch()