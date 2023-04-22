import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device
lpips_fn = lpips.LPIPS(net='alex').to(device)

sde = util.DenoisingSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], device=device)
sde.set_model(model.model)

degrad_sigma = opt["degradation"]["sigma"]

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"] + f'_sigma{degrad_sigma}'  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_times = []

    for ii, test_data in enumerate(test_loader):

        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        GT = test_data["GT"]
        LQ = util.add_noise(GT, degrad_sigma)

        model.feed_data(LQ, GT)
        tic = time.time()
        model.test(sde, sigma=degrad_sigma, save_states=True)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        output = util.tensor2img(visuals["Output"].squeeze())  # uint8
        LQ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT = util.tensor2img(visuals["GT"].squeeze())  # uint8
        
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".tif")
        util.save_img(output, save_img_path)

        LQ_img_path = os.path.join(dataset_dir, img_name + "_noisy.png")
        GT_img_path = os.path.join(dataset_dir, img_name + "_clean.png")
        util.save_img(LQ, LQ_img_path)
        util.save_img(GT, GT_img_path)

        if need_GT:
            psnr = util.calculate_psnr(output, GT)
            ssim = util.calculate_ssim(output, GT)
            lp_score = lpips_fn(
                visuals["GT"].to(device) * 2 - 1, visuals["Output"].to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)

            logger.info(
                "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                    img_name, psnr, ssim
                )
            )

        else:
            logger.info(img_name)

        # break


    ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )

    logger.info(
            "----average LPIPS\t: {:.6f}\n".format(ave_lpips)
        )

    print(f"average test time: {np.mean(test_times):.4f}")
