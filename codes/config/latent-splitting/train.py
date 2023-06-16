import argparse
import logging
import math
import os
import random
import sys
import copy
import socket

from tqdm import tqdm
import wandb
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

import options as option
from models import create_model
from core.psnr import RangeInvariantPsnr
from utils.stitch_prediction import stitch_predictions

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr
from utils.train_utils import get_trimmed_pixel_count, save_imgs

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            # util.mkdir_and_rename(
            #     opt["path"]["experiments_root"]
            # )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            # os.system("rm ./log")
            # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        # util.setup_logger(
        #     "base",
        #     opt["path"]["log"],
        #     "train_" + opt["name"],
        #     level=logging.INFO,
        #     screen=False,
        #     tofile=True,
        # )
        # util.setup_logger(
        #     "val",
        #     opt["path"]["log"],
        #     "val_" + opt["name"],
        #     level=logging.INFO,
        #     screen=False,
        #     tofile=True,
        # )
        # logger = logging.getLogger("base")
        # logger.info(option.dict2str(opt))
        prefix= os.path.dirname(os.path.dirname(opt["path"]["experiments_root"]))
        expname = opt["path"]["experiments_root"][len(prefix)+1:]
        hostname = socket.gethostname()
        wandb.init(name=os.path.join(hostname,'latent', expname),
                         dir=opt["path"]["log"],
                         project="DiffusionSplitting",
                         config=opt)
        # tensorboard logger
        # if opt["use_tb_logger"] and "debug" not in opt["name"]:
        #     version = float(torch.__version__[0:3])
        #     if version >= 1.1:  # PyTorch 1.1
        #         from torch.utils.tensorboard import SummaryWriter
        #     else:
        #         logger.info(
        #             "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
        #                 version
        #             )
        #         )
        #         from tensorboardX import SummaryWriter
        #     tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        raise NotImplementedError('handle this case')
        # util.setup_logger(
        #     "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        # )
        # logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 1000  # enlarge the size of each epoch
    for phase in ['train', 'val']:
        dataset_opt = opt["datasets"][phase]
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                print(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                print(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            max_val = train_set.compute_max_val()
            dataset_opt['max_val'] = max_val
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                print(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None
    mean_, std_ = train_set.compute_mean_std()
    train_set.set_mean_std(mean_, std_)
    val_set.set_mean_std(mean_, std_)

    #### create model
    model = create_model(opt) 
    device = model.device

    #### resume training
    if resume_state:
        print(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    # scale = opt['degradation']['scale']

    #### training
    print(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            # print(current_step)
            LQ, GT = train_data["LQ"], train_data["GT"]
            latent_LQ, _ = model.encode(LQ.to(device))
            latent_GT, _ = model.encode(GT.to(device))
            timesteps, states = sde.generate_random_states(x0=latent_GT, mu=latent_LQ)

            model.feed_data(states, latent_LQ, latent_GT) # xt, mu, x0
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            wandb.log({k: v,'step':current_step})
                if rank <= 0:
                    print(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                prediction, target = get_dset_prediction(model, sde, val_set, val_loader)
                assert prediction.shape[-1] == target.shape[-1]
                assert prediction.shape[-1] == 1

                prediction = prediction[...,0]
                target = target[...,0]
                avg_psnr = RangeInvariantPsnr(target, prediction)
                avg_psnr = torch.mean(avg_psnr).item()
                
                save_imgs(val_set.unnormalize_img(prediction[:,None])[:,0], 
                          val_set.unnormalize_img(target[:,None])[:,0],
                          current_step, 
                          opt["path"]["val_images"])

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step
                    print("Saving models and training states.", current_step)
                    model.save('best')
                    model.save_training_state(epoch, current_step, filename='best')

                # log
                wandb.log({"val_psnr":avg_psnr})
                wandb.log({'step':current_step})

                # log
                # logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                # logger_val = logging.getLogger("val")  # validation logger
                # logger_val.info(
                #     "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                #         epoch, current_step, avg_psnr
                #     )
                # )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                # if opt["use_tb_logger"] and "debug" not in opt["name"]:
                #     tb_logger.add_scalar("psnr", avg_psnr, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            # if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
            #     if rank <= 0:
            #         logger.info("Saving models and training states.")
            #         model.save(current_step)
            #         model.save_training_state(epoch, current_step)

    if rank <= 0:
        print("Saving the final model.")
        model.save("latest")
        print("End of Predictor and Corrector training.")

def get_dset_prediction(model, sde, dset, dloader):
    # avg_psnr = 0.0
    # idx = 0
    # for _, val_data in enumerate(val_loader):
    #     LQ, GT = val_data["LQ"], val_data["GT"]
    #     with torch.no_grad():
    #         latent_LQ, hidden = model.encode(LQ.to(device))
    #         noisy_state = sde.noise_state(latent_LQ)

    #     # valid Predictor
    #     model.feed_data(noisy_state, latent_LQ, GT)
    #     model.test(sde, hidden)
    #     visuals = model.get_current_visuals()

    #     output = util.tensor2img(visuals["Output"].squeeze())  # uint8
    #     gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

    #     # calculate PSNR
    #     avg_psnr += util.calculate_psnr(output, gt_img)
    #     idx += 1

    #     # break
    #     torch.cuda.empty_cache()

    # avg_psnr = avg_psnr / idx

    idx = 0
    predictions = []
    target = []
    patchwise_psnr = []
    for val_data in tqdm(dloader):

        LQ, GT = val_data["LQ"], val_data["GT"]
        with torch.no_grad():
            latent_LQ, hidden = model.encode(LQ.cuda())
            noisy_state = sde.noise_state(latent_LQ)


        # valid Predictor
        model.feed_data(noisy_state, latent_LQ, GT)
        model.test(sde, hidden)
        visuals = model.get_current_visuals()

        output = visuals["Output"]#.squeeze().cpu().detach().numpy()
        gt_img = visuals["GT"]#.squeeze().cpu().detach().numpy()
        patchwise_psnr.append(RangeInvariantPsnr(gt_img, output).mean().item())

        # store for tiled prediction
        predictions.append(output.cpu().numpy())
        target.append(gt_img.cpu().numpy())
        idx += len(LQ)
        torch.cuda.empty_cache()

    predictions = np.concatenate(predictions, axis=0)[:,None]
    target = np.concatenate(target, axis=0)[:,None]
    predictions= stitch_predictions(predictions, dset)
    target = stitch_predictions(target, dset)
    pixel_count = get_trimmed_pixel_count(predictions)
    return predictions[:,:-pixel_count, :-pixel_count].copy(), target[:,:-pixel_count, :-pixel_count].copy()


if __name__ == "__main__":
    main()
