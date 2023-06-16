import imageio
import os
import wandb
import numpy as np
import socket
import argparse
import math
import sys

import torch.multiprocessing as mp
import torch
import torch.distributed as dist

sys.path.insert(0, "../")
import options as option
import utils as util



from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

def get_trimmed_pixel_count(pred):
    """
    Return a number of pixels for which no predictions were made because they were at the edge.
    """
    ignored_pixels = 1
    while(pred[0,-ignored_pixels:,-ignored_pixels:,].std() ==0):
        ignored_pixels+=1
    ignored_pixels-=1
    return ignored_pixels

def save_imgs(prediction, target, current_step, direc, save_to_wandb=False):
    if not os.path.exists(direc):
        os.mkdir(direc)
    idx = 0
    imageio.imwrite(os.path.join(direc,f'target_{idx}.png'), target[idx].astype(np.int16), format='png')
    imageio.imwrite(os.path.join(direc,f'prediction_{idx}_{current_step}.png'), 
                    prediction[idx].astype(np.int16), format='png')
    if save_to_wandb:
        wandb.log({"prediction": [wandb.Image(prediction)]})

        

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


def get_opt_dict():
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
    assert args.launcher == "none", "Look at the code for distributed"
    opt["dist"] = False
    opt["dist"] = False
    rank = -1
    print("Disabled distributed training.")
    return opt, rank

def get_resume_state(opt):
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
    return resume_state

def init_directories_logging(opt,rank, resume_state):
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
        prefix= os.path.dirname(os.path.dirname(os.path.dirname(opt["path"]["experiments_root"])))
        expname = opt["path"]["experiments_root"][len(prefix)+1:]
        hostname = socket.gethostname()
        wandb.init(name=os.path.join(hostname, expname),
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
        #         # from tensorboardX import SummaryWriter
        #     # tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        raise NotImplementedError('handle this case')
        # util.setup_logger(
        #     "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        # )
        # logger = logging.getLogger("base")

def get_dataloaders(opt, rank, dataset_ratio):
    for phase in ['train', 'val']:
        dataset_opt = opt["datasets"][phase]
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            # total_iters = int(opt["train"]["niter"])
            # total_epochs = int(math.ceil(total_iters / train_size))
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
    
    return {'dataset':{'val':val_set,'train':train_set}, 'loader':{'val':val_loader,'train':train_loader}}

def get_total_epochs_iters(train_set, opt):
    dataset_opt = opt["datasets"]['train']
    train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
    total_iters = int(opt["train"]["niter"])
    total_epochs = int(math.ceil(total_iters / train_size))
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

    return total_epochs, total_iters

def publish_model_logs(model, opt, rank, epoch, current_step):
    logs = model.get_current_log()
    message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
        epoch, current_step, model.get_current_learning_rate()
    )
    for k, v in logs.items():
        message += "{:s}: {:.4e} ".format(k, v)
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            if rank <= 0:
                wandb.log({k: v,'step':current_step})#, current_step)
    if rank <= 0:
        print(message)
