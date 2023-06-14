"""create dataset and dataloader"""
import logging

import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt["phase"]
    if phase == "train":
        if opt["dist"]:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt["n_workers"]
            assert dataset_opt["batch_size"] % world_size == 0
            batch_size = dataset_opt["batch_size"] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt["n_workers"] * len(opt["gpu_ids"])
            batch_size = dataset_opt["batch_size"]
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(phase=="val")
        )


def create_dataset(dataset_opt):
    mode = dataset_opt["mode"]
    if mode == "LQ":  # Predictor
        from data.LQ_dataset import LQDataset as D
        dataset = D(dataset_opt)
    elif mode == "LQGT":  # SFTMD
        from data.LQGT_dataset import LQGTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGT_splitting':
        from data.splitting_dataset import SplittingDataset as D
        from core.data_split_type import DataSplitType
        dataset = D(dataset_opt)
    elif mode == "GT":  # Corrector
        from data.GT_dataset import GTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'SteLQGT':
        from data.StereoLQGT_dataset import StereoLQGTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'SteLQ':
        from data.StereoLQ_dataset import StereoLQDataset as D
        dataset = D(dataset_opt)
    elif mode == 'BokehLQGT':
        from data.BokehLQGT_dataset import BokehLQGTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'BokehLQ':
        from data.BokehLQ_dataset import BokehLQDataset as D
        dataset = D(dataset_opt)
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
