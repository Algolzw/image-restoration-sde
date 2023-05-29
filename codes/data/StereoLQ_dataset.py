import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class StereoLQDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and LR image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LQ_paths = None
        self.LR_env = None  # environment for lmdb
        self.LR_size = opt["LR_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LQ_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
        elif opt["data_type"] == "img":
            self.LQ_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.LQ_paths, "Error: LQ paths are empty."

        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.LR_env = lmdb.open(
            self.opt["dataroot_LR"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.LR_env is None:
                self._init_lmdb()

        LR_path_L, LR_path_R = None, None
        scale = self.opt["scale"]
        LR_size = self.opt["LR_size"]

        # get LR image
        LR_path_L = self.LQ_paths[index*2]
        LR_path_R = self.LQ_paths[index*2+1]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.LR_sizes[index].split("_")]
        else:
            resolution = None
        imgL_LR = util.read_img(self.LR_env, LR_path_L, resolution)  # return: Numpy float32, HWC, BGR, [0,1]
        imgR_LR = util.read_img(self.LR_env, LR_path_R, resolution)  # return: Numpy float32, HWC, BGR, [0,1]

        # change color space if necessary
        if self.opt["color"]:
            imgL_LR, imgR_LR = util.channel_convert(
                imgL_LR.shape[2], self.opt["color"], [imgL_LR, imgR_LR])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if imgL_LR.shape[2] == 3:
            imgL_LR = imgL_LR[:, :, [2, 1, 0]]
            imgR_LR = imgR_LR[:, :, [2, 1, 0]]
        imgL_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(imgL_LR, (2, 0, 1)))).float()
        imgR_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(imgR_LR, (2, 0, 1)))).float()

        img_LR = torch.cat([imgL_LR, imgR_LR], dim=0)

        return {"LQ": img_LR, "LQ_path": LR_path_L}

    def __len__(self):
        return len(self.LQ_paths) // 2
