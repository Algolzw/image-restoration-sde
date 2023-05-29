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


class StereoLQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path_L, LR_path_L = None, None
        scale = self.opt["scale"]
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        # get GT image
        GT_path_L = self.GT_paths[index*2]
        GT_path_R = self.GT_paths[index*2+1]

        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT_L = util.read_img(self.GT_env, GT_path_L, resolution)  # return: Numpy float32, HWC, BGR, [0,1]
        img_GT_R = util.read_img(self.GT_env, GT_path_R, resolution)  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_GT_L = util.modcrop(img_GT_L, scale)
            img_GT_R = util.modcrop(img_GT_R, scale)

        # get LR image
        if self.LR_paths:  # LR exist
            LR_path_L = self.LR_paths[index*2]
            LR_path_R = self.LR_paths[index*2+1]
            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            img_LR_L = util.read_img(self.LR_env, LR_path_L, resolution, scale=4)
            img_LR_R = util.read_img(self.LR_env, LR_path_R, resolution, scale=4)

        if self.opt["phase"] == "train":
            H, W, C = img_LR_L.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR_L = img_LR_L[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            img_LR_R = img_LR_R[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT_L = img_GT_L[rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :]
            img_GT_R = img_GT_R[rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LR_L, img_LR_R, img_GT_L, img_GT_R = util.augment(
                [img_LR_L, img_LR_R, img_GT_L, img_GT_R],
                self.opt["use_flip"],
                self.opt["use_rot"],
                swap=False,
                mode=self.opt["mode"],
            )
        elif LR_size is not None:
            H, W, C = img_LR_L.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            if LR_size < H and LR_size < W:
                # center crop
                rnd_h = H // 2 - LR_size//2
                rnd_w = W // 2 - LR_size//2
                img_LR_L = img_LR_L[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
                img_LR_R = img_LR_R[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT_L = img_GT_L[rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :]
                img_GT_R = img_GT_R[rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :]

        # change color space if necessary
        if self.opt["color"]:
            H, W, C = img_LR_L.shape
            img_LR_L, img_LR_R = util.channel_convert(
            C, self.opt["color"], [img_LR_L, img_LR_R])  # TODO during val no definition
            img_GT_L, img_GT_R = util.channel_convert(
                img_GT_L.shape[2], self.opt["color"], [img_GT_L, img_GT_R])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT_L.shape[2] == 3:
            img_GT_L = img_GT_L[:, :, [2, 1, 0]]
            img_GT_R = img_GT_R[:, :, [2, 1, 0]]
            img_LR_L = img_LR_L[:, :, [2, 1, 0]]
            img_LR_R = img_LR_R[:, :, [2, 1, 0]]
        img_GT_L = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_L, (2, 0, 1)))).float()
        img_GT_R = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_R, (2, 0, 1)))).float()
        img_LR_L = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_L, (2, 0, 1)))).float()
        img_LR_R = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_R, (2, 0, 1)))).float()

        img_GT = torch.cat([img_GT_L, img_GT_R], dim=0)
        img_LR = torch.cat([img_LR_L, img_LR_R], dim=0)

        if LR_path_L is None:
            LR_path_L = GT_path_L

        return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path_L, "GT_path": GT_path_L}

    def __len__(self):
        return len(self.GT_paths) // 2
