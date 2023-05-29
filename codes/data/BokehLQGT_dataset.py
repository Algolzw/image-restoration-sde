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


class BokehLQGTDataset(data.Dataset):
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

        # read image list from image files
        if opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
            self.alpha_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_alpha"]
            )  # Alpha list
            self.metas = self._read_meta_data(opt["dataroot_meta"])
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

    def _read_meta_data(self, meta_file_path: str):
        """Read the meta file containing source / target lens and disparity for each image.
        Args:
            meta_file_path (str): File path
        Raises:
            ValueError: File not found.
        Returns:
            dict: Meta dict of tuples like {id: (id, src_lens, tgt_lens, disparity)}.
        """
        if not os.path.isfile(meta_file_path):
            raise ValueError(f"Meta file missing under {meta_file_path}.")

        meta = {}
        with open(meta_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
            meta[id] = (src_lens, tgt_lens, disparity)
        return meta

    def lenstr2tensor(self, lenstr, scale=1.):
        # Canon50mm -> -1, Sony50mm -> 1
        lenstr = lenstr.replace('Canon50mmf', '-')
        lenstr = lenstr.replace('Sony50mmf', '')
        lenstr = lenstr.replace('BS', '')
        return torch.tensor(float(lenstr)) * scale

    def __getitem__(self, index):

        GT_path, LR_path = None, None
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        img_GT = util.read_img(self.GT_env, GT_path, None)  # return: Numpy float32, HWC, BGR, [0,1]

        # get LR image
        LR_path = self.LR_paths[index]
        img_LR = util.read_img(self.LR_env, LR_path, None)

        # get LR image
        alpha_path = self.alpha_paths[index]
        img_alpha = util.read_img(None, alpha_path, None)

        id = os.path.basename(alpha_path).split(".")[0]
        src_lens, tgt_lens, disparity = self.metas[id]

        src_lens = self.lenstr2tensor(src_lens, scale=10.)
        tgt_lens = self.lenstr2tensor(tgt_lens, scale=10.)
        disparity = self.lenstr2tensor(disparity, scale=1.)

        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape
            assert LR_size == GT_size, "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            img_GT = img_GT[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]
            img_alpha = img_alpha[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LR, img_GT, img_alpha = util.augment(
                [img_LR, img_GT, img_alpha],
                self.opt["use_flip"],
                self.opt["use_rot"],
                swap=False,
                mode=self.opt["mode"],
            )
        elif LR_size is not None:
            H, W, C = img_LR.shape
            assert LR_size == GT_size, "GT size does not match LR size"

            if LR_size < H and LR_size < W:
                # center crop
                rnd_h = H // 2 - LR_size//2
                rnd_w = W // 2 - LR_size//2
                img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
                img_GT = img_GT[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]
                img_alpha = img_alpha[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]
                
        # change color space if necessary
        if self.opt["color"]:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.opt["color"], [img_LR])[
                0
            ]  # TODO during val no definition
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()
        img_alpha = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_alpha, (2, 0, 1)))
        ).float()

        if self.opt["phase"] == "train" \
            and self.opt['use_swap'] and random.random() < 0.5 \
            and (src_lens > 100 or tgt_lens > 100):
            return {
                "LQ": img_GT, 
                "GT": img_LR,
                "alpha": img_alpha,
                "src_lens": tgt_lens,
                "tgt_lens": src_lens,
                "disparity": disparity, 
                "LQ_path": GT_path, 
                "GT_path": LR_path,
                }
        else:
            return {
                "LQ": img_LR, 
                "GT": img_GT,
                "alpha": img_alpha,
                "src_lens": src_lens,
                "tgt_lens": tgt_lens,
                "disparity": disparity, 
                "LQ_path": LR_path, 
                "GT_path": GT_path,
                }

    def __len__(self):
        return len(self.GT_paths)
