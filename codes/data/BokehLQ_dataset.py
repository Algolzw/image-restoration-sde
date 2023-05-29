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


class BokehLQDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths = None
        self.LR_env = None  # environment for lmdb
        self.LR_size = opt["LR_size"]

        # read image list from image files
        if opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.metas = self._read_meta_data(opt["dataroot_meta"])
        else:
            print("Error: data_type is not matched in Dataset")

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

        # get LR image
        LR_path = self.LR_paths[index]
        img_LR = util.read_img(self.LR_env, LR_path, None)

        id = os.path.basename(LR_path).split(".")[0]
        src_lens, tgt_lens, disparity = self.metas[id]

        src_lens = self.lenstr2tensor(src_lens, 10)
        tgt_lens = self.lenstr2tensor(tgt_lens, 10)
        disparity = self.lenstr2tensor(disparity)

        # change color space if necessary
        if self.opt["color"]:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.opt["color"], [img_LR])[
                0
            ]  # TODO during val no definition
            

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        return {
            "LQ": img_LR, 
            "src_lens": src_lens,
            "tgt_lens": tgt_lens,
            "disparity": disparity, 
            "LQ_path": LR_path, 
            }

    def __len__(self):
        return len(self.LR_paths)
