import os, sys
import cv2
import torch
import argparse
import numpy as np
sys.path.insert(0, "../")
from utils.sde_utils import IRSDE

def interpolate(sde, source, target, save_dir):
    sde.set_mu(target)
    sde.forward(source, save_dir=save_dir)

def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    sde = IRSDE(max_sigma=1, T=100, device=device)

    print(args.source, args.target, args.save)
    os.makedirs(args.save, exist_ok=True)

    source_numpy = cv2.imread(args.source) / 255.
    target_numpy = cv2.imread(args.target) / 255.

    # bgr to rgb
    source_numpy = source_numpy[:, :, [2, 1, 0]]
    target_numpy = target_numpy[:, :, [2, 1, 0]]

    assert source_numpy.shape == target_numpy.shape

    source = torch.tensor(source_numpy).permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(target_numpy).permute(2, 0, 1).unsqueeze(0)

    if args.cuda:
        source = source.cuda()
        target = target.cuda()

    interpolate(sde, source, target, args.save)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='interpolate source to target')
    parser.add_argument('-s', '--source', type=str, default='', help='path of the source image')
    parser.add_argument('-t', '--target', type=str, default='', help='path of the target image')
    parser.add_argument('--save', type=str, help='dir to save the interpolation images')
    parser.add_argument('--cuda', action="store_true", help='use cuda')
    args = parser.parse_args()

    main(args)
