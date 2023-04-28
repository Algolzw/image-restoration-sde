import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from models.modules.loss import MatchingLoss, PerceptualMatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class LatentModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        # else:
        #     self.model = DataParallel(self.model)
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            # self.loss_fn = PerceptualMatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            # self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, LQ, GT=None):
        self.LQ = LQ.to(self.device)  # LQ
        self.GT = GT.to(self.device) if GT is not None else None

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        if self.opt["dist"]:
            encode_fn = self.model.module.encode
            decode_fn = self.model.module.decode
        else:
            encode_fn = self.model.encode
            decode_fn = self.model.decode

        L_lq, H_lq = encode_fn(self.LQ)
        L_gt, H_gt = encode_fn(self.GT)

        rec_llq_hlq = decode_fn(L_lq, H_lq) # latent LQ, hidden LQ
        rec_lgt_hlq = decode_fn(L_gt, H_lq) # latent GT, hidden LQ

        loss_rec = self.loss_fn(rec_llq_hlq, self.LQ) #+ self.loss_fn(rec_lgt_hgt, self.GT)
        loss_rep = self.loss_fn(rec_lgt_hlq, self.GT) #+ self.loss_fn(rec_llq_hgt, self.LQ)

        # loss_reg = torch.mean(L_lq**2) + torch.mean(L_gt**2)
        loss_reg = (L_lq.mean()-self.LQ.mean()).abs() + (L_lq.std() - self.LQ.std()*0.5).abs()

        loss = loss_rec + loss_rep + loss_reg * 0.001
        loss.backward()
        self.optimizer.step()

        # set log
        self.log_dict["loss_rec"] = loss_rec.item()
        self.log_dict["loss_rep"] = loss_rep.item()
        self.log_dict["loss_reg"] = loss_reg.item()

    def test(self):
        self.model.eval()

        if self.opt["dist"]:
            encode_fn = self.model.module.encode
            decode_fn = self.model.module.decode
        else:
            encode_fn = self.model.encode
            decode_fn = self.model.decode

        with torch.no_grad():
            L_lq, H_lq = encode_fn(self.LQ)
            L_gt, H_gt = encode_fn(self.GT)

            self.real_lq = decode_fn(L_lq, H_lq) # latent LQ, hidden LQ
            self.fake_gt = decode_fn(L_gt, H_lq) # latent GT, hidden LQ

            self.fake_lq = decode_fn(L_lq, H_gt) # latent LQ, hidden GT
            self.real_gt = decode_fn(L_gt, H_gt) # latent GT, hidden GT

        self.model.train()

        tvutils.save_image(self.LQ.data, f'image/LQ.png', normalize=False)
        tvutils.save_image(self.GT.data, f'image/GT.png', normalize=False)
        
        tvutils.save_image(self.fake_gt.data, f'image/GT_fake.png', normalize=False)
        tvutils.save_image(self.fake_lq.data, f'image/LQ_fake.png', normalize=False)
        tvutils.save_image(self.real_gt.data, f'image/GT_real.png', normalize=False)
        tvutils.save_image(self.real_lq.data, f'image/LQ_real.png', normalize=False)

        # tvutils.save_image(L_lq[:, :].data, f'image/LQ_latent.png', normalize=False)
        # tvutils.save_image(L_gt[:, :].data, f'image/GT_latent.png', normalize=False)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.LQ.detach()[0].float().cpu()
        out_dict["Output"] = self.fake_gt.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.GT.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)

