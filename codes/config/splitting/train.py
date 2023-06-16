import sys

import wandb
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
# from IPython import embed

from models import create_model
from core.psnr import RangeInvariantPsnr

sys.path.insert(0, "../../")
import utils as util
from utils.stitch_prediction import stitch_predictions
from utils.train_utils import (get_trimmed_pixel_count, save_imgs, get_opt_dict,
                               get_resume_state,init_directories_logging,get_dataloaders,
                               get_total_epochs_iters, publish_model_logs)
# torch.autograd.set_detect_anomaly(True)


def evaluate_validation(model, sde, val_set, val_loader, opt, current_step, epoch, best_psnr):
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
        print("Saving models and training states.", current_step)
        model.save('best')
        model.save_training_state(epoch, current_step, filename='best')


    # log
    wandb.log({"val_psnr":avg_psnr})
    wandb.log({'step':current_step})

    # print("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
    # logger_val = logging.getLogger("val")  # validation logger
    # logger_val.info(
    #     "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
    #         epoch, current_step, avg_psnr
    #     )
    # )
    print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
            epoch, current_step, avg_psnr
        ))
    # # tensorboard logger
    # if opt["use_tb_logger"] and "debug" not in opt["name"]:
    #     tb_logger.add_scalar("psnr", avg_psnr, current_step)
    return best_psnr

def training_step(model, sde, train_data, opt, current_step):
    LQ, GT = train_data["LQ"], train_data["GT"]
    timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)

    model.feed_data(states, LQ, GT) # xt, mu, x0
    model.optimize_parameters(current_step, timesteps, sde)
    model.update_learning_rate(
        current_step, warmup_iter=opt["train"]["warmup_iter"]
    )

def main():
    opt, rank = get_opt_dict()
    # choose small opt for SFTMD test, fill path of pre-trained model_F
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    resume_state = get_resume_state(opt)

    #### mkdir and loggers
    init_directories_logging(opt,rank, resume_state)


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    data_dict = get_dataloaders(opt, rank, dataset_ratio)
    val_set = data_dict['dataset']['val']
    train_set = data_dict['dataset']['train']
    val_loader = data_dict['loader']['val']
    train_loader = data_dict['loader']['train']
    total_epochs, total_iters = get_total_epochs_iters(train_set, opt)

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

            training_step(model, sde, train_data, opt, current_step)

            if current_step % opt["logger"]["print_freq"] == 0:
                publish_model_logs(model, opt, rank, epoch, current_step)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                best_psnr = evaluate_validation(model, sde, val_set, val_loader, opt, current_step, epoch, best_psnr)

            if error.value:
                sys.exit(0)

        if rank <= 0:
            # print("Saving the final model.")
            model.save("latest")
            # print("End of Predictor and Corrector training.")





def get_dset_prediction(model, sde, dset, dloader):
    idx = 0
    predictions = []
    target = []
    patchwise_psnr = []
    for val_data in tqdm(dloader):

        LQ, GT = val_data["LQ"], val_data["GT"]
        noisy_state = sde.noise_state(LQ)

        # valid Predictor
        model.feed_data(noisy_state, LQ, GT)
        model.test(sde)
        visuals = model.get_current_visuals()

        output = visuals["Output"]#.squeeze().cpu().detach().numpy()
        gt_img = visuals["GT"]#.squeeze().cpu().detach().numpy()
        patchwise_psnr.append(RangeInvariantPsnr(gt_img, output).mean().item())

        # store for tiled prediction
        predictions.append(output.cpu().numpy())
        target.append(gt_img.cpu().numpy())
        idx += len(LQ)

    predictions = np.concatenate(predictions, axis=0)[:,None]
    target = np.concatenate(target, axis=0)[:,None]
    predictions= stitch_predictions(predictions, dset)
    target = stitch_predictions(target, dset)
    pixel_count = get_trimmed_pixel_count(predictions)
    return predictions[:,:-pixel_count, :-pixel_count].copy(), target[:,:-pixel_count, :-pixel_count].copy()

if __name__ == "__main__":
    main()
