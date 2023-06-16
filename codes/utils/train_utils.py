import imageio
import os
import wandb
import numpy as np

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

        
