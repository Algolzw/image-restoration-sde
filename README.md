
## Image Restoration with Mean-Reverting Stochastic Differential Equations <br><sub>Official PyTorch Implementation of IR-SDE [[Paper]](https://arxiv.org/abs/2301.11699)[[Project]](https://algolzw.github.io/ir-sde/index.html)</sub>


![IR-SDE](figs/overview.png)
You can find more details and results from our [Project page](https://algolzw.github.io/ir-sde/index.html).


## Dependenices

* OS: Ubuntu 20.04
* nvidia :
	- cuda: 11.7
	- cudnn: 8.5.0
* python3
* pytorch >= 1.13.0
* Python packages: `pip install -r requirements.txt`

## How to use our Code?

Here we provide an example for **image deraining** task, but can be changed to any problem with replacing the dataset. 

We retrained the deraining model from scratch on 4 Nvidia TITAN XP GPUs and find that it achieves a **new SOTA even in terms of PSNR** on Rain100H:

| Method |  PSNR   | SSIM  | LPIPS   | FID  |
| :--- |  :----:  | :----:  | :----:  | :----:  |
| **IR-SDE** | **41.65**  | **0.9041** | **0.047** | **18.64** |
| Restormer | 41.46  | 0.904 | - | - |
| MPRNet | 30.41 | 0.8906 | 0.158 | 61.59 |
| PReNet | 29.46 | 0.8990 | 0.128 | 52.67 |

Note that **we didn't tune any parameter**, the last saved checkpoint was used to evaluation.

The pretrained model is provided [here](https://drive.google.com/file/d/1o6ELATcKOw96Uno8rJVB20vcLRWWBnu2/view?usp=share_link), and the performances of other SOTAs can be find in [here](https://paperswithcode.com/sota/single-image-deraining-on-rain100h).

### Dataset Preparation

We employ Rain100H datasets for training (totally 1,800 images) and testing (100 images). 

Download [training](http://www.icst.pku.edu.cn/struct/att/RainTrainH.zip) and [testing](http://www.icst.pku.edu.cn/struct/att/Rain100H.zip) datasets and process it in a way such that rain images and no-rain images are in separately directories, as

```bash
#### for training dataset ####
datasets/rain/trainH/GT
datasets/rain/trainH/LQ


#### for testing dataset ####
datasets/rain/testH/GT
datasets/rain/testH/LQ

```

Then get into the `codes/config/deraining` directory and modify the dataset paths in option files in 
`options/derain/train/train_sde_derain.yml` and `options/derain/test/test_sde_derain.yml`.


### Train
The main code for training is in `codes/config/deraining` and the core algorithms for IR-SDE is in `codes/utils/sde_utils.py`.

You can train the model following below bash scripts:

```bash
cd codes/config/deraining

# For single GPU:
python3 train.py -opt=options/derain/train/train_sde_derain.yml

# For distributed training, need to change the gpu_ids in option file
python3 -m torch.distributed.launch --nproc_per_node=2 --master_poer=4321 train.py -opt=options/derain/train/train_sde_derain.yml --launcher pytorch
```

Then the models and training logs will save in `log/derain_sde/`. 
You can print your log at time by running `tail -f log/derain_sde/train_derain_sde_***.log -n 100`.

### Evaluation
To evalute our method, please modify the benchmark path and model path and run

```bash
cd codes/config/deraining
python test.py -opt=options/derain/test/test_sde_derain.yml
```

### Some Results
![IR-SDE](figs/results.png)

### Interpolation
We also provide a interpolation demo to perform interpolation between two images in `codes/demos/interpolation.py`, the usage is:

```bash
cd codes/demos
python interpolation.py -s source_image_path -t target_image_path --save save_dir
```

#### Example of interpolation:
![IR-SDE](figs/interpolation.png)


## Citations
If our code helps your research or work, please consider citing our paper.
The following is a BibTeX reference:

```
@article{luo2023image,
  title={Image Restoration with Mean-Reverting Stochastic Differential Equations},
  author={Luo, Ziwei and Gustafsson, Fredrik K and Zhao, Zheng and Sj{\"o}lund, Jens and Sch{\"o}n, Thomas B},
  journal={arXiv preprint arXiv:2301.11699},
  year={2023}
}
```

---

#### Contact
E-mail: ziwei.luo@it.uu.se

#### --- Thanks for your interest! --- ####
![visitors](https://visitor-badge.glitch.me/badge?page_id=Algolzw/image-restoration-sde)
