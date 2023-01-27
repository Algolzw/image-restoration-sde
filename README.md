
## Image Restoration with Mean-Reverting Stochastic Differential Equations

This is the implementation of paper "Image Restoration with Mean-Reverting Stochastic Differential Equations".

![IR-SDE](figs/overview.png)


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

### Dataset Preparation

We employ Rain100H datasets for training (totally 1,800 images) and test (100 images). 

Download [training](http://www.icst.pku.edu.cn/struct/att/RainTrainH.zip) and [testing](http://www.icst.pku.edu.cn/struct/att/Rain100H.zip) datasets and process it in a way such that rain images and no-rain image are in separately directories, as

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

---

#### --- Thanks for your interest! --- ####

