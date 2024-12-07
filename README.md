# HouseDiffusion
**[HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising](https://arxiv.org/abs/2211.13287)**
<img src='figs/teaser.png' width=100%>
## Installation
**1. Clone our repo and install the requirements:**

Our implementation is based on the public implementation of [guided-diffusion](https://github.com/openai/guided-diffusion). For installation instructions, please refer to their repository. Keep in mind that our current version has not been cleaned and some features from the original repository may not function correctly.

```
git clone https://github.com/aminshabani/house_diffusion.git
cd house_diffusion
pip install -r requirements.txt
pip install -e .
```
I basically reproduced the results of the original paper and the original code on December 5, 2024. I use Linux system (no GUI) and GTX3090 GPU. There are many details in the original code that are not clear, and there are many python libraries that cannot be used today.

You need to modify the "requirements.txt file", and change "torch==2.0.0.dev20221212" to "torch==2.0.0." Because the original version is no longer valid.You also need to delete "mpi4py==3.1.4", and this library cannot be installed directly through "pip" now. You can install "Conda Install-C Conda-Forge mpi4py" at the terminal.
```
conda install -c conda-forge mpi4py
```

**2. Download the dataset and create the datasets directory**

- You can download the datasets from [RPLAN's website](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) or by filling [this](https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform) form.
- We also use data preprocessing from House-GAN++ which you can find in [this](https://github.com/sepidsh/Housegan-data-reader) link.
Put all of the processed files from the downloaded dataset in a `datasets` folder in the current directory:

```
house_diffusion
├── datasets
│   ├── rplan
|   |   └── 0.json
|   |   └── 1.json
|   |   └── ...
|   └── ...
└── guided_diffusion
└── scripts
└── ...
```
- We have provided a temporary model that you can download from [Google Drive](https://drive.google.com/file/d/16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR/view?usp=share_link).
  
In fact, this file path is wrong, but I don't know why no one mentioned it. I think the correct thing should be this:
```
house_diffusion
├── datasets
│   ├── rplan
|   |   └── 0.json
|   |   └── 1.json
|   |   └── ...
|   |   └── list.txt
|   └── ...
|
└── scripts
│   ├── guided_diffusion
│   ├── house_diffusion   
│   ├── image_sample.py   
│   ├── image_train.py
│   ├── outputs
│   ├── processed_rplan   
│   └──  ckpts
│        └── exp
│             └──  model250000.pt
└── ......
```
In short, you need to ensure that "house_diffusion", "image_train.py" and "image_sample.py" are in the same directory.You also need to add two folders "outputs" and "processed_rplan", otherwise you may report an error.You also need to add a "list.txt" file to the "rplan" folder.

## Running the code

Firstly, make sure your current directory is the scripts folder.
```
pwd
cd scripts
```

**1. Training**

Note that it may take 20 hours to train from 0 step to 1000000 step. Please make sure you have enough time. I use the GPU of GTX3090.

You can run a single experiment using the following command:
```
python image_train.py --dataset rplan --batch_size 32 --set_name train --target_set 8
```

If you want to train from a saved checkpoint (such as model250000.pt), you can use the following code:
```
python image_train.py --dataset rplan --batch_size 32 --set_name train --target_set 8 --resume_checkpoint "/root/house_diffusion/scripts/ckpts/exp/model250000.pt"
```
Please correct the path of `-resume_checkpoint` .




**2. Sampling**
To sample floorplans, you can run the following command from inside of the `scripts` directory. To provide different visualizations, please see the `save_samples` function from `scripts/image_sample.py`

```
python image_sample.py --dataset rplan --batch_size 32 --set_name eval --target_set 8 --model_path ckpts/exp/model250000.pt --num_samples 64
```
You can also run the corresponding code from `scripts/script.sh`. 


## Citation

```
@article{shabani2022housediffusion,
  title={HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising},
  author={Shabani, Mohammad Amin and Hosseini, Sepidehsadat and Furukawa, Yasutaka},
  journal={arXiv preprint arXiv:2211.13287},
  year={2022}
}
```
