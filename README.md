![](./imgs/Title.png)
<p align="center"> 
<a href="" ><img src="https://img.shields.io/badge/HOME-Paper-important.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/PDF-Paper-blueviolet.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/-Poster-ff69b7.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/-Video-brightgreen.svg"></a>
</p>

# Architecture

![](./imgs/arch_small.png)

### Shared Encoder-Decoder Network
![](./imgs/network.png)

# Usage
### Installation:
1. Create the environment from the <kbd>environment.yml</kbd> file:

        conda env create -f environment.yml

2. Activate the new environment:

        conda activate uie

3. Verify that the new environment was installed correctly:

        conda env list

You can also use <kbd>conda info --envs</kbd>.

### Train:
Use this line to train the model

        python train.py --cuda_id 0 --exp CCMSRNet.yml
### Test:
Use this line to predict results

        python test.py --cuda_id 0 --exp CCMSRNet.yml --ckpt ./weights/checkpoint.pth --input path_to_img_folder --output path_to_save_folder

# Results

![](./imgs/C60_half.png)
![](./imgs/RUIE_half.png)

# Citation
If our work is useful for your research, please cite our work

        @ARTICLE{10336777,
      author={Qi, Hao and Zhou, Huiyu and Dong, Junyu and Dong, Xinghui},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Deep Color-Corrected Multi-scale Retinex Network for Underwater Image Enhancement}, 
      year={2023},
      doi={10.1109/TGRS.2023.3338611}}
