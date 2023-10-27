[![HOME](https://user-images.githubusercontent.com/85012818/205025966-a8e7ce83-e43c-447c-89ad-b0430561dab0.png)](https://bmvc2022.mpi-inf.mpg.de/839/)
<p align="center"> 
<a href="https://britishmachinevisionassociation.github.io/bmvc" ><img src="https://img.shields.io/badge/HOME-BMVC-blue.svg"></a>
<a href="https://bmvc2022.mpi-inf.mpg.de/839/" ><img src="https://img.shields.io/badge/HOME-Paper-important.svg"></a>
<a href="https://bmvc2022.mpi-inf.mpg.de/0839.pdf" ><img src="https://img.shields.io/badge/PDF-Paper-blueviolet.svg"></a>
<a href="https://bmvc2022.mpi-inf.mpg.de/0839_poster.pdf" ><img src="https://img.shields.io/badge/-Poster-ff69b7.svg"></a>
<a href="https://bmvc2022.mpi-inf.mpg.de/0839_video.mp4" ><img src="https://img.shields.io/badge/-Video-brightgreen.svg"></a>
<a href="https://bmvc2022.mpi-inf.mpg.de/0839_supp.pdf" ><img src="https://img.shields.io/badge/-Supplementary-green.svg"></a>
<a href="https://drive.google.com/file/d/1ixW7l4HMrvcSg6KOadQwD--VMBdegL3c/view?usp=sharing" ><img src="https://img.shields.io/badge/-WeightsFiles-blue.svg"></a>
</p>

# Architecture

![archioverall](https://user-images.githubusercontent.com/85012818/205044316-995d7972-93ad-4673-b10d-44fd731913a7.png)
<!-- ![1669618815992](https://user-images.githubusercontent.com/85012818/204213933-9d91852f-cad1-45a1-94a3-23da70fd1857.png) -->

# Usage
### Installation:
1. Create the environment from the <kbd>environment.yml</kbd> file:

        conda env create -f environment.yml

2. Activate the new environment:

        conda activate bmvc

3. Verify that the new environment was installed correctly:

        conda env list

You can also use <kbd>conda info --envs</kbd>.

### Configuration:
1. Reset the correct path of the data sets in <kbd>dataer.py</kbd>.
2. Reset the model in <kbd>models.py</kbd> and init it in <kbd>main.py</kbd>.
3. Ensure all the hyparameters is what you need and then:

        python main.py

You can also use command like this:

    python main.py -c <cuda.id> -e <epoch> -b <batch_size> -d <dataset> -fp <save_dir> -nn <model_name> -lr <learning_rate>

# Results
## Revision  
In Table1, Table2: KLCC -> KRCC  
In Table1: 0.9376 -> 0.8376
![matrixonpertex](https://user-images.githubusercontent.com/85012818/205043674-8a309e54-9443-4295-935c-be9d91b8ba75.png)

![texture-retrieval](https://user-images.githubusercontent.com/85012818/205043749-ecb5111b-1086-4e52-9acf-1fe7535eaf32.png)

# Citation

    @inproceedings{Wang_2022_BMVC,
        author    = {Weibo Wang and Xinghui Dong},
        title     = {Unifying the Visual Perception of Humans and Machines on Fine-Grained Texture Similarity},
        booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
        publisher = {{BMVA} Press},
        year      = {2022},
        url       = {https://bmvc2022.mpi-inf.mpg.de/0839.pdf}
    }
