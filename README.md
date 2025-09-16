# SAOC
This repository contains code to run the Structural Aware Orthogonal Constraint (SAOC) Clinical outcome prediction system.

## Dataset 
Thanks for open-souring the project by beep https://github.com/allenai/BEEP/tree/main, we utilize the same strategy to filter the dataset from the MIMIC-III dataset ([download here](https://physionet.org/content/mimiciii-demo/1.4/)). You will also need to process and segment the dataset by the strategy in (https://github.com/LuChang-CS/semi-structured-icd-coding).

## Code Setup
This code was developed in python 3.8 using the libraries listed in environment.yml. The easiest way to run this code is to set up a conda environment using the .yml file via the following command:

```conda env create -f environment.yml```

Activate the conda environment using the command: ```conda activate beep-env```


## Replicating Outcome Prediction Results
To replicate any of our training processes, you only need to run using the following command:

sbatch --export=ALL,init_model=$Model_Name,module_type=('Orth_inly'/'Co_orth'),use_section='true',batch_size=16,accumulation_steps=48,num_heads=0,orth_weight=0.4,delta=0.05 mp_wandb_adm.sbatch
