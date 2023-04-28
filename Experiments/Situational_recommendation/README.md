# SiTunes Dataset: Situational Recommendation Experiment

This repository provides instructions to reproduce the situational recommendation experiment using the SiTunes dataset and RecBole.

## Step 1: Generate Datasets

To create the 30 different datasets, run the `dataloader.py` script located at:
`SiTunes_dataset/Experiments/Situational_recommendation/dataset/dataloader.py`


30  dataset splits will for 10 different random seeds, ranging from 101 to 110 for each of the 3 settings will be generated in the 
`SiTunes_dataset/Experiments/Situational_recommendation/dataset/` folder

## Step 2: Install RecBole

Download RecBole into the `SiTunes_dataset/Experiments/` directory using the following command:

<pre>
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
</pre>


Alternatively, you can follow other installation methods available in the [RecBole documentation](https://recbole.io).

## Step 3: Copy Datasets and Configs

Copy the `dataset` and `Configs` folders to the installed RecBole repository for convenience:

- Dataset folder: `SiTunes_dataset/Experiments/Situational_recommendation/dataset/`
- Configs folder: `SiTunes_dataset/Experiments/Situational_recommendation/Configs/`

Configs are created for each experiment setting and the 4 models used in the experiment.

## Step 4: Run the Experiment

To run the experiment, follow these steps:

1. Choose a desired model and setting (also consider with/without situation).
2. Find the correspondig config file.
3. For each run, select the appropriate dataset and modify the seed in the corresponding config file according to the selected model over 10 random seeds [101,110].
4. Take the average result of each evaluation metric produced as output of the 10 runs.

Please make sure to update the dataset path and the config file path in your RecBole experiment commands accordingly.
