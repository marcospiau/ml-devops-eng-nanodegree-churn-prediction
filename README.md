# Predict Customer Churn
This repository contain my submission for the project **Predict Customer Churn**, final project for lesson $COLOCAR LESSONS$, first course of [ML DevOps Engineer Nanodegree Udacity](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

In this project, we use machine learning to predict which customers are more likely to cancel their account, using good pratices of coding blabla.


## How to setup the environment
We use miniconda to manage Python packages installation.

Install minconda (skip if Anaconda or Miniconda are installed):
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

The following command creates a conda env named `uda_1`.
```bash
conda env create -f environment.yml python=3.8 --name uda_1
```

Activate the environement:
```bash
conda activate uda_1
```
## Running Files
How do you run your files? What should happen when you run your files?
## Main script
The main script (`churn_library.py`) performs the entire modeling pipeline:
 - load data
 - perform EDA and save plots
 - performe feature engineering, treating quantitative and categorical features accordingly
 - fits a logistic regression model and perform a grid search over a random forest model, and saves objects that can be used for making predictions on new data
 - saves results for each model

The script can be executed with:
```bash
python3 churn_library.py
```


