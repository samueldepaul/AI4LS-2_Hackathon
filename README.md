# AI for Life Sciences Hackathon (2nd Edition) - Task 1 Solution

## Overview

This repository contains my porposed solution for **Task 1** of the **AI for Life Sciences (AI4LS) Hackathon - Second Edition** and access to the code and documentation that generated the solution for **Task 2** through the folder `supplemental_material_for_task_2`. 

The first task involved forecasting groundwater levels for specific locations within Austria, covering the period from January 2022 to June 2024.

The repository includes various Jupyter notebooks documenting the entire process: from data processing and integration, to the selection of exogenous variables and model evaluation.

In this `README.md`, we first explain how to train the model for a single location and make the corresponding predictions. Afterward, we explain the methodology used to solve the challenges, while providing a brief summary of the content of each notebook that makes up the project.


## Table of Contents

0. [How to Run](#how-to-run)
1. Preliminary Remarks
2. [Data Processing and Cleaning](#data-processing-and-cleaning)
4. [Exogenous Variables Integration](#exogenous-variables-integration)
5. [Model and Exogenous Variables Selection](#model-and-exogenous-variables-selection)
6. [Re-training and Final Predictions](#re-training-and-final-predictions)
7. [Results](#results)
8. [License](#license)

## Data Processing and Cleaning

**Notebook**: `Data_Processing_and_Cleaning.ipynb`

In this notebook, the initial data processing pipeline is implemented, followed by an exploratory data analysis (EDA) to better understand the structure and distribution of the dataset. Some intermediate files are saved to avoid reprocessing long-running computations.

- **Groundwater Level Time Series**: Historical time series of groundwater levels from March 1930 to December 2021 are retrieved for each location from the [Austria groundwater data](https://ehyd.gv.at/).
- **Dataframe Construction**: The notebook consolidates groundwater data from multiple CSV files, extracts geographical information (coordinates, terrain elevation, etc.), and merges the data into a single DataFrame for further analysis.
- **Dataset Preparation**: The dataset is transformed to include `lag` features (past observations) and `step` features (future values to predict).
- **Feature Engineering**: Temporal features and historical aggregates are generated, ensuring no data leakage.
- **Data Reduction**: Since exogenous data prior to 1960 is hard to find, observations before 1960 are discarded, reducing the dataset by only 1%.

## Exogenous Variables Integration

**Notebook**: `Exogenous_Variables_Integration.ipynb`

This notebook integrates exogenous variables into the dataset to enrich the input features for machine learning model training. 

- **Climate-related Variables**: Exogenous variables related to climate are merged into the dataset.
- **Additional Exogenous Variables**: Variables related to soil, land cover, population, and distance to the nearest city are iteratively integrated. These variables are stored in files with a similar structure, facilitating the merging process.

## Model and Exogenous Variables Selection

**Notebook**: `Model_Exogenous_Variable_Selection.ipynb`

This notebook focuses on model training and the selection of exogenous variables.

- **Variable Classification**: Variables are categorized into essential variables (past lags and future steps), temporal variables, climatological variables, and more.
- **Exploratory Modeling**: Various models, including tree-based methods (XGBoost, CatBoost, LightGBM) and classical models (Ridge, ElasticNet), are fitted using Randomized Search and Cross Validation to find the best combination of variables and hyperparameters.
- **Performance Analysis**: The notebook evaluates model performance using blind test data. It was observed that models with the largest feature sets perform the best for many locations, but no clear variable set emerged as consistently superior.
- **Combined Approach**: Based on performance, a combined approach is adopted: each location is predicted using the model and set of features that perform best for that location.

## Re-training and Final Predictions

**Notebook**: `Re-train_and_Make_Predictions.ipynb`

This final notebook re-trains the models with the best-performing hyperparameters and makes predictions for each location. 

- **Best Models and Hyperparameters**: The best models and their configurations are loaded from a JSON file. 
- **Final Predictions**: The code pipeline handles memory constraints and dynamically selects the best model for each location, making predictions that are uploaded to the Taikai platform.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/samueldepaul/AI4LS-2_Hackathon.git
