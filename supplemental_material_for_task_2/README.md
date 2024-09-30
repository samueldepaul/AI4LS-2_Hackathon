# AI for Life Sciences Hackathon (2nd Edition) - Task 2 Solution

## Video Solution of Task 2:
This task required the participants of each team to submit a video summarizing their solution. My solution is available by clicking on the following link: https://drive.google.com/file/d/195k6ZjNAUabeaFG83punshN2iFvZJEUR/view?usp=sharing

<p align="center">
  <img src="misc/0930.gif" alt="Demo video" />
</p>
---

This repository contains the code and resources for solving the GRACE time-series forecasting problem, part of the **AI for Life Sciences Hackathon (Second Edition)**. The main goal is to predict Total Water Storage (TWS) anomalies over a 5-year time horizon using exogenous variables such as climate, vegetation, and soil properties.

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Jupyter Notebooks](#jupyter-notebooks)
4. [Combined Predictive Power Metric (CPPM) - Definition](#combined-predictive-power-metric-(cppm)-definition)
5. [Final Ranking of Exogenous Variables](final-ranking-of-expogenous-variables)
---

## Introduction

The GRACE mission provides global data on TWS anomalies, measured in Liquid Water Equivalent (LWE) thickness. The challenge is to forecast these anomalies using exogenous variables over a 5-year period. This repository contains all the necessary code to preprocess the data, engineer features, rank exogenous variables, and generate predictions using state-of-the-art tree-based models such as **CatBoost**.

## Methodology

1. **Data Preprocessing**: 
   - The dataset was provided as a global grid. Given its high resolution, preprocessing steps were necessary to reduce computational complexity. Hydrographic basins were used to group data, and around 170 basins were selected, covering 98% of the Earth's surface.
   - The preprocessing also included windowing the data to generate lag features and future-step targets for each basin.

2. **Feature Engineering**: 
   - Over 1,000 exogenous variables were introduced using data from the Copernicus Climate Data Store. These variables included climate, vegetation, and soil properties.
   - Atmospheric variables from the CMIP6 climate model were incorporated to capture future climate scenarios.

3. **Exogenous Variable Filtering**: 
   - Low-variance variables were removed to focus on the most impactful data.
   - Highly correlated variables (correlation > 0.9) were also discarded to avoid duplication.

4. **Model Training**: 
   - CatBoost was chosen as the primary model due to its superior performance on tabular data and its ease of integration with SHAP values for feature importance analysis.
   - Both multi-output models and independent stepwise models were trained to explore different approaches for predicting the future steps.

5. **Ranking Predictive Variables**: 
   - A custom metric called **Combined Predictive Power Metric (CPPM)** was designed to rank exogenous variables based on feature importance, SHAP values, and Recursive Feature Elimination (RFE).

## Jupyter Notebooks

The repository contains several Jupyter notebooks that cover different stages of the project:

1. **GRACE_Data_Windowing.ipynb**: 
   - This notebook explains the data windowing process, including the creation of lag variables and future-step targets.
   
2. **GRACE_Exogenous_Variable_Ranking.ipynb**: 
   - Here, we apply various techniques to rank the exogenous variables based on their predictive power.

3. **GRACE_Exogenous_Variables.ipynb**: 
   - This notebook demonstrates the extraction and aggregation of exogenous variables from external datasets such as the Copernicus Climate Data Store.

4. **GRACE_exog_vars.ipynb**: 
   - Further exploration and refinement of exogenous variable selection.

## Combined Predictive Power Metric (CPPM) - Definition

Let **$X$** represent the feature matrix and **$y$** the target variable (which can be multiple outputs in the case of multi-step forecasting).

For each feature **$f_i$** in the feature matrix **$X$**, we compute three separate metrics that quantify its predictive power:

1. **Feature Importance ($S_{FI}$)**: Denoted as $S_{FI}(f_i)$, it represents the importance of feature **$f_i$** as determined by the model's internal feature importance mechanism (e.g., CatBoost's built-in feature importance method).

2. **SHAP Values ($S_{SHAP}$)**: Denoted as $S_{SHAP}(f_i)$, these are the SHAP (Shapley Additive Explanations) values representing the average impact of feature **$f_i$** on the model's predictions across different samples.

3. **Recursive Feature Elimination ($S_{RFE}$)**: Denoted as $S_{RFE}(f_i)$, this metric is computed by performing Recursive Feature Elimination (RFE), ranking features based on how much their removal affects the model's performance. To align this with other metrics (higher values = better), we take the **inverse of the ranking** and scale it by the model's performance drop when the feature is removed.

### Normalization:
Each score is normalized to ensure comparability across features. The normalization for each metric $S_{FI}(f_i), S_{SHAP}(f_i), S_{RFE}(f_i)$ is defined as:

$$
\hat{S}(f_i) = \frac{S(f_i) - \min(S)}{\max(S) - \min(S)}
$$

Where $\hat{S}(f_i)$ is the normalized score for feature **$f_i$**.

### Combined Predictive Power Metric:
The CPPM for feature **$f_i$** is computed as a weighted sum of the normalized scores:

$$
CPPM(f_i) = w_1 \hat{S}_ {FI}(f_i) + w_2 \hat{S}_ {SHAP}(f_i) + w_3 \hat{S}_{RFE}(f_i)
$$

Where:
- $w_1, w_2, w_3$ are the weights assigned to the respective metrics (e.g. $w_1 = 0.33, w_2 = 0.34, w_3 = 0.33$ by default).
- $\hat{S}_ {FI}(f_i)$ , $\hat{S}_ {SHAP}(f_i)$ , $\hat{S}_{RFE}(f_i)$ are the normalized scores for **Feature Importance**, **SHAP values**, and **RFE rankings** respectively.

### Final CPPM for Ranking:
For multi-step forecasting, the **CPPM** for each feature is averaged across all 60 steps and across both the independent and multioutput models:

$$CPPM_{\text{final}}(f_i) = \frac{1}{2} \left( \frac{1}{60} \sum_{k=1}^{60} CPPM_k(f_i) + CPPM_{\text{multioutput}}(f_i) \right)$$

Where:
- $CPPM_k(f_i)$ is the **CPPM** for feature **f_i** for step **k**.
- $CPPM_{\text{multioutput}}(f_i)$ is the **CPPM** for feature **f_i** when using the multioutput model.

This final score provides a comprehensive measure of each feature’s predictive power, allowing us to rank the features accordingly.


## Final Ranking of Exogenous Variables

1. **Snow Albedo**
2. **Vertical Transport of Cloud Frozen Water**
3. **Surface Pressure**
4. **Total Column Water Vapour**
5. **Potential Evaporation**
6. **Snow Evaporation**
7. **Total Column Cloud Ice Water**
8. **High Cloud Cover**
9. **Leaf Area Index (LAI) for Low Vegetation**
10. **Soil Temperature and Volumetric Water (Layer 4)**
