## Combined Predictive Power Metric (CPPM) - Mathematical Definition

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
