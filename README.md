# Cape Town Airbnb Listings: Predicting 30-Day Availability

## Project Overview

This project uses statistical learning and machine learning in R to predict whether an Airbnb listing in Cape Town is expected to have at least one night available over the next 30 days.

Short-term rental platforms such as Airbnb create income opportunities for hosts and provide accommodation for visitors. However, they also raise concerns about housing affordability, rental availability, and the potential diversion of residential properties into tourist accommodation. Understanding listing availability can help explore the factors associated with short-term rental occupancy.

The project compares logistic regression, LASSO-regularised logistic regression, random forest, and gradient boosting models. Models are evaluated using 10-fold cross-validation, with log-loss as the primary evaluation metric.

## Project Objective

The objective is to estimate the probability that an Airbnb listing will **not be fully booked over the next 30 days**.

The target variable is `fully_booked_30`:

- `no`: The listing has at least one night available over the next 30 days. This is the positive class for the assignment.
- `yes`: The listing is fully booked over the next 30 days.

The models estimate the probability of the positive class:

\[
\hat{P}(\text{fully\_booked\_30}=\text{no}\mid X)
\]

The analysis uses the supplied 2025 Cape Town Airbnb listing data. The original dataset is treated as the provided modelling dataset rather than manually editing the source spreadsheet.

## Repository Contents

The repository contains the R and Quarto files, datasets, prediction outputs, variable descriptions, and rendered reports used in the project.

| File | Description |
|---|---|
| `airbnb_listings.qmd` | Quarto source document containing the analysis, modelling code, results, and interpretation |
| `listings.csv` | Supplied Cape Town Airbnb listings dataset from 2025 |
| `predictions.csv` | CSV file containing predicted probabilities generated for the prediction task |
| `testing.csv` | Unlabelled test data for which probabilities are predicted |
| `variable_descriptions.csv` | Descriptions of the dataset variables |
| Rendered HTML report | Browser-readable report generated from the Quarto source |
| Supporting files | Figures, intermediate outputs, and other files used in the analysis |

*The filenames above reflect the project files described for this repository. If the actual capitalization or filenames differ, use the exact names shown in GitHub.*

## Methodology

### 1. Introduction and Data Preparation

The project begins with an overview of the prediction problem and a brief exploratory data analysis focused on aspects relevant to listing availability.

Exploration examines variables such as price, review activity, and booking status. The analysis considers how listing characteristics and host-related features may be associated with whether a property is fully booked.

The supplied data are used for modelling, with any feature selection or additional preprocessing documented in the Quarto report.

### 2. Model Training

Four classification approaches are evaluated.

**Logistic regression**

A baseline logistic regression model is fitted using the available explanatory variables. Coefficient estimates and significance tests are used to identify variables associated with the target outcome, conditional on the other predictors.

**LASSO logistic regression**

L1 regularisation is applied to shrink coefficients and select a smaller set of predictors. The regularisation parameter is selected using cross-validation.

**Random forest**

A random forest model is tuned using a grid search over:

- `mtry`: 2, 4, 6, and 8.
- `splitrule`: Gini and Hellinger.
- `min.node.size`: 1, 5, and 10.

This produces 24 hyperparameter combinations, evaluated using 10-fold cross-validation. The number of trees is fixed at 300.

The selected configuration is `mtry = 2`, `splitrule = Hellinger`, and `min.node.size = 1`.

**Gradient boosting machine (GBM)**

A gradient boosting model is tuned over:

- `n.trees`: 100, 200, and 500.
- `interaction.depth`: 1, 3, and 5.
- `shrinkage`: 0.05 and 0.10.
- `n.minobsinnode`: fixed at 10.

The 18 hyperparameter combinations are evaluated using 10-fold cross-validation. The selected configuration is `n.trees = 500`, `interaction.depth = 3`, `shrinkage = 0.05`, and `n.minobsinnode = 10`.

### 3. Model Evaluation and Selection

All models are compared using cross-validation predictions. The evaluation metrics include:

- Log-loss
- Area under the precision-recall curve (AUPRC)
- Recall
- Specificity
- Precision
- F1 score
- Accuracy

The decision threshold for predicted labels is the proportion of training observations belonging to the target class. The report also presents a visual comparison of model performance and a ROC curve for the selected model.

#### Cross-validation results

The following results are reported in the project report.

| Model | Log-loss | AUPRC | Recall | Specificity | Precision | F1 score | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression (GLM) | 0.4434 | 0.9033 | 0.5672 | 0.7088 | 0.8966 | 0.6949 | 0.5932 |
| LASSO | 0.4436 | 0.9032 | 0.5591 | 0.7105 | 0.8958 | 0.6885 | 0.5869 |
| Random forest | 0.4224 | 0.9149 | 0.6594 | 0.6805 | 0.9019 | 0.7618 | 0.6633 |
| GBM | 0.4296 | 0.9107 | 0.7023 | 0.6276 | 0.8936 | 0.7865 | 0.6886 |

*Source: the project's reported 10-fold cross-validation results.*

The random forest achieves the lowest log-loss and highest AUPRC, while GBM achieves the highest recall, F1 score, and accuracy at the chosen decision threshold.

**Selected model: GBM.** The project selects GBM for its stronger balance of recall, F1 score, and accuracy under the specified decision rule. This choice prioritises identifying listings that are not fully booked while retaining high precision. The random forest remains a strong alternative, particularly when probability-based performance is prioritised.

### 4. ROC Curve and Decision Threshold

The ROC curve for the selected GBM model is presented in the report. The operating point is marked using the decision threshold:

\[
\tau = 0.817
\]

At this threshold, the report indicates a recall of approximately 0.70 and specificity of approximately 0.63.

The ROC curve illustrates the trade-off between correctly identifying listings that are not fully booked and incorrectly classifying fully booked listings as available.

### 5. Model Interpretation

The selected GBM model is investigated using variable importance plots and partial dependence plots.

**Variable importance**

The report examines impurity-based and permutation-based importance. Features highlighted in the report include review activity, minimum-night requirements, geographic coordinates, host experience, host acceptance and response rates, and review scores.

Importance measures help identify features used by the fitted model, but they do not establish causal relationships.

**Partial dependence**

The report investigates the modelled relationships between predicted vacancy probability and selected variables:

- **Price:** Higher prices are associated with increased predicted vacancy probability in the reported partial dependence results.
- **Reviews per month:** Greater review activity is associated with increased predicted vacancy probability in the reported analysis.
- **Minimum nights:** Stricter minimum-night requirements initially reduce predicted vacancy probability before the relationship stabilises.
- **Host experience:** Predicted vacancy probability generally declines with host experience, although the relationship is not entirely consistent.

These patterns describe modelled associations in the supplied dataset. They should not be interpreted as evidence that changing a single feature will necessarily cause a change in booking outcomes.

### 6. Prediction

The selected model is used to estimate probabilities for the unlabelled test dataset in `testing.csv`.

The resulting predictions are stored in `predictions.csv`. The assignment requires a single column of numeric probabilities between 0 and 1, with no column header, no blank cells, and rows in the same order as the test observations. The probabilities must represent:

\[
\hat{P}(\text{fully\_booked\_30}=\text{no})
\]

Before submission, the prediction file should be checked against these formatting requirements and renamed to the required student-ID filename if necessary.

## Evaluation Metric

Log-loss, also known as binary cross-entropy, is the primary evaluation metric:

\[
L=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
\]

Here, \(y_i=1\) when `fully_booked_30` is `no`, and \(p_i\) is the predicted probability of that class.

**Lower log-loss is better.** It rewards models that assign accurate probabilities and penalises confident incorrect predictions. Classification metrics such as recall, precision, and accuracy provide complementary information at the selected threshold.

## Tools and Technologies

- **R:** Data preparation, statistical modelling, prediction, and evaluation.
- **RStudio:** Development environment.
- **Quarto:** Reproducible reporting.
- **Logistic regression:** Baseline probabilistic classification.
- **LASSO:** Regularised classification and feature selection.
- **Random forest:** Ensemble tree classification.
- **GBM:** Gradient boosting classification.
- **CSV:** Storage of supplied data and predicted probabilities.
- **PDF:** Rendered project report.

## Reproducibility

To reproduce the project:

1. Clone or download the repository.
2. Open the `.qmd` source file in RStudio.
3. Install the R packages required by the report.
4. Confirm that the CSV files are available at the paths referenced in the source code.
5. Render the Quarto document to regenerate the HTML report.
6. Follow the report's documented cross-validation and modelling procedure.

Reproducibility depends on the package versions, random seeds, cross-validation folds, and preprocessing steps used by the original analysis.

## Limitations

- The analysis uses a curated dataset of Cape Town listings collected in 2025.
- Cross-validation results reflect the supplied data and modelling procedure and may not generalise to future market conditions.
- Observed associations do not establish causation.
- The selected decision threshold affects recall, specificity, precision, F1 score, and accuracy.
- The GBM selection prioritises threshold-dependent classification performance, whereas random forest performs better on the reported log-loss and AUPRC metrics.

## Conclusion

This project compares four classification approaches for predicting whether Cape Town Airbnb listings are expected to have at least one night available over the next 30 days. Ten-fold cross-validation, hyperparameter tuning, probability-based evaluation, classification metrics, ROC analysis, variable importance, and partial dependence are used to assess model performance and interpret predictive patterns.

The reported results identify GBM as the selected model based on recall, F1 score, and accuracy, while random forest provides the best reported log-loss and AUPRC. The final predictions estimate the probability that each unlabelled listing is not fully booked over the next 30 days.

The findings describe associations within the supplied 2025 dataset and should not be interpreted as causal explanations of housing availability or rental-market behaviour.
