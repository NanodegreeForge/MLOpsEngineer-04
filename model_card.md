# Model Card

## Model Details
The model is a simple Logistic Regression classifier. 

## Intended Use
This model is designed to predict income classes on census data.
-  The target variable is binary: Salary of >50K and <=50K.

## Training Data
The UCI Census Income Data Set was used for training. Detailed information about the dataset is available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income). 

**Training Set:** 80% of the dataset, consisting of 26,048 instances from the total 32,561 rows.

## Evaluation Data
**Testing Set:** 20% of the dataset, consisting of 6,512 instances from the total 32,561 rows.

## Metrics
- **Precision:** 0.7257
- **Recall:** 0.2519
- **F-Beta:** 0.3740

## Ethical Considerations
The dataset consists of publicly available, highly aggregated census data. 
- The Data Set may contain inherent biases related to gender, race, age, or other demographic factors. These biases could lead to unfair predictions and disparities in income classification.

## Caveats and Recommendations
Model performance and evaulation techniques can be improved.
- Perform hyperparameter optimization to improve model performance. 
- Use cross-validation techniques to obtain a more robust estimate of the model's performance.