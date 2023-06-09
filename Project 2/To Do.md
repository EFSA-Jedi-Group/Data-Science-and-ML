# Meets

| **Location** | **Date** |
| ------------------ | -------------- |
| Google meet        | 29.05          |

| **Agenda**                              | **Participantes**                           |
| --------------------------------------------- | ------------------------------------------------- |
| Ponto situação,<br />Distribuição tarefas | Diogo<br />Gonçalo Eloy <br />Rafa <br />André |

| To do                      | **Deadline** | **Owner** |
| -------------------------- | ------------------ | --------------- |
| Outliers Statistical tests | 02.06              | André          |
| Cross Validation           | 02.06              | Diogo           |
| Standardize Variables      | 02.06              | Rafa            |
| NaN’s - Mode, KNN         | 02.06              | Eloy            |
|                            |                    | Matos           |

**2.** 

Location: Google meets

Date: 09.06.23

| Agenda                 | Participantes                                    |
| ---------------------- | ------------------------------------------------ |
| Distribuição tarefas | Diogo<br />Matos<br />Eloy<br />Rafa<br />André |

| To do                                      | Deadlines | Owner  |
| ------------------------------------------ | --------- | ------ |
| Inputs & Outputs functionsLABs             | 11.06     | André |
| Finalize Data ExplorationPrepare functions | 11.06     | Diogo  |
| Viz + conclusions + add kaggle             | 11.06     | Rafa   |
| Report                                     | 11.06     | Eloy   |
| Report                                     | 11.06     | Matos  |

# To Do

* Data Exploration
  * Duplicates
  * Nan
  * Misclassification
  * Variable Distribution
  * Variable correlation
  * Outliers/ Extreme Values
  * Incongruencies

- Feature Engineering
- Cross Validation

  - K-Fold Cross-Validation
  - Stratified K-Fold Cross-Validation
  - Leave-One-Out Cross-Validation (LOOCV)
- Feature Selection

  - Categorical
  - Numeric

https://practicaldatascience.co.uk/machine-learning/how-to-add-feature-engineering-to-a-scikit-learn-pipeline

# Pipeline

1. **Data Preprocessing** : This step involves preparing the data for training and evaluation. It may include the following sub-steps:

* **Data Cleaning** : Handling missing values, outliers, or noisy data.
* **Feature Selection** : Choosing relevant features that contribute to the prediction task.
* **Feature Scaling** : Scaling numerical features to ensure they have similar ranges.
* **Categorical Encoding** : Transforming categorical variables into numerical representations.
* **Train/Test Split** : Splitting the data into training and testing sets for evaluation.

1. **Model Initialization** : Choose the appropriate machine learning algorithm or model for your task. In scikit-learn, this typically involves importing the appropriate class from the library.
2. **Model Training** : Fit the model to the training data using the `fit()` method of the chosen model class. This step involves finding the best parameters or weights that minimize the error or maximize the model's performance.
3. **Model Evaluation** : Evaluate the trained model's performance on unseen data. This step typically includes the following sub-steps:

* **Prediction** : Use the trained model to predict outcomes for the test set or new data.
* **Evaluation Metrics** : Calculate metrics such as accuracy, precision, recall, or others, depending on the specific problem.
* **Model Selection** : Compare the performance of different models or hyperparameter settings to choose the best-performing one.

1. **Hyperparameter Tuning** : Optimize the model's hyperparameters to improve performance. This can be done using techniques like grid search or randomized search to find the best combination of hyperparameters.
2. **Final Model Training** : Train the chosen model on the entire dataset using the optimized hyperparameters obtained in the previous step. This step ensures that the model is utilizing the maximum available data for training.
3. **Deployment and Prediction** : Once the final model is trained, it can be deployed for making predictions on new, unseen data.
