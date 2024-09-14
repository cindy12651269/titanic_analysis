## Titanic Survival Analysis

This project aims to predict the survival of passengers on the Titanic based on their demographic and ticket-related features. The dataset used is the famous **Titanic** dataset from Kaggle, which includes information such as age, sex, ticket class, and other details. The goal is to apply machine learning techniques to understand which factors had the greatest influence on passenger survival and to build a model that can predict survival outcomes.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Titanic dataset provides a great opportunity to apply machine learning techniques to a real-world classification problem. This project explores various machine learning models to predict the survival of Titanic passengers based on features such as age, sex, class, and fare. The main objectives are:

- To perform data cleaning and preprocessing.
- To explore relationships between features and survival rates.
- To apply feature engineering to enhance the model’s predictive capabilities.
- To train multiple machine learning models and select the one with the best performance.

## Dataset

The dataset is sourced from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic). It consists of the following key features:

- **PassengerId**: Unique identifier for each passenger.
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard the Titanic.
- **Parch**: Number of parents/children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Passenger fare.
- **Cabin**: Cabin number (if available).
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
- **Survived**: Survival (0 = No, 1 = Yes) — this is the target variable.

## Data Preprocessing

Data preprocessing is an essential step before training the model. It includes:

- **Handling Missing Values**: The dataset contains missing values in columns such as `Age`, `Cabin`, and `Embarked`. We fill in missing values appropriately:
  - **Age**: Missing ages are imputed using the median value or by grouping passengers with similar characteristics (e.g., by class or gender).
  - **Cabin**: Since many cabin values are missing, we either discard this feature or categorize it into groups based on available cabin letters.
  - **Embarked**: Missing values in the `Embarked` column are filled with the mode of the column, which is 'S' (Southampton).
- **Categorical Encoding**: We convert categorical variables such as `Sex`, `Embarked`, and `Pclass` into numerical representations using one-hot encoding to make them usable for machine learning models.
- **Feature Scaling**: Continuous variables such as `Fare` and `Age` are normalized or standardized to ensure that they are on the same scale for models like logistic regression and SVM.

## Exploratory Data Analysis (EDA)

Before building machine learning models, we perform EDA to better understand the data and the relationships between the features. Some key steps in EDA include:

- **Survival Rates by Gender**: We analyze the survival rates between males and females, showing that females had a higher survival rate.
- **Survival Rates by Class**: We explore how ticket class (1st, 2nd, 3rd) affects survival, showing that passengers in 1st class had a significantly higher survival rate.
- **Age Distribution**: We create histograms to visualize the age distribution of passengers, showing which age groups had higher chances of survival.
- **Correlation Matrix**: A heatmap is used to display the correlations between numerical features and the target variable, helping to identify which features are strongly related to survival.

## Feature Engineering

Feature engineering involves creating new features or modifying existing ones to improve the model’s predictive power. Some of the key feature engineering steps in this project include:

- **Creating Family Size**: We combine `SibSp` and `Parch` to create a new feature `FamilySize`, representing the total number of family members aboard. We hypothesize that passengers with larger families may have had different survival chances.
- **Title Extraction from Name**: Titles such as "Mr.", "Mrs.", "Miss.", etc., are extracted from the `Name` column and used as a new feature, as titles may provide insight into social status or age group.
- **Fare Binning**: We group passengers based on the range of their `Fare` to reduce noise in the data and potentially improve model performance.

## Model Selection

We experiment with multiple machine learning models to predict passenger survival. These models include:

- **Logistic Regression**: A simple but effective linear model for binary classification.
- **Decision Tree**: A model that splits the data based on features, making predictions by following decision rules.
- **Random Forest**: An ensemble method that combines multiple decision trees to reduce overfitting and improve generalization.
- **Support Vector Machines (SVM)**: A powerful classifier that finds the optimal hyperplane to separate the classes.
- **Gradient Boosting**: Another ensemble technique that builds models sequentially, with each new model correcting the errors of the previous one.

To ensure optimal performance, hyperparameters for each model are fine-tuned using cross-validation techniques like **GridSearchCV**.

## Results

We evaluate the models using several metrics, including:

- **Accuracy**: The overall correctness of the model.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positives that were correctly identified.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.

After evaluating the models, the best-performing model achieved an accuracy of approximately **X%** (replace with actual result after evaluation) on the test set. The confusion matrix and other evaluation metrics are visualized in the notebook.

## Usage

1. **Data Loading**: The dataset is loaded and preprocessed to handle missing values, categorical variables, and scaling.
2. **Model Training**: Various machine learning models are trained on the processed dataset.
3. **Model Evaluation**: The models are evaluated on the test data, and the one with the best performance is selected for final predictions.
4. **Predictions**: The trained model can be used to predict survival on unseen data.

To run the project:

- Open the Jupyter notebook `titanic_analysis.ipynb`.
- Follow the cells to preprocess data, train models, and evaluate performance.

## Contributing

Contributions to this project are welcome. You can fork the repository, make improvements or suggestions, and submit a pull request. We appreciate any feedback or improvements to the model performance.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
