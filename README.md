# Beat Kaggle housing submission with Azure

For this capstone project, I wanted to find out how good hyperdrive and AutoML are really performing. As a benchmark I've taken the Kaggle housing competition, a competition where I participated myself and spend quite some time. The goal of the competition is to predict the price of a house given certain features. The benchmark is my personal submission, which scores top 3%. If either the AutoML or Hyperdrive model is outperforming that, it indicated Azure is a very good platform for good machine learning models.

## Project Set Up and Installation
Since I wanted a level playing field between myself and Azure, I decided to take the dataset after feature engineering & selection as input for Azure. This because feature engineering is a critical step in the ML process and I was pretty sure Azure could not come up with a good score without it. It could be an interesting extension of the project to use the Kaggle dataset and submit this to AutoML.

## Dataset
The dataset was prepared by me by running the feature engineering notebook in my competition repo. It generated a [CSV file](https://raw.githubusercontent.com/jvanelteren/housing/master/datasets/housing_after_preprocessing.csv). This CSV file was then imported using the Azure Python SDK (udacity-project.ipynb). The dataset was split in a train dataset including the variable of interest y and a test dataset where the target variable was not present.

### Overview

### Task
As discussed, this dataset is already enriched by all kinds of feature engineering, e.g. combining existing features into sums, transforming features into ordinal variables, taking log transforms etc. In addition a custom Boruta algorithm was executed to take a subset of features. Note: for the AutoML function, both of these steps were not necessary since Azure does it itself. But I wanted to use the same dataset for both Hyperdrive and AutoML. No scaling of features did take place yet.

### Access
The CSV file was converted into a TabularDataset and registered in the workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
The question for hyperparameter tuning is of course which model to choose. Since I already ran AutoML, it made sense to me to try to improve that result even further with Hyperdrive. Seeing the top results from AutoML, I had no clarity on the contents of the top scoring method the VotingEnsemble, so decided to take the second best model (XGBoost) and try to improve that further with Hyperdrive. I used all the parameters that I've also used in my previous Kaggle submission to define the hyperparameter search space. The ranges where emperically determined by much googling for other ranges used for XGBoost.
After my first run with Hyperdrive, I inspected the parallel plot for the best results and increase the ranges somewhat when the best runs where close to the edges of the search space.

### Results
The hyperdrive results are not directly comparable with AutoML, since AutoML uses normalized RMSE and hyperdrive only RSME. But calculating the range of the target variable and multiplying that range with the normalized RMSE, it was still possile to compare.
AutoML Voting Ensemble RMSE:
AutoML XGBoost:
Hyperdrive XGBoost:

Parameters of the best Hyperdrive model where:
*TODO*

This score could be futher improved by making it into an ensemble with other models, or trying out different scaling

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
The AutoML model had the best score, so that was deployed. It can be queried using a rest endpoint. For this project I've tried out batch inference, to fill the test file with predictions and submit that to Kaggle.
The final score of the AutoML model was *TODO*

*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.