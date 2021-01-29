# Beat Kaggle housing submission with Azure

For this capstone project, I wanted to find out how good hyperdrive and AutoML are really performing. As a benchmark I've taken the Kaggle housing competition, a competition where I participated myself and spend quite some time. The goal of the competition is to predict the price of a house given certain features. The benchmark is my personal submission, which scores top 3%. If either the AutoML or Hyperdrive model is outperforming that, it indicated Azure is a very good platform for good machine learning models.

## Project Set Up and Installation
Since I wanted a level playing field between myself and Azure, I decided to take the dataset after feature engineering & selection as input for Azure. This because feature engineering is a critical step in the ML process and I was pretty sure Azure could not come up with a good score without it. It could be an interesting extension of the project to use the Kaggle dataset and submit this to AutoML.

## Dataset
The dataset was prepared by me by running the feature engineering notebook in my competition repo. It generated a [CSV file](https://raw.githubusercontent.com/jvanelteren/housing/master/datasets/housing_after_preprocessing.csv). This CSV file was then imported using the Azure Python SDK (udacity-project.ipynb). The dataset was split in a train dataset including the variable of interest y and a test dataset where the target variable was not present.

### Overview
I've split the main steps into separate notebooks
1) train_models.ipynb registers the dataset and trains automl and hyperdrive models
2) register_model.ipynb has the overview of the model scores and registers a model in the workspace
3) deploy_test_endpoint deploys the best scoring model and tests it's endpoint

### Task
As discussed, this dataset is already enriched by all kinds of feature engineering, e.g. combining existing features into sums, transforming features into ordinal variables, taking log transforms etc. In addition a custom Boruta algorithm was executed to take a subset of features. Note: for the AutoML function, both of these steps were not necessary since Azure does it itself. But I wanted to use the same dataset for both Hyperdrive and AutoML. No scaling of features did take place yet.

### Access
The train CSV file was converted into a TabularDataset and registered in the workspace.

## Automated ML
The automl settings used were:
- experiment_timeout_minutes=60 --> run the experiment for 1 hour
- enable_onnx_compatible_models=True --> to potentially deploy an onnx model at a later stage, e.g. for deployment on a different platform
- task='regression' --> this is a regression tast
- primary_metric='normalized_root_mean_squared_error' --> this comes closest to RMSE. The normalization just divided the RMSE by the range of the target. For the results we will use the RSME, since it is listed under the AutoML results.
- training_data= train --> the train dataset
- compute_target=cpu_cluster --> the compute cluster used
- label_column_name='y'--> the name of our target variable, in this case the log of the house price
- n_cross_validations=5 --> cross validation was used for both hyperdrive as automl to set an equal playing field for both experiments

### Results
AutoML scored a RSME of 0.117. Using a voting ensemble. This was more or less expected, since as a rule of thumb an ensemble of models gives a slight improvement in score. It could be further improved by e.g. getting more data, additional feature engineering (although this dataset is already pretty much engineered out) and further finetuning the model. An other alternative is to try deep learning, but I noticed Azure doesn't offer that option for regression.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
The question for hyperparameter tuning is of course which model to choose. Since I already ran AutoML, it made sense to me to try to improve that result even further with Hyperdrive. Seeing the top results from AutoML, I had no clarity on the contents of the top scoring method the VotingEnsemble, so decided to take the second best model (XGBoost) and try to improve that further with Hyperdrive. I used all the parameters that I've also used in my previous Kaggle submission to define the hyperparameter search space. The ranges where emperically determined by much googling for other ranges used for XGBoost.
After my first run with Hyperdrive, I inspected the parallel plot for the best results and increase the ranges somewhat when the best runs where close to the edges of the search space.

### Results and comparison with AutoML
AutoML on same dataset as hyperdrive	                    0.1167 (best run: VotingEnsemble)
AutoML on original dataset (except log transform target)	0.1248
Hyperdrive xgboost with bandit policy	                    0.1259
Hyperdrive xgboost with bayesian sampling	                0.1270
AutoML on same dataset as hyperdrive	                    0.1273 (second best run: XGBoost)

The both hyperdrive runs were quite close and did both improve the score of the AutoML XGBoost model. However, overall two AutoML Voting ensemble model performed best . I was really surprised to see the AutoML model on the original dataset perform so well! Apparently there is some good feature engineering happening in the AutoML pipeline.

The parameters of the best Hyperdrive model where:
--colsample_bylevel 0.8850097967656851 --colsample_bytree 0.6144104460255237 --gamma 0.0031595294406882857 --learning_rate 0.11537429336238596 --max_delta_step 9.453102902399772 --max_depth 4 --min_child_weight 3 --n_estimators 1439 --reg_lambda 0.5624749925551896 --subsample 0.7770671217398316

This score could be futher improved by making it into an ensemble with other models, or trying out different scaling options.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
The AutoML model had the best score, so that was deployed. It can be queried using a rest endpoint. For this project I've tried out batch inference, to run inference on many houses at once. This feature enabled me to fill in the final test dataset and submit it to Kaggle. The final score of the AutoML model on the private testset was 

*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
The model can be queried my loading a dataset with houses into pandas. Of course it needs to have the same columns the model was trained on. This dataset should then be converted into JSON and submitted to the HTTP endpoint. The endpoint converts this json back into a dataframe and submits to the model. The model returns a Numpy array (for all houses 1 prediction). Then, these predictions are converted into a list, converted to json and returned to the sender.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
