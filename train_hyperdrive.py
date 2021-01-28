#%%
import argparse
import os
import numpy as np
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb


# Create TabularDataset using TabularDatasetFactory
train_url = ["https://raw.githubusercontent.com/jvanelteren/housing/master/datasets/housing_after_preprocessing.csv"]
train_x = TabularDatasetFactory.from_delimited_files(train_url).drop_columns('y')
train_y = TabularDatasetFactory.from_delimited_files(train_url).keep_columns('y')

test_url = ["https://raw.githubusercontent.com/jvanelteren/housing/master/datasets/final_test.csv"]
test_x = TabularDatasetFactory.from_delimited_files(test_url)

train_x = train_x.to_pandas_dataframe()
train_y = train_y.to_pandas_dataframe()
test_x = test_x.to_pandas_dataframe()

train_x = train_x.fillna(value=np.nan)
test_x = train_x.fillna(value=np.nan)


# def get_pipeline(impute_cat='DFSIMPLEIMPUTER', impute_num =DFSIMPLEIMPUTER', scale=DFMINMAX',onehot='default',remove_outliers='default'):
class DFGetDummies(TransformerMixin):
    # actually this should be identical to sklearn OneHotEncoder()
    def fit(self, X, y=None):
        self.train = pd.get_dummies(X)
        return self
    def transform(self, X, y=None):
        self.test = pd.get_dummies(X)
        return self.test.reindex(columns=self.train.columns,fill_value=0)
    def __repr__(self):
        return 'DFGetDummies'

class DFSimpleImputer(SimpleImputer):
    # just like SimpleImputer, but retuns a df
    # this approach creates problems with the add_indicator=True, since more columns are returned
    # so don't set add_indicator to True
    def transform(self, X,y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns) 
    def __repr__(self):
        return f'SimpleImputer'

class DFMinMaxScaler(MinMaxScaler):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)
    def __repr__(self):
        return 'DFMinMaxScaler'

class DFColumnTransformer(ColumnTransformer):
    # works only with non-sparse matrices!
    def _hstack(self, Xs):
        Xs = [f for f in Xs]
        cols = [col for f in Xs for col in f.columns]
        df = pd.DataFrame(np.hstack(Xs), columns=cols)
        # print('final shape',df.shape)
        return df.infer_objects()
#%%
def get_pipeline():
    # in essence this splits the input into a categorical pipeline and a numeric pipeline
    # merged with a ColumnTransformer

    cat_steps = []
    cat_steps.append(('impute_cat', DFSimpleImputer(strategy='most_frequent')))
    cat_steps.append(('cat_to_num', DFGetDummies()))
    categorical_transformer = Pipeline(steps=cat_steps)

    num_steps = []
    num_steps.append(('impute_num', DFSimpleImputer(strategy='most_frequent')))
    # num_steps.append(('scale_num', DFMinMaxScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    col_trans = DFColumnTransformer(transformers=[
        ('numeric', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('category', categorical_transformer, make_column_selector(dtype_exclude=np.number)),
        ])

    preprocessor_steps = [('col_trans', col_trans)]
    preprocessor = Pipeline(steps=preprocessor_steps)

    return preprocessor

#%%
pipe = get_pipeline()
train_x = pipe.fit_transform(train_x, train_y)
test_x = pipe.transform(test_x)
param = {}

#%%
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--min_child_weight', type=float, default=1.7817)
    parser.add_argument('--gamma', type=float, default=0.0468)
    parser.add_argument('--subsample', type=float, default=0.5213)
    parser.add_argument('--colsample_bytree', type=float, default=0.75)
    parser.add_argument('--reg_alpha', type=float, default=0.4640)
    parser.add_argument('--reg_lambda', type=float, default=0.8571)
    parser.add_argument('--colsample_bylevel', type=float, default=0.85)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--max_delta_step', type=float, default=1.0)
    parser.add_argument('--n_estimators', type=float, default=2200)

    # args = parser.parse_args()
    param = vars(parser.parse_args())
    res = xgb.cv(param,xgb.DMatrix(train_x, train_y), num_boost_round =200, early_stopping_rounds = 5, nfold=5,seed=112)

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(pipe, 'outputs/preprocess.joblib')
    joblib.dump(param, 'outputs/param.joblib')
    run.log("rmse", res.loc[res.index[-1],'test-rmse-mean'])
    run.log("num_runs:", res.shape[0])

if __name__ == '__main__':
    main()