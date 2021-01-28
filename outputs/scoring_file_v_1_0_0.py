# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Column1": pd.Series([0], dtype="int64"), "Total_sqr_footage": pd.Series([0], dtype="int64"), "GarageFinish": pd.Series(["example_value"], dtype="object"), "OverallCond": pd.Series([0], dtype="int64"), "AllPorchSF": pd.Series([0], dtype="int64"), "Condition1": pd.Series(["example_value"], dtype="object"), "OverallQual_3": pd.Series([0], dtype="int64"), "AllSF": pd.Series([0], dtype="int64"), "Neighborhood": pd.Series(["example_value"], dtype="object"), "SimplOverallCond": pd.Series([0], dtype="int64"), "TotalSF": pd.Series([0], dtype="int64"), "OverallQual_sq": pd.Series([0.0], dtype="float64"), "AllFlrsSF_3": pd.Series([0], dtype="int64"), "BsmtExposure": pd.Series([0.0], dtype="float64"), "GarageYrBlt": pd.Series([0.0], dtype="float64"), "MSZoning": pd.Series(["example_value"], dtype="object"), "CentralAir": pd.Series([False], dtype="bool"), "YearBuilt": pd.Series([0], dtype="int64"), "OverallQual": pd.Series([0], dtype="int64"), "BsmtFinSF1": pd.Series([0], dtype="int64"), "AllSF_3": pd.Series([0], dtype="int64"), "OverallGrade": pd.Series([0], dtype="int64"), "LotArea_log": pd.Series([0.0], dtype="float64"), "LotArea": pd.Series([0], dtype="int64"), "SaleCondition": pd.Series(["example_value"], dtype="object"), "AllSF_2": pd.Series([0], dtype="int64"), "Functional": pd.Series([0], dtype="int64"), "TotalSF_log": pd.Series([0.0], dtype="float64"), "YrBltAndRemod": pd.Series([0], dtype="int64"), "SimplOverallQual_sq": pd.Series([0.0], dtype="float64"), "GarageScore_3": pd.Series([0.0], dtype="float64"), "SimplFireplaceQu": pd.Series([0.0], dtype="float64"), "GarageCars_log": pd.Series([0.0], dtype="float64"), "KitchenScore": pd.Series([0], dtype="int64"), "AllFlrsSF_sq": pd.Series([0.0], dtype="float64"), "1stFlrSF_log": pd.Series([0.0], dtype="float64"), "HalfBath": pd.Series([0], dtype="int64"), "Foundation": pd.Series(["example_value"], dtype="object"), "KitchenQual_sq": pd.Series([0.0], dtype="float64"), "FireplaceQu": pd.Series([0.0], dtype="float64"), "KitchenQual_3": pd.Series([0], dtype="int64"), "GarageQual": pd.Series([0.0], dtype="float64"), "GrLivArea": pd.Series([0], dtype="int64"), "GarageScore": pd.Series([0.0], dtype="float64"), "TotRmsAbvGrd_log": pd.Series([0.0], dtype="float64"), "TotalBsmtSF": pd.Series([0], dtype="int64"), "YearRemodAdd_log": pd.Series([0.0], dtype="float64"), "FullBath": pd.Series([0], dtype="int64"), "SimplOverallQual": pd.Series([0], dtype="int64"), "ExterGrade": pd.Series([0], dtype="int64"), "TotalBath_sq": pd.Series([0.0], dtype="float64"), "HalfBath_log": pd.Series([0.0], dtype="float64"), "2ndFlrSF_log": pd.Series([0.0], dtype="float64"), "SimplExterGrade": pd.Series([0], dtype="int64"), "GrLivArea_sq": pd.Series([0.0], dtype="float64"), "OpenPorchSF_log": pd.Series([0.0], dtype="float64"), "AllFlrsSF_2": pd.Series([0], dtype="int64"), "SimplGarageScore": pd.Series([0.0], dtype="float64"), "YearRemodAdd": pd.Series([0], dtype="int64"), "SimplBsmtFinType1": pd.Series([0.0], dtype="float64"), "SaleType": pd.Series(["example_value"], dtype="object"), "BldgType": pd.Series(["example_value"], dtype="object"), "TotRmsAbvGrd": pd.Series([0], dtype="int64"), "SimplOverallGrade": pd.Series([0], dtype="int64"), "SimplKitchenScore": pd.Series([0], dtype="int64"), "PavedDrive": pd.Series([0], dtype="int64"), "2ndFlrSF": pd.Series([0], dtype="int64"), "GarageCars_sq": pd.Series([0.0], dtype="float64"), "LotFrontage": pd.Series([0.0], dtype="float64"), "LotFrontage_log": pd.Series([0.0], dtype="float64"), "SimplExterQual": pd.Series([0], dtype="int64"), "GarageCars": pd.Series([0], dtype="int64"), "GarageArea": pd.Series([0], dtype="int64"), "SimplGarageCond": pd.Series([0.0], dtype="float64"), "Fireplaces": pd.Series([0], dtype="int64"), "HeatingQC": pd.Series([0], dtype="int64"), "GarageCond": pd.Series([0.0], dtype="float64"), "SimplOverallQual_2": pd.Series([0], dtype="int64"), "GrLivArea_2": pd.Series([0], dtype="int64"), "BsmtUnfSF": pd.Series([0], dtype="int64"), "ExterQual": pd.Series([0], dtype="int64"), "OverallQual_2": pd.Series([0], dtype="int64"), "OpenPorchSF": pd.Series([0], dtype="int64"), "LotShape": pd.Series([0], dtype="int64"), "GarageScore_2": pd.Series([0.0], dtype="float64"), "GrLivArea_3": pd.Series([0], dtype="int64"), "FireplaceScore": pd.Series([0.0], dtype="float64"), "BsmtFullBath": pd.Series([0], dtype="int64"), "BsmtFinType1": pd.Series([0.0], dtype="float64"), "SimplGarageQual": pd.Series([0.0], dtype="float64"), "GarageScore_sq": pd.Series([0.0], dtype="float64"), "SimplFireplaceScore": pd.Series([0.0], dtype="float64"), "KitchenQual_2": pd.Series([0], dtype="int64"), "BsmtUnfSF_log": pd.Series([0.0], dtype="float64"), "BedroomAbvGr": pd.Series([0], dtype="int64"), "TotalBath_3": pd.Series([0.0], dtype="float64"), "hasfireplace": pd.Series([0], dtype="int64"), "MoSold": pd.Series([0], dtype="int64"), "ExterQual_3": pd.Series([0], dtype="int64"), "SimplHeatingQC": pd.Series([0], dtype="int64"), "SimplBsmtQual": pd.Series([0.0], dtype="float64"), "AllSF_sq": pd.Series([0.0], dtype="float64"), "BsmtQual": pd.Series([0.0], dtype="float64"), "YrSold": pd.Series([0], dtype="int64"), "BsmtFinSF1_log": pd.Series([0.0], dtype="float64"), "TotalBath_2": pd.Series([0.0], dtype="float64"), "ScreenPorch_log": pd.Series([0.0], dtype="float64"), "hasgarage": pd.Series([0], dtype="int64"), "MasVnrArea": pd.Series([0.0], dtype="float64"), "GarageArea_log": pd.Series([0.0], dtype="float64"), "WoodDeckSF_log": pd.Series([0.0], dtype="float64"), "GarageType": pd.Series(["example_value"], dtype="object"), "TotalBsmtSF_log": pd.Series([0.0], dtype="float64"), "ExterQual_sq": pd.Series([0.0], dtype="float64"), "TotalBath": pd.Series([0.0], dtype="float64"), "KitchenQual": pd.Series([0], dtype="int64"), "GarageCars_3": pd.Series([0], dtype="int64"), "SimplFunctional": pd.Series([0], dtype="int64"), "AllFlrsSF": pd.Series([0], dtype="int64"), "1stFlrSF": pd.Series([0], dtype="int64"), "ScreenPorch": pd.Series([0], dtype="int64"), "GrLivArea_log": pd.Series([0.0], dtype="float64"), "GarageGrade": pd.Series([0.0], dtype="float64"), "Fireplaces_log": pd.Series([0.0], dtype="float64"), "BsmtFullBath_log": pd.Series([0.0], dtype="float64"), "ExterQual_2": pd.Series([0], dtype="int64"), "SimplOverallQual_3": pd.Series([0], dtype="int64"), "GarageCars_2": pd.Series([0], dtype="int64"), "SimplKitchenQual": pd.Series([0], dtype="int64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
