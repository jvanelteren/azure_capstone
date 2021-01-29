#%%
import os
import json
import time
import joblib
import pandas as pd
import packaging
# import azureml.train.automl
import azureml.automl.runtime

# Called when the deployed service starts
def init():
    global model
    # global pipe
    try:
        # Get the path where the deployed model can be found.
        print('hi')
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'outputs','model.pkl')
        print('ok')
        print(model_path)
        # print("Found model:", os.path.isfile(model_path)) # To inform us in logs whether the path was indeed found!
        # load models
        model = joblib.load(model_path)
        print('ok2')
        # pipe = joblib.load(model_path + '/pipe.pkl')
    except Exception as e:
        print(e)



# Predict sentiment using the model
def predict(data):
    start_at = time.time()
    # Tokenize text
    preds = model.predict(data)
    # Decode sentiment
    return {"predictions": preds.tolist(), "elapsed_time": time.time()-start_at}  


# Handle requests to the service
def run(data):
    print('new request')
    try:
        # Pick out the text property of the JSON request.
        # This expects a request in the form of {"text": "some text to score for sentiment"}
        data = json.loads(data)
        data = pd.read_json(data)

        # test_x = pipe.transform(data)
        prediction = predict(data)
        #Return prediction
        return prediction
    except Exception as e:
        error = str(e)
        return error


# import json
# import logging
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import joblib

# from azureml.core.model import Model

# from inference_schema.schema_decorators import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
# from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

# input_sample = pd.DataFrame({"age": pd.Series([0.0], dtype="float64"), "anaemia": pd.Series([0.0], dtype="float64"), "creatinine_phosphokinase": pd.Series([0.0], dtype="float64"), "diabetes": pd.Series([0.0], dtype="float64"), "ejection_fraction": pd.Series([0.0], dtype="float64"), "high_blood_pressure": pd.Series([0.0], dtype="float64"), "platelets": pd.Series([0.0], dtype="float64"), "serum_creatinine": pd.Series([0.0], dtype="float64"), "serum_sodium": pd.Series([0.0], dtype="float64"), "sex": pd.Series([0.0], dtype="float64"), "smoking": pd.Series([0.0], dtype="float64"), "time": pd.Series([0.0], dtype="float64")})
# output_sample = np.array([0])


# # -

# def init():
#     global model
#     # This name is model.id of model that we want to deploy deserialize the model file back
#     # into a sklearn model
#     model_path = Model.get_model_path("heart-failure-prediction-automl-model")
#     print(model_path)
#     path = os.path.normpath(model_path)
#     path_split = path.split(os.sep)
#     try:
#         model = joblib.load(model_path)
#     except Exception as e:
#         raise

# @input_schema('data', PandasParameterType(input_sample))
# @output_schema(NumpyParameterType(output_sample))
# def run(data):
#     try:
#         result = model.predict(data)
#         return json.dumps({"result": result.tolist()})
#     except Exception as e:
#         result = str(e)
#         return json.dumps({"error": result})
