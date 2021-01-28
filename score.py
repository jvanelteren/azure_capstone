

import os
import json
import time
import joblib
# import azureml.train.automl
# import azureml.automl.runtime

# Called when the deployed service starts
def init():
    global model
    # global pipe

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

# Handle requests to the service
def run(data):
    try:
        # Pick out the text property of the JSON request.
        # This expects a request in the form of {"text": "some text to score for sentiment"}
        data = json.loads(data)
        # test_x = pipe.transform(data)
        prediction = predict(data)
        #Return prediction
        return prediction
    except Exception as e:
        error = str(e)
        return error



@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})




# Predict sentiment using the model
def predict(data):
    start_at = time.time()
    # Tokenize text
    preds = model.predict(data)
    # Decode sentiment
    return {"predictions": preds, "elapsed_time": time.time()-start_at}  