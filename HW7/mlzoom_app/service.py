import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

# model_runner = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5").to_runner() # model 1
model_runner = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5").to_runner() # model 2

svc = bentoml.Service("mlzoomcamp_cool_svc", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(application_data):
    prediction = model_runner.predict.run(application_data)
    print(prediction)

    return prediction
