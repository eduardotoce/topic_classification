import sys
from core.model.train import train
from core.model.predict import predict

print(sys.argv[1] )
if sys.argv[1] == "train":
    train()

if sys.argv[1] == "predict":
    predict(model_name='model_best')

