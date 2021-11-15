from train2 import deap_preprocess, train

datasets, labels = deap_preprocess("s01","arousal")
datasets = datasets[0:720]
labels = labels[0:720]
test_datasets = datasets[720:800]
test_labels = labels[720:800]
train(datasets, labels, 40, True)

from sklearn.metrics import accuracy_score
from ACRNN import ACRNN
import torch
import numpy as np
n_channel=32
model = ACRNN(n_channel)
PATH='./param2/model_40_True.pth'
model.load_state_dict(torch.load(PATH))
map,y_map,output = model(test_datasets)
#print(y_map.shape)
output = output.to('cpu').detach().numpy().copy()
pred_test = np.argmax(output,axis=1)
print(accuracy_score(test_labels,pred_test))