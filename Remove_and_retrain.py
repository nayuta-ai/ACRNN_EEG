import numpy as np
import pickle
import torch
from ACRNN import ACRNN
from train import space_roar, rank_channel
from sklearn.metrics import accuracy_score

def deap_preprocess(data_file,emotion):
    dataset_extention = ".mat_win_384_dataset.pkl"
    label_extention = ".mat_win_384_label.pkl"
    arousal_or_valence = emotion
    dataset_dir = "../DEAP_pickle_"+arousal_or_valence+"/"
    with open(dataset_dir+data_file+dataset_extention, "rb") as fp:
        datasets = pickle.load(fp)
    with open(dataset_dir+data_file+label_extention,"rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)
    # print(labels)
    # print(datasets.shape)
    # labels = np.asarray(pd.get_dummies(labels),dtype=np.int8)
    # print(labels.shape)
    datasets = datasets.reshape(-1,384,32,1).astype('float32')
    labels = labels.astype('int64')
    # print(type(labels))
    return datasets, labels

datasets, labels = deap_preprocess("s01","arousal")
datasets = torch.from_numpy(datasets).clone()
n_channel=32
model = ACRNN(n_channel)
PATH='./param/model.pth'
model.load_state_dict(torch.load(PATH))
datasets_roar = datasets
datasets = datasets.permute(0,3,1,2)
map,_,output = model(datasets)
output = output.to('cpu').detach().numpy().copy()
pred_test = np.argmax(output,axis=1)
print(accuracy_score(labels,pred_test))
rank = rank_channel(map)
# list is index
datasets_roar = space_roar(datasets_roar,rank[:1])
_,_,output_roar = model(datasets_roar)
#print(y_map.shape)
output_roar = output_roar.to('cpu').detach().numpy().copy()
pred_test = np.argmax(output_roar,axis=1)
print(accuracy_score(labels,pred_test))