import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from scipy.stats import rankdata
from ACRNN import ACRNN,A_CRNN,CRNN_A

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
datasets = datasets.permute(0,3,1,2)

n_channel=32
model = ACRNN(n_channel)
PATH='./param/model.pth'
model.load_state_dict(torch.load(PATH))
model_A=CRNN_A(n_channel)
A_model=A_CRNN(n_channel)
model_A.state_dict()['cnn.conv.0.weight'][0:40]=model.state_dict()['cnn.conv.0.weight']
model_A.state_dict()['cnn.conv.0.bias'][0:40]=model.state_dict()['cnn.conv.0.bias']
model_A.state_dict()['lstm.lstm.weight_ih_l0'][0:256]=model.state_dict()['lstm.lstm.weight_ih_l0']
model_A.state_dict()['lstm.lstm.weight_hh_l0'][0:256]=model.state_dict()['lstm.lstm.weight_hh_l0']
model_A.state_dict()['lstm.lstm.bias_ih_l0'][0:256]=model.state_dict()['lstm.lstm.bias_ih_l0']
model_A.state_dict()['lstm.lstm.bias_hh_l0'][0:256]=model.state_dict()['lstm.lstm.bias_hh_l0']
model_A.state_dict()['lstm.lstm.weight_ih_l1'][0:256]=model.state_dict()['lstm.lstm.weight_ih_l1']
model_A.state_dict()['lstm.lstm.weight_hh_l1'][0:256]=model.state_dict()['lstm.lstm.weight_hh_l1']
model_A.state_dict()['lstm.lstm.bias_ih_l1'][0:256]=model.state_dict()['lstm.lstm.bias_ih_l1']
model_A.state_dict()['lstm.lstm.bias_hh_l1'][0:256]=model.state_dict()['lstm.lstm.bias_hh_l1']
model_A.state_dict()['self_attention.dense.W1'][0:64]=model.state_dict()['self_attention.dense.W1']
model_A.state_dict()['self_attention.dense.W2'][0:64]=model.state_dict()['self_attention.dense.W2']
model_A.state_dict()['self_attention.dense.b'][0:64]=model.state_dict()['self_attention.dense.b']
model_A.state_dict()['self_attention.dense.vector.weight'][0:64]=model.state_dict()['self_attention.dense.vector.weight']
model_A.state_dict()['self_attention.dense.vector.bias'][0:64]=model.state_dict()['self_attention.dense.vector.bias']
model_A.state_dict()['self_attention.self_attention.1.weight'][0:64]=model.state_dict()['self_attention.self_attention.1.weight']
model_A.state_dict()['self_attention.self_attention.1.bias'][0:64]=model.state_dict()['self_attention.self_attention.1.bias']
model_A.state_dict()['softmax.0.weight'][0:2]=model.state_dict()['softmax.0.weight']
model_A.state_dict()['softmax.0.bias'][0:2]=model.state_dict()['softmax.0.bias']
A_model.state_dict()['channel_wise_attention.fc.0.weight'][0:15]=model.state_dict()['channel_wise_attention.fc.0.weight']
A_model.state_dict()['channel_wise_attention.fc.0.bias'][0:15]=model.state_dict()['channel_wise_attention.fc.0.bias']
A_model.state_dict()['channel_wise_attention.fc.2.weight'][0:32]=model.state_dict()['channel_wise_attention.fc.2.weight']
A_model.state_dict()['channel_wise_attention.fc.2.bias'][0:32]=model.state_dict()['channel_wise_attention.fc.2.bias']
A_model.state_dict()['cnn.conv.0.weight'][0:40]=model.state_dict()['cnn.conv.0.weight']
A_model.state_dict()['cnn.conv.0.bias'][0:40]=model.state_dict()['cnn.conv.0.bias']
A_model.state_dict()['lstm.lstm.weight_ih_l0'][0:256]=model.state_dict()['lstm.lstm.weight_ih_l0']
A_model.state_dict()['lstm.lstm.weight_hh_l0'][0:256]=model.state_dict()['lstm.lstm.weight_hh_l0']
A_model.state_dict()['lstm.lstm.bias_ih_l0'][0:256]=model.state_dict()['lstm.lstm.bias_ih_l0']
A_model.state_dict()['lstm.lstm.bias_hh_l0'][0:256]=model.state_dict()['lstm.lstm.bias_hh_l0']
A_model.state_dict()['lstm.lstm.weight_ih_l1'][0:256]=model.state_dict()['lstm.lstm.weight_ih_l1']
A_model.state_dict()['lstm.lstm.weight_hh_l1'][0:256]=model.state_dict()['lstm.lstm.weight_hh_l1']
A_model.state_dict()['lstm.lstm.bias_ih_l1'][0:256]=model.state_dict()['lstm.lstm.bias_ih_l1']
A_model.state_dict()['lstm.lstm.bias_hh_l1'][0:256]=model.state_dict()['lstm.lstm.bias_hh_l1']
A_model.state_dict()['softmax.0.weight'][0:2]=model.state_dict()['softmax.0.weight']
A_model.state_dict()['softmax.0.bias'][0:2]=model.state_dict()['softmax.0.bias']