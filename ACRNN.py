import numpy as np
import pandas as pd
import pickle
import torch
from module.channel_wise_attention import channel_wise_attention
from module.CNN import CNN
from module.LSTM import LSTM

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
    # print(labels.shape)
    # print(datasets.shape)
    labels = np.asarray(pd.get_dummies(labels),dtype=np.int8)
    # print(labels.shape)
    datasets = datasets.reshape(-1,384,32,1).astype('float32')
    labels = labels.astype('float32')
    return datasets, labels

# EEG sample
window_size = 384
n_channel = 32
# input
input_channel_num = 1
input_height = 32
input_width = 384
num_labels = 2

# channel wise attention
k = 15
## model
channel_wise_attention = channel_wise_attention(input_channel_num,input_width,input_height,k)
print(channel_wise_attention)

# CNN
## conv
kernel_height = 32
kernel_width = 45
kernel_stride = 1
conv_channel_num = 40
## pooling
pooling_height = 1
pooling_width = 75
pooling_stride = 10
## model
cnn = CNN(input_channel_num,input_height,input_width,kernel_height,kernel_width,kernel_stride,pooling_height,pooling_width,pooling_stride,conv_channel_num)
print(cnn)

# LSTM
n_hidden_state = 64
lstm = LSTM(n_hidden_state)
print(LSTM)
# dataset_deap
deap_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11','s12', 's13','s14','s15', 's16', 's17','s18', 's19', 's20',
 's21', 's22', 's23', 's24', 's25', 's26','s27', 's28', 's29', 's30', 's31', 's32']
# dataset load
datasets, labels = deap_preprocess("s01","arousal")
datasets = torch.from_numpy(datasets).clone()
labels = torch.from_numpy(labels).clone()
# data Hight Windowsize Channel 
datasets = datasets.permute(0,3,1,2)

map, a = channel_wise_attention(datasets)
print(map.shape)
print(a.shape)
b = cnn(a)
print(b.shape)
h,c = lstm(b)
print(h.shape)
print(c.shape)