import numpy as np
import pickle
import torch
from ACRNN import ACRNN
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
from graph import acc_graph

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
print(datasets.shape)
map,_,output = model(datasets)
output = output.to('cpu').detach().numpy().copy()
pred_test = np.argmax(output,axis=1)
print(accuracy_score(labels,pred_test))
data_origin = datasets.permute(3,0,1,2)

def data_32_mean(data_origin, rank = []):
    data = torch.randn(32,32,800,1,384)
    data_mean = data_origin.mean()
    if rank == []:
        for i in range(32):
            data_i = data_origin.clone()
            data_i[i] = data_mean
            data[i] = data_i
    else:
        for i in rank:
            data_i = data_origin
            data_i[i-1] = data_mean
            data[i-1] = data_i
    data = data.permute(0,2,3,4,1)
    return data

def data_32_zero(data_origin, rank = []):
    data = torch.randn(32,32,800,1,384)
    data_zero = torch.zeros(800,1,384)
    if rank == []:
        for i in range(32):
            data_i = data_origin.clone()
            data_i[i] = data_zero
            data[i] = data_i
    else:
        for i in rank:
            data_i = data_origin
            data_i[i-1] = data_zero
            data[i-1] = data_i
    data = data.permute(0,2,3,4,1)
    return data

def colleration_coefficinet(labels, map):
    labels_mean = labels.mean()
    map_mean = map.mean()
    s_x = 0
    s_y = 0
    s_xy = 0
    for i in range(len(labels)):
        s_x += (labels[i] - labels_mean) ** 2
        s_y += (map[i] - map_mean) ** 2
        s_xy += (labels[i] - labels_mean) * (map[i] - map_mean)
    s_xy = s_xy / len(labels)
    s_x = pow(s_x / len(labels), 0.5)
    s_y = pow(s_y / len(labels), 0.5)
    return s_xy / (s_x * s_y)

def rank_col(data, reverse = False):
    data = data.to('cpu').detach().numpy().copy()
    rank_list = rankdata(data).astype(int)
    if reverse:
        np.flipud(rank_list)
    return rank_list

col_list = rank_col(colleration_coefficinet(labels,map))
col_list_reverse = rank_col(colleration_coefficinet(labels,map), reverse=True)
print(col_list)
data_origin1 = data_origin.clone()
data_origin2 = data_origin.clone()
data_loss_mean = data_32_mean(data_origin1, col_list)
data_loss_mean_reverse = data_32_mean(data_origin2, col_list_reverse)
# data_loss_zero = data_32_zero(data_origin2, col_list)
accuracy_mean = []
accuracy_mean_reverse = []
for i in range(32):
    _,_,output = model(data_loss_mean[i])
    output = output.to('cpu').detach().numpy().copy()
    pred_test = np.argmax(output,axis=1)
    accuracy_mean.append(accuracy_score(labels,pred_test))
    _,_,output = model(data_loss_mean_reverse[i])
    output = output.to('cpu').detach().numpy().copy()
    pred_test = np.argmax(output,axis=1)
    accuracy_mean_reverse.append(accuracy_score(labels,pred_test))
acc_graph(accuracy_mean,accuracy_mean_reverse,col_list,col_list_reverse)