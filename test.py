import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from scipy.stats import rankdata
from ACRNN import ACRNN,A_CRNN,CRNN_A
import matplotlib.pyplot as plt
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
datasets = datasets.permute(0,3,1,2)

n_channel=32
model = ACRNN(n_channel)
PATH='./param/model.pth'
model.load_state_dict(torch.load(PATH))
map,y_map,output = model(datasets)
#print(y_map.shape)
output = output.to('cpu').detach().numpy().copy()
pred_test = np.argmax(output,axis=1)
print(accuracy_score(labels,pred_test))

# channel_wise_attetion Survey
map_T = map.permute(1,0)
map_cw = torch.mean(map,axis=0)
map_cw = map_cw.to('cpu').detach().numpy().copy()
rank = rankdata(-map_cw)
index = ['FP1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','FP2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
data=[]       
map_reshape = torch.reshape(torch.cat([map]*(1*384),axis=1),[-1,1,384,32])
for i in range(10):
    trans=np.where(rank==i+1)[0][0]
    data.append(index[trans])
print(data)
plt.figure()
plt.plot(index,map_cw)
plt.grid()
plt.savefig("channel_weight.png")
plt.show()
model_A=CRNN_A(n_channel)
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
min_index=np.where(rank==32)[0][0]
accuracy=[]
accuracy.append(accuracy_score(labels,pred_test))
random_accuracy=[]
random_accuracy.append(accuracy_score(labels,pred_test))
random=map_T
for i in range(32):
    trans=np.where(rank==i+1)[0][0]
    map_T[trans]=map_T[min_index]
    map_reshape=map_T.permute(1,0)
    map_reshape = torch.reshape(torch.cat([map_reshape]*(1*384),axis=1),[-1,1,384,32])
    random[i] = map_T[min_index]
    random_reshape = torch.reshape(torch.cat([random]*(1*384),axis=1),[-1,1,384,32])
    data=map_reshape*datasets
    data_random = random_reshape*datasets
    output=model_A(data)
    output = output.to('cpu').detach().numpy().copy()
    pred = np.argmax(output,axis=1)
    accuracy.append(accuracy_score(labels,pred))
    output=model_A(data_random)
    output = output.to('cpu').detach().numpy().copy()
    pred = np.argmax(output,axis=1)
    random_accuracy.append(accuracy_score(labels,pred))
plt.figure(figsize=(8,6))
plt.plot(accuracy,label='True')
plt.plot(random_accuracy,label='Random')
plt.title('Learning Curve of acc roar')
plt.legend(fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid()
plt.savefig("./result/acc_roar_graph.png")
plt.show()

# self_attention Survey
print('_____________________________________________')
y_map = y_map.reshape(800,64)
y_map_mean=torch.mean(y_map,axis=1)
y_map_mean = y_map_mean.to('cpu').detach().numpy().copy()
#print(y_map_mean.shape)
y_min_index=np.argmin(y_map_mean)
#print(y_min_index)
min_sort=np.argsort(-y_map_mean)
#print(min_sort)
y_accuracy=[]
y_accuracy.append(accuracy_score(labels,pred_test))
y_random_accuracy=[]
y_random_accuracy.append(accuracy_score(labels,pred_test))
data=datasets
random_data=datasets
for i in range(800):
    j=min_sort[i]
    data[j]=datasets[y_min_index]
    random_data[i]=datasets[y_min_index]
    _,_,output=model(data)
    output = output.to('cpu').detach().numpy().copy()
    pred = np.argmax(output,axis=1)
    y_accuracy.append(accuracy_score(labels,pred))
    _,_,output=model(random_data)
    output = output.to('cpu').detach().numpy().copy()
    pred = np.argmax(output,axis=1)
    y_random_accuracy.append(accuracy_score(labels,pred))
plt.figure(figsize=(8,6))
plt.plot(y_accuracy,label='True')
plt.plot(y_random_accuracy,label='Random')
plt.title('Learning Curve of acc roar self_attention')
plt.legend(fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid()
plt.savefig("./result/acc_roar_self_attention_graph.png")
plt.show()