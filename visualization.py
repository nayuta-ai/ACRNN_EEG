import scipy
import mne
import pickle
import torch
from ACRNN import ACRNN
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def topomap(data):
    norm = scipy.stats.zscore(data)
    biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    n_channels = len(biosemi_montage.ch_names)
    fake_info = mne.create_info(ch_names=biosemi_montage.ch_names,sfreq=128.,
                                ch_types='eeg')
    rng = np.random.RandomState(0)
    data_plot = norm[0:32,0:1]
    fake_evoked = mne.EvokedArray(data_plot, fake_info)
    fake_evoked.set_montage(biosemi_montage)
    mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info)
    plt.savefig("./result/topomap_data.png")
    plt.show()

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
print(map.shape)

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
    
col = colleration_coefficinet(labels,map)
col = col.reshape(32,1)
col = col.to('cpu').detach().numpy().copy()
topomap(col)
data = datasets.mean(axis=2)
data = data.reshape(800,32)
col_data = colleration_coefficinet(data, map)
col_data = col_data.reshape(32,1)
col_data = col_data.to('cpu').detach().numpy().copy()
topomap(col_data)