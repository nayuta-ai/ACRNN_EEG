import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

from ACRNN import ACRNN
import torch.optim as optim

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
"""
# dataset_deap
deap_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11','s12', 's13','s14','s15', 's16', 's17','s18', 's19', 's20',
 's21', 's22', 's23', 's24', 's25', 's26','s27', 's28', 's29', 's30', 's31', 's32']
# dataset load
datasets, labels = deap_preprocess("s01","arousal")
datasets = torch.from_numpy(datasets).clone()
labels = torch.from_numpy(labels).clone()
# data Hight Windowsize Channel 
datasets = datasets.permute(0,3,1,2)
print(datasets.shape)
"""
if __name__ == '__main__':
    # parameter
    ## training
    training_epochs = 100
    batch_size = 10
    emotion = "arousal"
    # deap_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11','s12', 's13','s14','s15', 's16', 's17','s18', 's19', 's20','s21', 's22', 's23', 's24', 's25', 's26','s27', 's28', 's29', 's30', 's31', 's32']
    deap_subjects = ['s01']
    ## EEG sample
    window_size = 384
    n_channel = 32
    ## input
    input_channel_num = 1
    input_height = 32
    input_width = 384
    num_labels = 2
    ## channel wise attention
    k = 15
    ## conv
    kernel_height = 32
    kernel_width = 45
    kernel_stride = 1
    conv_channel_num = 40
    ## pooling
    pooling_height = 1
    pooling_width = 75
    pooling_stride = 10
    ## LSTM
    n_hidden_state = 64
    ## self attention
    attention_size = 512
    ## loss
    learning_rate = 1e-4
    # network
    model = ACRNN(input_channel_num,input_width,input_height,k,kernel_height,kernel_width,kernel_stride,pooling_height,pooling_width,pooling_stride,conv_channel_num,n_hidden_state,attention_size)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for list in deap_subjects:
        # dataset load
        datasets, labels = deap_preprocess(list,"arousal")
        datasets = torch.from_numpy(datasets).clone()
        labels = torch.from_numpy(labels).clone()
        # data Hight Windowsize Channel 
        datasets = datasets.permute(0,3,1,2)
        fold = 10
        test_accuracy_all_fold = np.zeros(shape=[0],dtype=float)
        # fold '0-9'
        for curr_fold in range(fold):
            fold_size = datasets.shape[0]//fold
            indexes_list = [i for i in range(len(datasets))]
            indexes = np.array(indexes_list)
            split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
            split = np.array(split_list)
            test_y = labels[split]
            test_x = datasets[split]
            split_set = set(indexes_list)^set(split_list)
            split = [x for x in split_set]
            split = np.array(split)
            train_x = datasets[split]
            train_y = labels[split]
            # gpu setting
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)

            # set train batch number per epoch
            batch_num_epoch = train_x.shape[0]//batch_size
            train_acc = []
            test_acc = []
            best_test_acc = []
            train_loss = []
            # training_epochs
            for epoch in range(training_epochs):
                loss = None
                # training process
                model.train(True)
                for batch in range(batch_num_epoch):
                    offset = (batch*batch_size%(train_y.shape[0]-batch_size))
                    batch_x = train_x[offset:(offset+batch_size),:,:,:]
                    batch_x = batch_x.reshape(len(batch_x),1,window_size,n_channel)
                    print(batch_x.shape)
                    batch_y = train_y[offset:(offset+batch_size),:]
                    print(batch_y.shape)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = loss_function(output,batch_y)
                    loss.backward()
                    optimizer.step()

