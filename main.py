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
    # print(labels)
    # print(datasets.shape)
    # labels = np.asarray(pd.get_dummies(labels),dtype=np.int8)
    # print(labels.shape)
    datasets = datasets.reshape(-1,384,32,1).astype('float32')
    labels = labels.astype('int64')
    # print(type(labels))
    return datasets, labels

if __name__ == '__main__':
    # parameter
    ## training
    training_epochs = 1
    batch_size = 10
    emotion = "arousal"
    # deap_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11','s12', 's13','s14','s15', 's16', 's17','s18', 's19', 's20','s21', 's22', 's23', 's24', 's25', 's26','s27', 's28', 's29', 's30', 's31', 's32']
    deap_subjects = ['s01']
    ## EEG sample
    window_size = 384
    n_channel = 32
    ## loss
    learning_rate = 1e-4
    # gpu setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # network
    model = ACRNN(n_channel)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for list in deap_subjects:
        # dataset load
        datasets, labels = deap_preprocess(list,"arousal")
        datasets = torch.from_numpy(datasets).clone()
        labels = torch.from_numpy(labels).clone()
        # print(labels)
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

            # set train batch number per epoch
            batch_num_epoch_train = train_x.shape[0]//batch_size
            batch_num_epoch_test = test_x.shape[0]//batch_size
            train_acc = []
            test_acc = []
            train_loss = []
            test_loss = []
            # training_epochs
            for epoch in range(training_epochs):
                loss = None
                correct_train = 0
                loss_train_total = 0
                batch_train_loss = []
                # training process
                model.train(True)
                for batch in range(batch_num_epoch_train):
                    offset = (batch*batch_size%(train_y.shape[0]-batch_size))
                    batch_x = train_x[offset:(offset+batch_size),:,:,:]
                    batch_x = batch_x.reshape(len(batch_x),1,window_size,n_channel)
                    # print(batch_x.shape)
                    batch_y = train_y[offset:(offset+batch_size)]
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    # print(batch_y)
                    # print(batch_y.shape)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    # print(output.shape)
                    # print(output.shape)
                    # target = torch.empty(batch_size,dtype=torch.long).random_(2) # 修正必要
                    loss = loss_function(output,batch_y)
                    batch_train_loss.append(loss.item())  
                    pred_train = output.argmax(dim=1,keepdim=True)
                    loss_train_total += loss.item()
                    correct_train += (pred_train == batch_y).sum().item()
                    loss.backward()
                    optimizer.step()
                    
                avg_loss_train = loss_train_total/batch_num_epoch_train
                avg_acc_train = correct_train/batch_num_epoch_train
                print('Training log: {} fold. {} epoch. Loss: {}'.format(curr_fold+1,epoch+1,avg_loss_train))
                train_loss.append(avg_loss_train)
                train_acc.append(avg_acc_train)
                # print(train_loss)

                # test process
                model.eval()
                loss_test_total = 0
                correct_test = 0
                with torch.no_grad():
                    for batch in range(batch_num_epoch_test):
                        offset = (batch*batch_size%(test_y.shape[0]-batch_size))
                        batch_x = test_x[offset:(offset+batch_size),:,:,:]
                        batch_x = batch_x.reshape(len(batch_x),1,window_size,n_channel)
                        # print(output.shape)
                        batch_y = test_y[offset:(offset+batch_size)]
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        output = model(batch_x)
                        loss_test_total += loss_function(output,batch_y).item()
                        pred_test = output.argmax(dim=1,keepdim=True)
                        correct_test += (pred_test == batch_y).sum().item()
                avg_loss_test = loss_test_total/batch_num_epoch_test
                avg_acc_test = correct_test/batch_num_epoch_test
                test_acc.append(avg_loss_test)
                test_loss.append(avg_acc_test)
                print('Train Loss: {}. Train Accuracy {}.'.format(avg_loss_train,avg_acc_train))
                print('Test Loss: {}. Test Accuracy: {}.'.format(avg_loss_test,avg_acc_test))