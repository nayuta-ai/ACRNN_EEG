import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import numpy as np
from scipy.stats import rankdata
from ACRNN import ACRNN

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

def train(datasets, labels ,roar, judge):
    # parameter
    ## training
    training_epochs = 1000
    batch_size = 10
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
    # dataset load
    # datasets = torch.from_numpy(datasets).clone()
    labels = torch.from_numpy(labels).clone()
    # data Hight Windowsize Channel 
    fold = 10
    total_train_acc=[]
    total_train_loss=[]
    total_test_acc=[]
    total_test_loss=[]
    # fold '0-9'
    for curr_fold in range(fold):
        fold_size = datasets.shape[0]//fold
        indexes_list = [i for i in range(len(datasets))]
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
        batch_num_epoch_train = train_x.shape[0] // batch_size
        batch_num_epoch_test = test_x.shape[0] // batch_size
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
                batch_x = Variable(batch_x)
                # print(batch_y)
                # print(batch_y.shape)
                optimizer.zero_grad()
                _, _, output = model(batch_x)
                # print(output)
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
                
            avg_loss_train = loss_train_total / batch_num_epoch_train
            avg_acc_train = correct_train / batch_num_epoch_train
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
                    _, _, output = model(batch_x)
                    loss_test_total += loss_function(output,batch_y).item()
                    pred_test = output.argmax(dim=1,keepdim=True)
                    correct_test += (pred_test == batch_y).sum().item()
            avg_loss_test = loss_test_total/batch_num_epoch_test
            avg_acc_test = correct_test/batch_num_epoch_test
            test_acc.append(avg_acc_test)
            test_loss.append(avg_loss_test)
            print('Train Loss: {}. Train Accuracy {}.'.format(avg_loss_train,avg_acc_train))
            print('Test Loss: {}. Test Accuracy: {}.'.format(avg_loss_test,avg_acc_test))
            PATH='./param1/model_{}_{}.pth'.format(roar,judge)
            torch.save(model.state_dict(),PATH)
        total_train_acc.append(train_acc)
        total_train_loss.append(train_loss)
        total_test_acc.append(test_acc)
        total_test_loss.append(test_loss)
    with open("./data1/train_loss_{}_{}.pickle".format(roar,judge),mode="wb") as f:
        pickle.dump(total_train_loss,f)
    with open("./data1/test_loss_{}_{}.pickle".format(roar,judge),mode="wb") as f:
        pickle.dump(total_test_loss,f)
    with open("./data1/train_acc_{}_{}.pickle".format(roar,judge),mode="wb") as f:
        pickle.dump(total_train_acc,f)
    with open("./data1/test_acc_{}_{}.pickle".format(roar,judge),mode="wb") as f:
        pickle.dump(total_test_acc,f)

def space_roar(dataset,roar_level):
    datasets = dataset.permute(2,0,1,3)
    data_mean = datasets.mean(axis=0)
    data_zero = torch.zeros([800,384,1])
    for level in roar_level:
        datasets[level] = data_zero
    datasets = datasets.permute(1,3,2,0)
    return datasets

def rank_channel(map):
    map = map.mean(axis=0)
    map = map.to('cpu').detach().numpy().copy()
    rank_list = rankdata(map).astype(int)
    return rank_list

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

def data_32_mean(data_origin, rank, roar_level):
    rank = rank[:roar_level]
    data = torch.randn(32,32,800,1,384)
    data_origin = torch.torch.from_numpy(data_origin).clone()
    data_origin = data_origin.permute(2,0,3,1)
    data_mean = data_origin.mean()
    if rank == None:
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
    return data[:roar_level]

def data_32_zero(data_origin, rank, roar_level):
    rank = rank[:roar_level]
    data = torch.randn(32,32,800,1,384)
    data_origin = torch.torch.from_numpy(data_origin).clone()
    data_origin = data_origin.permute(2,0,3,1)
    data_zero = torch.zeros(800,1,384)
    if rank == None:
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
