from train2 import deap_preprocess, train, data_32_zero

datasets, labels = deap_preprocess("s01","arousal")
datasets = datasets[0:720]
labels = labels[0:720]
test_datasets = datasets[720:800]
test_labels = labels[720:800]
judge = True
col_list = [10 ,13 ,25  ,7  ,5  ,4 ,31 ,2 ,1 ,12 ,15 ,26 ,32 ,19 ,3 ,28 ,22 ,27 ,24 ,16 ,8 ,6 ,11 ,14 ,20, 29 ,23 ,18 ,9 ,21 ,17 ,30]
roar_level = 32
if judge:
    print(col_list)
else:
    col_list.reverse()
    print(col_list)
data = data_32_zero(datasets, col_list, roar_level)

for level in range(roar_level):
    data_train = data[level]
    train(data_train, labels, level+1, judge)
