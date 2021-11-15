from train import deap_preprocess, train, data_32_mean

datasets, labels = deap_preprocess("s01","arousal")
judge = True
col_list = [10 ,13 ,25  ,7  ,5  ,4 ,31 ,2 ,1 ,12 ,15 ,26 ,32 ,19 ,3 ,28 ,22 ,27 ,24 ,16 ,8 ,6 ,11 ,14 ,20, 29 ,23 ,18 ,9 ,21 ,17 ,30]
col_list_reverse = col_list.reverse()
roar_level = 32
if judge:
    data = data_32_mean(datasets, col_list, roar_level)
else:
    data = data_32_mean(datasets, col_list_reverse, roar_level)
print(data.shape)
for level in range(roar_level):
    data_train = data[level]
    train(data_train, labels, level+1, judge)