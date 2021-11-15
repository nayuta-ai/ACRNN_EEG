import matplotlib.pyplot as plt

def loss_graph(train_loss,test_loss):
    plt.figure(figsize=(8,6))
    plt.plot(train_loss,label='train_loss')
    plt.plot(test_loss,label='test_loss')
    plt.title('Learning Curve of loss')
    plt.legend(fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()
    plt.savefig("./result/loss_graph.png")
    plt.show()

def acc_graph(train_acc,test_acc):
    plt.figure(figsize=(8,6))
    plt.plot(train_acc,label='train_acc')
    plt.plot(test_acc,label='test_acc')
    plt.title('Learning Curve of accuracy')
    plt.legend(fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()
    plt.savefig("./result/accuracy_graph.png")
    plt.show()

def acc_graph(true_rank,fake_rank,rank_list,rank_list_reverse):
    plt.figure(figsize=(8,6))
    l = []
    l_reverse = []
    for i in rank_list:
        l.append(true_rank[i-1])
    for i in rank_list_reverse:
        l_reverse.append(fake_rank[i-1])
    plt.plot(l, label='true')
    plt.plot(l_reverse, label='fake')
    plt.title('Learning Curve of accuracy loss')
    plt.legend(fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()
    plt.savefig("./result/accuracy_loss.png")
    plt.show()