import pickle
from graph import loss_graph,acc_graph
import matplotlib.pyplot as plt

with open('./data/train_loss.pickle', "rb") as fp:
     train_loss= pickle.load(fp)
total_train_loss=[]
for i in range(len(train_loss)):
    total_train_loss += train_loss[i]

with open('./data/train_acc.pickle', "rb") as fp:
     train_acc= pickle.load(fp)
total_train_acc=[]
for i in range(len(train_acc)):
    total_train_acc += train_acc[i]

with open('./data/test_loss.pickle', "rb") as fp:
     test_loss= pickle.load(fp)
total_test_loss=[]
for i in range(len(test_loss)):
    total_test_loss += test_loss[i]

with open('./data/test_acc.pickle', "rb") as fp:
     test_acc= pickle.load(fp)
total_test_acc=[]
for i in range(len(test_acc)):
    total_test_acc += test_acc[i]

plt.figure(figsize=(8,6))
plt.plot(total_train_loss,label='train_loss')
plt.title('Learning Curve of loss')
plt.legend(fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid()
plt.savefig("./result/loss_train_graph.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(total_test_loss,label='test_loss')
plt.title('Learning Curve of loss')
plt.legend(fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid()
plt.savefig("./result/loss_test_graph.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(total_train_acc,label='train_acc')
plt.title('Learning Curve of acc')
plt.legend(fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid()
plt.savefig("./result/acc_train_graph.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(total_test_acc,label='test_acc')
plt.title('Learning Curve of acc')
plt.legend(fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid()
plt.savefig("./result/acc_test_graph.png")
plt.show()

loss_graph(total_train_loss,total_test_loss)
acc_graph(total_train_acc,total_test_acc)