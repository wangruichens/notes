import matplotlib.pyplot as plt

step = [2000, 4000, 6000, 8000, 10000,12000]
eval_auc = [0.75383, 0.764226, 0.767599, 0.76829, 0.770303,0.7700608]
eval_loss = [0.5246, 0.5226, 0.5142, 0.5283, 0.5295,0.5301698]
train_loss = [0.513,0.52,0.5015,0.4918,0.4696,0.4672]
test_auc = [0.75225,0.76438,0.76816,0.76896,0.7704,0.770805]
plt.subplot(211)
plt.plot(step,eval_auc,'C0',step,test_auc,'C3')
plt.legend(labels=['eval auc','test auc'])
plt.xlabel('step')
plt.ylabel('auc')

plt.subplot(212)
plt.plot(step,eval_loss,'C0',step,train_loss,'C1')
plt.legend(labels=['eval loss','train loss'])
plt.xlabel('step')
plt.ylabel('loss')
plt.show()
