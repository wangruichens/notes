# Transformer

相关论文是: [Attention Is All You Need](https://github.com/wangruichens/papers-machinelearning/blob/master/nlp/Attention%20Is%20All%20You%20Need.pdf)

论文主要几个关键点 : 
* Replace RNN with attention
* Scaled dot production 
* Multi head attention
* Position representation

在以往，RNN结构有一个突出的问题，比较难以并行化，需要序列化计算。一个解决办法是替换rnn模块为cnn。
通过调整cnn的kernel大小，也可以实现一个node同时观察序列多个位置的效果。

Transformer 也可以叫 Self Attention Layer。也是为了解决 rnn 而提出来的。理论上可以替换任何rnn based模型。
我们从基础的模块讲起

## Scaled Dot-Production Attention
每一个input X 由三个值共同表达， 分别是Q,K,V。
![2](2.png)
- Query (to match others)
- Key (to be matched)
- Value (information to be extracted)

首先，对于pos<sub>1</sub>位置的token, 有Q<sub>1</sub> × K<sub>i</sub> 得到位于pos <sub>i</sub>的权重。再加上softmax()以后，按相对应的比例取对应token<sub>i</sub>的Value。

![1](1.png)

