# Cold start problem 

[相关paper](https://github.com/wangruichens/papers-machinelearning/tree/master/recsys/coldstart)

# 概述

冷启动问题大概可以分为 用户冷启动，物品冷启动。在目前大部分推荐算法基于协同过滤 or 矩阵分解的情况下，无法有效的分析新用户喜好 or 无法感知新物品。 导致整个系统中无法推荐新的物品。

exploitation-exploration trade-off

---

以下内容摘自： https://zhuanlan.zhihu.com/p/32502139

(千篇一律的新闻) 现在的新闻客户端都使用了机器学习进行智能排序，你有没有跟我一样的体验：

- 某类型的新闻，你点击的越多，下次登录时就会看到的越多
- 看到的越多，点的机会也就越多
- 最后满眼的新闻都是千篇一律

借着这个例子，我们说说把现实问题映射到监督学习的过程中存在的坑：

历史数据的收集直接受到了算法影响，是有偏向性(Bias)的：算法推荐了一个新闻，用户才有机会给出反馈，系统才会收集到反馈。但是还有千千万万的新闻是没推荐出来的，用户不知道它们好不好，系统也没机会收集那些新闻的反馈
正反馈是很容易获得的，负反馈却需要自己去猜：算法推荐了k个新闻，用户只点击其中的一个，这并不100%意味着是对剩下的新闻的否定：（1）这个新闻没有被用户注意到，（2）这个新闻用户也感兴趣喜欢，只是时间有限，这次没点
结论：既然历史数据是受算法影响的，用户又只提供了正反馈，那么根据历史数据训练就会不断强化自己去推荐已经推荐过的东西，使得模型陷入一个局部最优，潜在的好的东西迟迟得不到推荐。

---


address这个问题：

不考虑冷启动问题的策略： 
- 随机： 无论新用户还是新物品， 都采用随机推荐。 往往这样需要较长时间，较大成本。准确率不会超过50%。长期会伤害整个系统的良性循环。
- 数值填充： 比如初始一个假设，新物品的点击率就是平均值。并以此来填充计算MF。这样不准确，也并不能保证个体性差异。

考虑冷启动问题的策略：
- 混合推荐： 融合多种算法。 比如设计算法，分析文章间相关性，文章质量评分等。
- 融合其他数据： 引用更细粒度的特征，side info。引入上下文信息

### Addressing the Item Cold-start Problem by Attribute-driven Active Learning

2018年5月

主要解决 item cold start problem
1) 结合上下文信息，使用LCE(Local Collective Embeddings)
2) 基于item 特征与用户rating history 来筛选用户。 

### A Simple Multi-Armed Nearest-Neighbor Bandit for Interactive Recommendation

2019年10月 Recsys 2019 

主要解决 user cold start problem 

多臂老虎机是一个经典的强化学习问题。推荐场景中也有类似概念（active learning），一方面需要直接给出符合用户兴趣的next recommendation(exploitation)。 另一方面，也需要考虑长期收益(exploration)， gaining knowledge 来获得用户兴趣

1) a simple multi-armed bandit elaboration of neighbor-based
collaborative filtering
2) 可以看作是nearest neighbor 的一个变种， 不需要side info。pure collaborative filtering。

#### multi-armed bandits
- eplison greedy
- Upper Confidence Bounds (UCB) 
- Thompson Sampling  (Bayesian)
- EXP3

类比强化学习（Based on PMF, decompose 用户的reward）：推荐一个item就类似于pull一个arm。 用户的反馈可以看作是reward。

Probabilistic Matrix Factorization (PMF)

标准的MF方法就是N个用户，M个物品，构造M*N的矩阵来做矩阵分解。一般情况下矩阵很稀疏，需要降维来做。比如抽象出物品的K个特征。这就是Latent feature。也就是latent factor model (LFM)

本文类比的是MAB问题，不需要额外参数，不需要训练，只需要调整超参数。
把用户看作arm。目标用户u,邻居用户v, 目标item i。选择v，根据p(i|v)推荐i给u。再根据u的反馈来更新u的邻居关系p(u|v)，二项分布表示，使用thompson sampling 来更新。 先验为beta分布。其中alpha,beta参数需要设置初始值，后续由用户反馈更新。

优势： bandit相比于正常MF模型，冷启动问题中recall结果更好，更适应稀疏情况，收敛更快。 

相关MAB: https://banditalgs.com/2016/10/01/adversarial-bandits/

### ExcUseMe: Asking Users to Help in Item Cold-Start Recommendations

Recsys 2015 from yahoo 

主动探索算法，选择一部分用户探索new items

- online selecting users for item cold-start

user cold start : 使用seed sets

item cold start : additional attributes. side info embeddings linear combination.

主要思想是选择k个用户（采用k秘书问题）来确定new item latent embedding. 用户线上实时打分，根据评分选择最有可能和新item 进行交互的用户来推荐，打分函数需要训练一个UseMe向量，用来表达那些会对新item感兴趣的用户向量平均，也就是guides to select users who tend to provide feedback

### Feature-based factorized Bilinear Similarity Model for Cold-Start Top-n Item Recommendation

SDM 2015 

Feature-based factorized Bilinear Similarity Model (FBSM)

想法其实就是item cf。不过使用的是细粒度特征。
 
 论文做了计算上的优化，计算权重的矩阵转化为对角线矩阵+非对角线矩阵，其中对角线矩阵直接计算两item i,j 同一维度k的交叉权重，非对角线计算ij之间不同维度的权重（降维）。

举例movieLens, 根据tfidf权重选择词作为item特征维度。

### From Zero-Shot Learning to Cold-Start Recommendation

AAAI 2019年6月 
代码(matlab) ： https://github.com/lijin118/LLAE

比较新颖的想法，将图像zero shot learning 和冷启动问题结合到一起。提出Low rank Linear AutoEncoder方法。 

用户偏好来源于用户本身标签+用户行为。文章定义的新用户为具有用户标签，但是没有用户行为的那些用户。
定义auto encoder: 将用户行为map到用户标签上， 再根据用户标签decode 用户行为。

用户行为、用户标签矩阵也是高维稀疏的，使用svd分解降维。实际论文中使用的数据集，用户attribute, item维度都还是很低的（不到1000）。

### Item Cold-Start Recommendations: Learning Local Collective Embeddings

Recsys 2014 yahoo 结合user与item side info共同做MF

本质上也是MF的方法。LCE Local Collective Embeddings。 结合item本身content与collaborative info。假设 item 与user在相同的低维空间上。

item本来是一个二维矩阵，行i表示第i个item,列j表示item的j个特征。可以使用词向量One hot or tfidf权重。
用户矩阵表示行i用户与列j item的交互信息。
输入一个item 向量， 计算哪些user 会点击。


matlab代码: https://github.com/msaveski/LCE

### Spectral Collaborative Filtering

Recsy 2018 

python代码： https://github.com/lzheng21/SpectralCF

基于graph的思想，更多的考虑user与item连通性。
图的傅立叶变换 L=D-A

### Variational Autoencoders for Collaborative Filtering

WWW 2018
github 代码： https://github.com/dawenl/vae_cf

采用变分推断，KL divergence来生成隐式反馈。
文章对比了： WMF,  SLIM, CDAE, NCF

### Neural Collaborative Filtering

WWW 2017
经典的使用nn来学习user and item embedding的方法。

our method can be easily adjusted to address the cold-start problem by using content features to represent users and items。

其实就是使用更细粒度的特征来解决 。

### Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches

Recsys 2019的 best paper
比较有意思了，仅供参考来看吧

比较有价值的是作者公开了代码： https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation

