# Cold start problem 

# 概述

冷启动问题大概可以分为 用户冷启动，物品冷启动。在目前大部分推荐算法基于协同过滤 or 矩阵分解的情况下，无法有效的分析新用户喜好 or 无法感知新物品。 导致整个系统中无法推荐新的物品。

exploitation-exploration trade-off


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
- Thompson Sampling 
- EXP3

类比强化学习（Based on PMF, decompose 用户的reward）：推荐一个item就类似于pull一个arm。 用户的反馈可以看作是reward。

Probabilistic Matrix Factorization (PMF)

标准的MF方法就是N个用户，M个物品，构造M*N的矩阵来做矩阵分解。一般情况下矩阵很稀疏，需要降维来做。比如抽象出物品的K个特征。这就是Latent feature。也就是latent factor model (LFM)

本文类比的是MAB问题，不需要额外参数，不需要训练，只需要调整超参数。
把用户看作arm。目标用户u,邻居用户v, 目标item i。选择v，根据p(i|v)推荐i给u。再根据u的反馈来更新u的邻居关系p(u|v)，二项分布表示，使用thompson sampling 来更新。 先验为beta分布。