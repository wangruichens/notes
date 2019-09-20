# Evaluate Metrics

### 掺杂一些个人理解，欢迎讨论

一般常见的模型评估指标有 logloss, RMSE, auc, accuracy, precision, recall, F1 score 等等。


> - [Loss vs. AUC](#loss-vs-auc)
> - [Is overfitting always bad ? ](#is-overfitting-always-bad-)




# Loss vs. AUC

针对分类任务来说，如果样本分布很不均衡，模型可以直接predict majority class来得到low loss. 使用AUC评估指标就可以解决这样的问题。 AUC具备 scale/classification invariant的能力。也就是说，对于阈值0.5，预测0.51和0.99,得到的AUC值是一样的。AUC更多的是表现模型的rank能力。 loss更多的关注模型的calibration。二者并没有太多相关性。

举例来说，代码可以参考[这里](loss_auc_explain.py)。

对于一个有4条数据的数据集，它的真实label为[0, 1, 1, 1]。
现在考虑两个不同的模型A,B。假设Model A预测的结果为： [0.5, 0.9, 0.9, 0.5], Model B预测的结果为[0.499, 0.501, 0.501, 0.501]。分别计算它们的AUC与LOSS。

    *** Model A : [0.5, 0.9, 0.9, 0.5] ***
    loss: 0.39925384810888576
    auc: 0.8333333333333334
    *** Model B : [0.499, 0.501, 0.501, 0.501] ***
    loss: 0.6911491778972723
    auc: 1.0
    
对于模型A来说，它对于预测正负样本的准确度要更高，但是AUC的值确不如B，因为有的样本被错误分类了。如果对模型的True Positive Rate， True Negative Rate要求较高，显然应该选择A。实际应用场景比如刷脸支付等。

对于模型B来说，似乎看起来就像random瞎猜一样，但是它能够完美的区分正负样本。如果模型训练的目的是作为一个排序模型（rank），显然应该选择B。实际应用场景就是推荐资讯，广告等。

在我们训练模型的时候，经常需要面对过拟合问题。有这样一种情况，从eval loss上来看，模型已经明显过拟合（train loss下降，eval loss 上升）。实际的结果就有点类似上面例子中由A到B的情况。模型的预测值会越来越接近0.5。也就是logloss在不断上升。但是AUC基本上一直是上升的。**那么问题来了，这种情况下的过拟合模型是我们想要的吗？应该如何选择呢？**


# Is overfitting always bad ? 