import numpy as np


def logloss(y_true, y_pred, eps=1e-15):
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	assert (len(y_true) and len(y_true) == len(y_pred))
	p = np.clip(y_pred, eps, 1 - eps)
	loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1 - p))

	return loss / len(y_true)


def auc_calculate(labels, preds, n_bins=10):
	postive_len = sum(labels)
	negative_len = len(labels) - postive_len
	total_case = postive_len * negative_len
	pos_histogram = [0 for _ in range(n_bins)]
	neg_histogram = [0 for _ in range(n_bins)]
	bin_width = 1.0 / n_bins
	for i in range(len(labels)):
		nth_bin = int(preds[i] / bin_width)
		if labels[i] == 1:
			pos_histogram[nth_bin] += 1
		else:
			neg_histogram[nth_bin] += 1
	accumulated_neg = 0
	satisfied_pair = 0
	for i in range(n_bins):
		satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
		accumulated_neg += neg_histogram[i]

	return satisfied_pair / float(total_case)


y = [0, 1, 1, 1]
pred_a = [0.5, 0.9, 0.9, 0.5]
pred_b = [0.499, 0.501, 0.501, 0.501]
print('*** Model A : [0.5, 0.9, 0.9, 0.5] ***')
print('loss:', logloss(y, pred_a))
print('auc:', auc_calculate(y, pred_a))
print('*** Model B : [0.499, 0.501, 0.501, 0.501] ***')
print('loss:', logloss(y, pred_b))
print('auc:', auc_calculate(y, pred_b))
