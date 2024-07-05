import numpy as np
from sklearn.metrics import confusion_matrix

def _compute_sensitivities(y_true, y_pred, labels=None):
	if len(y_true.shape) > 1:
		y_true = np.argmax(y_true, axis=1)
	if len(y_pred.shape) > 1:
		y_pred = np.argmax(y_pred, axis=1)

	conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

	sum = np.sum(conf_mat, axis=1)
	mask = np.eye(conf_mat.shape[0], conf_mat.shape[1])
	correct = np.sum(conf_mat * mask, axis=1)
	sensitivities = correct / sum

	sensitivities = sensitivities[~np.isnan(sensitivities)]

	return sensitivities

def minimum_sensitivity(y_true, y_pred, labels=None):
	return np.min(_compute_sensitivities(y_true, y_pred, labels=labels))

def accuracy_off1(y_true, y_pred, labels=None):
	if len(y_true.shape) > 1:
		y_true = np.argmax(y_true, axis=1)
	if len(y_pred.shape) > 1:
		y_pred = np.argmax(y_pred, axis=1)

	conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
	n = conf_mat.shape[0]
	mask = np.eye(n, n) + np.eye(n, n, k=1), + np.eye(n, n, k=-1)
	correct = mask * conf_mat

	return 1.0 * np.sum(correct) / np.sum(conf_mat)