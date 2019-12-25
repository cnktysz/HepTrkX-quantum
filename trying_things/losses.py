import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
log_location = 'logs/tensorflow/ENE/lr_0_1/'

with open(log_location + 'log_validation_preds.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid_preds = np.array(list(reader)).astype(float) # [preds,labels]


y_true = valid_preds[:,1]
y_pred = valid_preds[:,0]
n_1    = sum(y_true)
n_0    = len(y_true) - n_1
class_weights = compute_class_weight('balanced', np.unique(y_true), y_true)


loss0 = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=0)
#loss2 = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=class_weights)

# Take the cost like normal
error = tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true)
print(error)
# Scale the cost by the class weights
#scaled_error = tf.multiply(error, class_weights)

# Reduce
#loss2 = tf.reduce_mean(scaled_error) 

print('True Edge: %d, False Edge: %d' %(n_1,n_0))
print(class_weights)
print(loss0)
print(loss1)

