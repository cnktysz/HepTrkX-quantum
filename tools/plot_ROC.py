import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
log_location = 'logs/'
png_location = 'png/'
with open(log_location + 'log_validation_preds.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	preds = np.array(list(reader)).astype(float)

fpr,tpr,thresholds = metrics.roc_curve(preds[:,1].astype(int),preds[:,0],pos_label=1 )
auc = metrics.auc(fpr,tpr)
print('AUC: ' + str(auc))
# Plot
plt.clf()   
plt.plot(fpr,tpr,c='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(png_location+'ROC.pdf')

