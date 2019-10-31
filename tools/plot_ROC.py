import matplotlib.pyplot as plt 
import numpy as np
import csv

log_location = 'logs/'
png_location = 'png/'
with open(log_location + 'log_validation_preds.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	preds = np.array(list(reader)).astype(float)

threshold = np.linspace(0,1,50)
TP = np.zeros(len(threshold))
TN = np.zeros(len(threshold))
FP = np.zeros(len(threshold))
FN = np.zeros(len(threshold))
for idx,val in enumerate(threshold):
	for pred in preds:
		if pred[0] >= val and pred[1]==1.:
			TP[idx] += 1.
		elif pred[0] >= val and pred[1]==0.:
			FP[idx] += 1.
		elif pred[0] < val and pred[1]==1.:
			FN[idx] += 1.
		elif pred[0] < val and pred[1]==0.:
			TN[idx] += 1.
TPR = TP / (TP+FN)
FPR = FP / (FP+TN)
# Plot
plt.clf()   
plt.scatter(FPR,TPR,c='blue',marker='.')
plt.scatter(threshold,threshold,c='orange',marker='D')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(png_location+'ROC.pdf')

