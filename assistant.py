import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg


def logist(x,w):	#input: (x: N*d, w: d*1), output: Probability array N*1
	return np.exp(-np.logaddexp(0,-np.dot(x,w)))

def logist_sparse(x,w):	#input: (x: N*d, w: d*1) both are sp.sparse.***_matrix, output: Probability sparse matrix N*1
	#return sp.sparse.csc_matrix(np.exp(-np.logaddexp(0,-(x*w).todense())))
	return np.exp(-np.logaddexp(0,-(x*w)))

def one_minus_logist_sparse(x,w):
	return np.exp(-np.logaddexp(0,(x*w)))

def logist_and_one_minus_logist_sparse(x,w): #output: NOT SPARSE !!!
	#temp = (x*w).todense()
	temp = (x*w)
	return [np.exp(-np.logaddexp(0,-temp)), np.exp(-np.logaddexp(0,temp))]

def one_minus_logist(x,w):
	return np.exp(-np.logaddexp(0,np.dot(x,w)))

def log_logist(x,w):#input: (x: N*d, w: d*1), output: log(Probability) array N*1
	return -np.logaddexp(0,-np.dot(x,w))

def log_1minus_logist(x,w):#input: (x: N*d, w: d*1), output: log(1-Probability) array N*1
	return -np.logaddexp(0,np.dot(x,w)-w)

def logLogist(x,w):	#input: (x: N*d, w: d*1), output: Log Probability array N*1
	return -np.logaddexp(0,-np.dot(x,w))

def logLogist_sparse(x,w): #input: (x: N*d, w: d*1) both are sp.sparse.***_matrix, output: Log probability sparse matrix N*1
	#return sp.sparse.csc_matrix(-np.logadexp(0,-(x*w).todense()))
	return -np.logaddexp(0,-(x*w))

def log_1_minus_logist_sparse(x,w):
	return -np.logaddexp(0,x*w)

def logLogist2(x,w):	#input: (x: N*d, w: d*1), output: Log Probability array N*1 ***CAUTION*** no bias version
	return -np.logaddexp(0,-np.dot(x,w))

def logist_prime(x,w):
	return (1.0)/(np.exp(np.logaddexp(0,-np.dot(x,w))))

########################################################################################################
#Calculate Precision, Recall and F-measure
#	input
#Y_true: N*1 array. True labels. {0,1}^N
#Y_predicted: N*1 array. Predicted labels. {0,1}^N
#
#       output
# (precision, recall, f-measure)
#
def calculate_Precision_Recall_FMeasure(Y_true,Y_predicted):
	######### comments ############
	#tp_plus_fp = #(True Positive) + #(False Positive)
	#tp_plus_fn = #(True Positive) + #(False Negative)
	#tp = #(True Positive)
	##############################
	
	tp_plus_fp = np.array([Y_predicted[i] for i in range(Y_predicted.size) if Y_predicted[i]==1]).sum() * 1.0
	print "tp+fp =", tp_plus_fp
	tp_plus_fn = np.array([Y_true[i] for i in range(Y_true.size) if Y_true[i]==1]).sum() * 1.0
	print "tp+fn =", tp_plus_fn
	tp = np.array([Y_true[i] for i in range(Y_true.size) if Y_predicted[i]==1]).sum() * 1.0
	print "tp =", tp
	precision = tp/tp_plus_fp
	recall = tp/tp_plus_fn
	f_measure = (2.0*recall*precision)/(recall + precision)
	print "precision =", precision
	print "recall =", recall
	print "f_measure =", f_measure
	return (precision,recall,f_measure)
