'''
SVM model
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.svm import SVC

class SVM(BaseModel):
	extra_log_args = ['regularization','kernel',]

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--regularization',type=float,default=1.0) # C in document is the reverse of regularization
		parser.add_argument('--kernel',type=str,default='rbf') # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
		parser.add_argument('--degree',type=int,default=3) # only useful for kernel==poly
		parser.add_argument('--gamma',type=str,default='scale') # for kernel in [rbf,poly,sigmoid]
		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = SVC(random_state=args.random_seed, C=1/args.regularization,
						kernel=args.kernel, degree=args.degree,gamma=args.gamma)
