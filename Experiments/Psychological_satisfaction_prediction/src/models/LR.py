'''
Logistic Regression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.linear_model import LogisticRegression

class LR(BaseModel):
	extra_log_args = ['penalty','solver']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--penalty',type=str,default='l2')
		parser.add_argument('--regularization',type=float,default=1.0) # C in document is the reverse of regularization
		parser.add_argument('--max_iter',type=int,default=100)
		parser.add_argument('--solver',type=str,default='lbfgs') # lbfgs, liblinear

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = LogisticRegression(random_state=args.random_seed,penalty=args.penalty,
									C=1/args.regularization, max_iter=args.max_iter,
									solver=args.solver)