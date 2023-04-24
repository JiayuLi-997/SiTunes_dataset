'''
SGD classifier
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.linear_model import SGDClassifier

class SGD(BaseModel):
	extra_log_args = ['loss','penalty','regularization','l1_ratio']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss',type=str,default='hinge')
		parser.add_argument('--penalty',type=str,default='l2')
		parser.add_argument('--regularization',type=float,default=1e-4) # C in document is the reverse of regularization
		parser.add_argument('--l1_ratio',type=float,default=0.15)
		parser.add_argument('--max_iter',type=int,default=1000)

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = SGDClassifier(random_state=args.random_seed,loss=args.loss,penalty=args.penalty,
									alpha=args.regularization,l1_ratio=args.l1_ratio, max_iter=args.max_iter,
									early_stopping=True, validation_fraction=0.125)