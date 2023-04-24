'''
Random Forest
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RF(BaseModel):
	extra_log_args = ['n_estimators','max_depth','max_features']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--n_estimators',type=int,default=100)
		parser.add_argument('--max_depth',type=int,default=None)
		parser.add_argument('--min_samples_split',type=int,default=2)
		parser.add_argument('--min_samples_leaf',type=int,default=1)
		parser.add_argument('--max_leaf_nodes',type=int,default=None)
		parser.add_argument('--max_features',type=str,default=None)
		parser.add_argument('--criterion',type=str,default='gini')

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = RandomForestClassifier(random_state=args.random_seed, n_estimators=args.n_estimators,
					max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
					min_samples_split=args.min_samples_split, max_leaf_nodes = args.max_leaf_nodes,
					max_features=args.max_features,criterion=args.criterion)