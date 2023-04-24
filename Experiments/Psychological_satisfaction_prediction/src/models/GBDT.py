'''
Gradient Boosted Decision Trees
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.ensemble import GradientBoostingClassifier

class GBDT(BaseModel):
	extra_log_args = ['lr','n_estimators']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--lr',type=float,default=0.1)
		parser.add_argument('--n_estimators',type=int,default=100)
		parser.add_argument('--subsample',type=float,default=1.0)
		parser.add_argument('--max_depth',type=int,default=None)
		parser.add_argument('--min_samples_split',type=int,default=2)
		parser.add_argument('--min_samples_leaf',type=int,default=1)
		parser.add_argument('--max_leaf_nodes',type=int,default=None)
		parser.add_argument('--max_features',type=str,default=None)

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = GradientBoostingClassifier(random_state=args.random_seed, learning_rate=args.lr,
		n_estimators=args.n_estimators, subsample=args.subsample,
					max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
					min_samples_split=args.min_samples_split, max_leaf_nodes = args.max_leaf_nodes,
					max_features=args.max_features,
					validation_fraction=0.125, n_iter_no_change=10)
