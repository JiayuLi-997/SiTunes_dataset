'''
Decision Trees
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.tree import DecisionTreeClassifier

class DT(BaseModel):
	extra_log_args = ['max_depth','min_samples_split']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--max_depth',type=int,default=None)
		parser.add_argument('--min_samples_split',type=int,default=2)
		parser.add_argument('--min_samples_leaf',type=int,default=1)
		parser.add_argument('--max_features',type=str,default=None)

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = DecisionTreeClassifier(random_state=args.random_seed,max_depth=args.max_depth,
					min_samples_leaf=args.min_samples_leaf,min_samples_split=args.min_samples_split,
					max_features=args.max_features)