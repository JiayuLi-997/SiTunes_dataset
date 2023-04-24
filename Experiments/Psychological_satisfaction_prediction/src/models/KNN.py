'''
KNN
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.neighbors import KNeighborsClassifier

class KNN(BaseModel):
	extra_log_args = ['n_neighbors','algorithm']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--n_neighbors',type=int,default=5)
		parser.add_argument('--algorithm',type=str,default='auto')
		parser.add_argument('--leaf_size',type=int,default=30)

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = KNeighborsClassifier(n_neighbors=args.n_neighbors,
								algorithm=args.algorithm, leaf_size=args.leaf_size)