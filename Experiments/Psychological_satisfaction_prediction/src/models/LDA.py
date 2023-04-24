'''
Linear Discriminant Analysis
https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA(BaseModel):
	extra_log_args = ['solver',]

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--solver',type=str,default='svd') # svd, lsqr, eigen

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		self.clf = LinearDiscriminantAnalysis(solver=args.solver)