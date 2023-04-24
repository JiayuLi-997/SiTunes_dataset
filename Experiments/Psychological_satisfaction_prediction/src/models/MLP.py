'''
MLP
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
'''
import os
import sys
import sklearn
from models.BaseModel import BaseModel
from sklearn.neural_network import MLPClassifier

class MLP(BaseModel):
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--solver',type=str,default='adam') # adam, sgd, lbfgs
		parser.add_argument('--alpha',type=float,default=1e-4)
		parser.add_argument('--batch_size',type=int,default=200)
		parser.add_argument('--learning_rate',type=float,default=1e-3)
		parser.add_argument('--hidden_layer_sizes',type=str,default='(100,)')
		parser.add_argument('--activation',type=str,default='relu') # relu, logistic, tanh
		parser.add_argument('--max_iter',type=int,default=1000)
		parser.add_argument('--patience',type=int,default=50)

		return BaseModel.parse_model_args(parser)

	def __init__(self,args):
		super().__init__(args)
		hidden_layer_sizes = eval(args.hidden_layer_sizes)
		self.clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=args.activation,
						solver=args.solver,alpha=args.alpha,batch_size=args.batch_size,
						learning_rate_init=args.learning_rate,max_iter=args.max_iter,
						random_state=args.random_seed,
						early_stopping=True,validation_fraction=0.1,n_iter_no_change=args.patience)
	