import numpy as np
import pandas as pd
import os
import sys
import sklearn

class BaseModel:
	reader='BaseReader'
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--model_path', type=str, default='',
							help='Model save path.')
		return parser

	def __init__(self,args):
		self.model_path = args.model_path
	
	def train(self, X, y):
		self.clf = self.clf.fit(X,y)

	def predict(self, X):
		pred_y = self.clf.predict(X)
		return pred_y
	
	def save(self):
		pass