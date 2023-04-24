import numpy as np
import pandas as pd
import os
import sys
import sklearn
from sklearn import preprocessing

from models import *
from helpers.configs import *

# Load data
class BaseReader:

	@staticmethod
	def parse_data_args(parser):
		parser.add_argument('--datadir', type=str, default='../data/',
							help='Input data dir.')
		parser.add_argument('--dataname', type=str, default='basedata',
							help='Choose the name of dataset.')
		parser.add_argument('--load_metadata',type=int,default=1,
							help='Whether to load item metadata.')
		parser.add_argument('--context_column_group',type=str,default='CONTEXT_all')
		return parser

	def __init__(self, args, normalize=True):
		self.datadir = args.datadir
		self.dataname = args.dataname
		context_columns = eval(args.context_column_group) # from config file
		self.load_columns = [UID,IID,LABEL]+context_columns
		self.normalize=normalize
		
		self._load_inter_data() # load interactions
		if args.load_metadata:
			self._load_itemmeta() # load item meta
		self._split_data()
	
	def _load_inter_data(self):
		self.train = pd.read_csv(os.path.join(self.datadir,self.dataname,
											  self.dataname+".train.inter"))[self.load_columns]
		self.val = pd.read_csv(os.path.join(self.datadir,self.dataname,
											self.dataname+".val.inter"))[self.load_columns]
		self.test = pd.read_csv(os.path.join(self.datadir,self.dataname,
											 self.dataname+".test.inter"))[self.load_columns]
		if self.normalize:
			self._normalize_features(self.train,[self.val,self.test])
		
	def _load_itemmeta(self):
		self.item_meta = pd.read_csv(os.path.join(self.datadir,self.dataname,self.dataname+".item"))
		if self.normalize:
			self._normalize_features(self.item_meta,[])
		self.train = self.train.merge(self.item_meta,on=[IID],how="left")
		self.val = self.val.merge(self.item_meta,on=[IID],how="left")
		self.test = self.test.merge(self.item_meta,on=[IID],how="left")
	
	def _normalize_features(self, fit_df, transform_dfs):
		ss_features, enc_features = [],[]
		for col in fit_df:
			if col.split(":")[-1] == 'float':
				ss_features.append(col)
			elif col.split(":")[-1] == 'token' and col not in [UID,IID]:
				enc_features.append(col)
		if len(ss_features):
			scaler = preprocessing.StandardScaler().fit(fit_df[ss_features])
			fit_df[ss_features] = scaler.transform(fit_df[ss_features])
			for df in transform_dfs:
				df[ss_features] = scaler.transform(df[ss_features])
		
		if len(enc_features):
			enc = preprocessing.OneHotEncoder().fit(fit_df[enc_features])
			out_shape = enc.transform(fit_df[enc_features]).shape[1]
			fit_df[['enc_%d'%(i) for i in range(out_shape)]] = enc.transform(fit_df[enc_features]).toarray()
			fit_df.drop(columns=enc_features,inplace=True)
			for df in transform_dfs:
				df[['enc_%d'%(i) for i in range(out_shape)]] = enc.transform(df[enc_features]).toarray()
				df.drop(columns=enc_features,inplace=True)


	def _split_data(self):
		X_columns = self.train.drop(columns=['mood_improvement:label']).columns
		self.train_X = self.train.drop(columns=['mood_improvement:label'])[X_columns].to_numpy()
		self.train_y = self.train['mood_improvement:label']
		self.val_X = self.val.drop(columns=['mood_improvement:label'])[X_columns].to_numpy()
		self.val_y = self.val['mood_improvement:label']
		self.test_X = self.test.drop(columns=['mood_improvement:label'])[X_columns].to_numpy()
		self.test_y = self.test['mood_improvement:label']
		print(self.train_X.shape)
		print(self.val_X.shape)