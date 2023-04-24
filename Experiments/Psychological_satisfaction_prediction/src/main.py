import os
import sys
import logging
import argparse
import time
from imp import reload
from tqdm import tqdm
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.metrics import *

from helpers import *
from models import *
from helpers.configs import *


def parse_global_args(parser):
	parser.add_argument('--verbose', type=int, default=logging.INFO,
						help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='',
						help='Logging file path')
	parser.add_argument('--random_seed', type=int, default=0,
						help='Random seed of numpy and pytorch')
	parser.add_argument('--save_prediction',type=int,default=0,
						help='Whether to save the prediction results')
	parser.add_argument('--save_anno',type=str,default='test',
							help='Annotation for saving files.')
	parser.add_argument('--metrics', type=str, default='accuracy,f1,roc_auc,log_loss,rmse,mae',
								help='metrics')
	parser.add_argument('--class_num',type=int,default=2, help='number of class to classify')
	return parser

def evaluate_binary(pred_y, y, args):
	# binary classification
	select_metrics = args.metrics.split(",")
	results = dict()
	for metric in select_metrics:
		if metric in ['accuracy','f1','roc_auc']:
			eval_metric = eval(metric+'_score')
		elif metric == 'log_loss':
			eval_metric = eval(metric)
		elif metric == 'rmse':
			eval_metric = eval('mean_squared_error')
			results[metric] = eval_metric(y,pred_y,squared=False)
		elif metric == 'mae':
			eval_metric = eval('mean_absolute_error')
		else:
			logging.warning("No metric named %s"%(metric))
		if metric not in results:
			results[metric] = eval_metric(y,pred_y)
	return results

def evaluate_multi(pred_y, y, args):
	# multi-class classification
	select_metrics = args.metrics.split(",")
	results = dict()
	for metric in select_metrics:
		if metric in ['accuracy']:
			eval_metric = eval(metric+'_score')
			results[metric] = eval_metric(y, pred_y)
		elif metric == 'macro_f1':
			results[metric] = f1_score(y, pred_y, average='macro')
		elif metric == 'micro_f1':
			results[metric] = f1_score(y, pred_y, average='micro')
		elif metric in ['macro_ap','micro_ap',]:
			encoder = np.eye(args.class_num)
			y_class = encoder[y.astype(int)]
			y_pred_class = encoder[pred_y.astype(int)] 
			if metric == 'macro_ap':
				results[metric] = average_precision_score(y_class, y_pred_class, average='macro')
			elif metric == 'micro_ap':
				results[metric] = average_precision_score(y_class, y_pred_class, average='micro')
		else:
			logging.warning("No metric named %s"%(metric))
	return results


def evaluate(pred_y, y, args):
	if args.class_num == 2:
		results = evaluate_binary(pred_y,y,args)
	else:
		results = evaluate_multi(pred_y,y,args)

	return results

def run():
	logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
	exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
				'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
	logging.info(utils.format_arg_str(args, exclude_lst=exclude))
	
	# Random seed
	np.random.seed(args.random_seed)

	# Read data
	data = reader_name(args,normalize=True)

	# Define model
	model = model_name(args)

	# run the model
	model.train(data.train_X,data.train_y)

	# predict
	for phase, X, y in zip(['train','val','test'],[data.train_X,data.val_X,data.test_X],
							[data.train_y,data.val_y,data.test_y]):
		pred_y = model.predict(X)
		evaluations = evaluate(pred_y, y, args)
		logging.info('%s results -- '%(phase)+utils.format_metric(evaluations))
		if args.save_prediction:
			np.save(os.path.join(args.save_path,"{}_pred.npy".format(phase)),pred_y)


if __name__=="__main__":
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='LR', help='Choose a classification model.')
	init_args, init_extras = init_parser.parse_known_args()
	model_name = eval('{0}.{0}'.format(init_args.model_name))
	reader_name = eval('{0}.{0}'.format(model_name.reader))
	
	# Args
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = model_name.parse_model_args(parser)
	args, extras = parser.parse_known_args()
	logging.info("Extra args: %s"%(str(extras)))


	# Logging configuration
	log_args = [args.dataname, str(args.random_seed),args.save_anno]
	for arg in model_name.extra_log_args:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name = '__'.join(log_args).replace(' ', '__')
	append = "_3class" if args.class_num == 3 else ""
	if args.log_file == '':
		args.log_file = '../logs/{}{}/{}/model.txt'.format(init_args.model_name,append, log_file_name)
	if args.model_path == '':
		args.model_path = '../models/{}{}/{}/model.pt'.format(init_args.model_name,append, log_file_name)
	if args.save_prediction:
		args.save_path = "../models/{}{}/{}".format(init_args.model_name,append,log_file_name)
	os.makedirs(os.path.dirname(args.log_file),exist_ok=True)
	os.makedirs(os.path.dirname(args.model_path),exist_ok=True)
	
	reload(logging)
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	
	logging.info("Save model to %s"%(args.model_path))
	logging.info(init_args)

	# run the models
	run()
