'''Reference:
	https://github.com/THUwangcy/ReChorus
'''
# -*- coding: UTF-8 -*-

import os
import logging
import torch
import json
import datetime
import numpy as np
import pandas as pd

def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
	linesep = os.linesep
	arg_dict = vars(args)
	keys = [k for k in arg_dict.keys() if k not in exclude_lst]
	values = [arg_dict[k] for k in keys]
	key_title, value_title = 'Arguments', 'Values'
	key_max_len = max(map(lambda x: len(str(x)), keys))
	value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
	key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
	horizon_len = key_max_len + value_max_len + 5
	res_str = linesep + '=' * horizon_len + linesep
	res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
			   + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
	for key in sorted(keys):
		value = arg_dict[key]
		if value is not None:
			key, value = str(key), str(value).replace('\t', '\\t')
			value = value[:max_len-3] + '...' if len(value) > max_len else value
			res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
					   + value + ' ' * (value_max_len - len(value)) + linesep
	res_str += '=' * horizon_len
	return res_str

def get_time():
	return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_metric(result_dict) -> str:
	assert type(result_dict) == dict
	record_metrics = []
	format_str = []
	metrics = [key for key in result_dict.keys()]
	for metric in np.sort(metrics):
		name = metric
		m = result_dict[name]
		if name in record_metrics:
			continue
		if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
			format_str.append('{}:{:<.4f}'.format(name, m))
		elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
			format_str.append('{}:{}'.format(name, m))
		record_metrics.append(name)
	return ','.join(format_str)
