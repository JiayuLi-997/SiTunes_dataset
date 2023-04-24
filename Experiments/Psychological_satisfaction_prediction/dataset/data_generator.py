'''
Generate data from the dataset for situational recommendation.
Add labels as mood_improvement:label
'''
import os
import pandas as pd

recommendation_datadir = "../../Situational_recommendation/dataset/"

for setting in [2,3]:
	for seed in range(101,111):
		data_dir = "setting%d-%d"%(setting,seed)
		os.makedirs(data_dir,exist_ok=True)
		# items
		item = pd.read_csv(os.path.join(recommendation_datadir,data_dir,data_dir+'.item'))
		item.to_csv(os.path.join(data_dir,data_dir+'.item'),index=False)
		# interaction
		for stage in ['train','valid','test']:
			inter_name = 'setting%d-%d.%s.inter'%(setting,seed,stage)
			inter = pd.read_csv(os.path.join(recommendation_datadir,data_dir,inter_name))
			inter['valence_delta'] = inter['emo_post_valence:float'] - inter['emo_pre_valence:float']
			inter["mood_improvement:label"] = pd.cut(
                    inter.valence_delta,bins=[-2,-0.125,0.125,2],labels=False)
			inter.drop(columns='valence_delta').to_csv(os.path.join(data_dir,inter_name),index=Fase)