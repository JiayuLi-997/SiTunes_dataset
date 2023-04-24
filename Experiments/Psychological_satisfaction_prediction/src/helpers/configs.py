#coding=utf-8

'''
Definition of constants used in the file.
'''

UID='user_id:token'
IID='item_id:token'
LABEL='mood_improvement:label'

CONTEXT_sub = ['emo_pre_valence:float', 'emo_pre_arousal:float']

CONTEXT_obj = ['morning:float', 'afternoon:float', 'evening:float', 'relative_HB_mean:float',
       'activity_intensity_mean:float', 'activity_step_mean:float','relative_HB_std:float', 
	   'activity_intensity_std:float', 'activity_step_std:float', 'still:float', 'act2still:float',
       'walking:float', 'none:float', 'sleeping:float', 'running:float','weather_sunny:float', 
	   'weather_cloudy:float', 'weather_rainy:float', 'weather2:float', 'weather3:float', 'weather4:float', 
	   'GPS1:float', 'GPS2:float', 'GPS3:float', 'timestamp:float',]

CONTEXT_all = CONTEXT_sub+CONTEXT_obj

