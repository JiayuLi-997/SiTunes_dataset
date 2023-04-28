#!/usr/bin/env python
# coding: utf-8
import os
import json
import gzip
import subprocess
import statistics
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

dataset_dir = "../../../SiTunes"

lab_study_dir = dataset_dir + "/Stage1"
field_study_dir = dataset_dir +"/Stage2"
online_study_dir = dataset_dir + "/Stage3"
music_dir = dataset_dir + "/music_metadata"


# ## Field study (Stage 2)
# ### interractions file
interactions = pd.read_csv(os.path.join(field_study_dir, "interactions.csv"))
inter_df = interactions[['inter_id', 'user_id', 'item_id', 'timestamp', 'rating', 
                         'emo_pre_valence', 'emo_pre_arousal', 'emo_post_valence', 
                         'emo_post_arousal']]                                
inter_df.head()

#Loading the bracelet data 
bracelet = json.load(open(os.path.join(field_study_dir,"env.json")))
d = pd.DataFrame.from_dict(bracelet).T.reset_index()
d.columns = ["inter_id", "time", "weather", "GPS"]
d['inter_id'] = d['inter_id'].astype(int)
inter_df = inter_df.merge(d, on=["inter_id"], how="inner")
inter_df.head()

#Loading the wrist data
wrist_data = np.load(os.path.join(field_study_dir,"wrist.npy"))
item = wrist_data[0]

feature_columns_AT = []
feature_columns_av = []
dominant_activity_types = []
feature_columns_av = []
feature_columns_std = []

for item in wrist_data:
    zipped_item = zip(*item)
    first_four_columns = list(itertools.islice(zipped_item, 4))
    
    column_means = [sum(col) / len(col) for col in first_four_columns[:3]]
    column_stds = [statistics.stdev(col) for col in first_four_columns[:3]]
    
    try:
        fourth_column_mode = statistics.mode(first_four_columns[3])
    except statistics.StatisticsError:
        # Handle the case where there is no unique mode
        fourth_column_mode = first_four_columns[3][0]
    
    feature_columns_av.append(column_means)
    feature_columns_std.append(column_stds)
    dominant_activity_types.append(fourth_column_mode)


means_df = pd.DataFrame(feature_columns_av, columns=['relative_HB_mean', 'activity_intensity_mean', 'activity_step_mean'])
stds_df = pd.DataFrame(feature_columns_std, columns=['relative_HB_std', 'activity_intensity_std', 'activity_step_std'])
act_type_df = pd.DataFrame(dominant_activity_types, columns=['activity_type'])
inter_df = pd.concat([inter_df, means_df, stds_df, act_type_df], axis=1)

inter_df.head()

# Split the "weather" and "GPS" columns into separate columns
inter_df[['weather1', 'weather2', 'weather3', 'weather4']] = pd.DataFrame(inter_df['weather'].to_list(), index=inter_df.index)
inter_df[['GPS1', 'GPS2', 'GPS3']] = pd.DataFrame(inter_df['GPS'].to_list(), index=inter_df.index)
inter_df.drop('weather', axis=1, inplace=True)
inter_df.drop('GPS', axis=1, inplace=True)
inter_df.head()

#One-hot encode the weather1 column
encoded_weather = pd.get_dummies(inter_df['weather1'], prefix='weather1')

# Rename the new columns
encoded_weather.columns = ['weather_sunny', 'weather_cloudy', 'weather_rainy']

insert_position = inter_df.columns.get_loc('weather2')
for idx, column_name in enumerate(encoded_weather.columns[::-1]):
    inter_df.insert(insert_position, column_name, encoded_weather[column_name])
    
inter_df = inter_df.drop('weather1', axis=1)
#One-hot encode the activity_type column
encoded_activity_type = pd.get_dummies(inter_df['activity_type'], prefix='activity_type')

# Rename the new columns
encoded_activity_type.columns = ['still', 'act2still', 'walking', 'none', 'sleeping']

insert_position = inter_df.columns.get_loc('weather_sunny')
for idx, column_name in enumerate(encoded_activity_type.columns[::-1]):
    inter_df.insert(insert_position, column_name, encoded_activity_type[column_name])
 
#inserting a 'running' column with all 0's 
insert_index = inter_df.columns.get_loc('weather_sunny')
inter_df.insert(insert_index, 'running', 0)

inter_df = inter_df.drop('activity_type', axis=1)

#One-hot encode the time column
encoded_time = pd.get_dummies(inter_df['time'], prefix='time')

# Rename the new columns
encoded_time.columns = ['morning', 'afternoon', 'evening']
#print(encoded_time)

insert_position = inter_df.columns.get_loc('relative_HB_mean')
for idx, column_name in enumerate(encoded_time.columns[::-1]):
    inter_df.insert(insert_position, column_name, encoded_time[column_name])
    
inter_df = inter_df.drop('time', axis=1)

print(inter_df.columns)
print(inter_df.isnull().values.any())


# ### User file
user_df = pd.DataFrame({'user_id': range(1, 31)})
user_df.head()


# ### Items file
music_features = pd.read_csv(os.path.join(music_dir,"music_info.csv"))
#music_features.rename(columns={"i_id_c":"music_id"},inplace=True)

#Only retain items that appear in interaction data
useful_meta_df = music_features[music_features['item_id'].isin(inter_df['item_id'])].reset_index(drop=True)
all_items = set(useful_meta_df['item_id'].values.tolist())

#Dropping general_genre column because it has the same info as the general_genre_id
useful_meta_df.drop('general_genre', axis=1, inplace=True)

print(useful_meta_df.isnull().values.any())
print(useful_meta_df.columns)
useful_meta_df.head()


# ### Transform to RecBole format
temp_out_df = inter_df.rename(columns={'user_id':'user_id:token', 'item_id': 'item_id:token', 
                                         'rating':'rating:float','timestamp':'timestamp:float',
                                         
                                         'morning':'morning:float', 'afternoon':'afternoon:float',
                                         'evening':'evening:float', 
                                         
                                         'relative_HB_mean':'relative_HB_mean:float', 'activity_intensity_mean':'activity_intensity_mean:float', 
                                         'activity_step_mean':'activity_step_mean:float', 
                                         
                                         'relative_HB_std':'relative_HB_std:float', 'activity_intensity_std':'activity_intensity_std:float',
                                         'activity_step_std':'activity_step_std:float', 'activity_type_std':'activity_type_std:float', 
                                         
                                         'still':'still:float', 'act2still':'act2still:float', 'walking':'walking:float', 
                                         'none':'none:float', 'sleeping':'sleeping:float', 'running':'running:float',
                                         
                                         'weather_sunny':'weather_sunny:float',
                                         'weather_cloudy':'weather_cloudy:float', 'weather_rainy':'weather_rainy:float',
                                         'weather2':'weather2:float','weather3':'weather3:float', 'weather4':'weather4:float',
                                         
                                         'GPS1':'GPS1:float','GPS2':'GPS2:float','GPS3':'GPS3:float',
                                  
                                         'emo_pre_valence':'emo_pre_valence:float','emo_pre_arousal':'emo_pre_arousal:float',
                                         'emo_post_valence':'emo_post_valence:float', 'emo_post_arousal':'emo_post_arousal:float',
                                        })

#Drop duplicates without considering the emotion columns and interaction id
temp_out_df = temp_out_df.drop_duplicates(subset=temp_out_df.columns.difference(['inter_id', 'emo_pre_valence:float', 'emo_pre_arousal:float', 'emo_post_valence:float', 'emo_post_arousal:float']))

out_df = temp_out_df[['user_id:token', 'item_id:token', 'rating:float', 
                 'morning:float', 'afternoon:float', 'evening:float',
                 'relative_HB_mean:float', 'activity_intensity_mean:float',
                 'activity_step_mean:float', 'relative_HB_std:float', 
                 'activity_intensity_std:float', 'activity_step_std:float', 
                 'still:float', 'act2still:float', 'walking:float',
                 'none:float', 'sleeping:float', 'running:float',
                 'weather_sunny:float', 'weather_cloudy:float', 'weather_rainy:float',
                 'weather2:float', 'weather3:float', 'weather4:float', 
                 
                 'GPS1:float', 'GPS2:float', 'GPS3:float', 'timestamp:float',
                      
                 'emo_pre_valence:float', 'emo_pre_arousal:float',
                 'emo_post_valence:float', 'emo_post_arousal:float',]]

out_df = out_df.drop_duplicates()
out_df = out_df.sort_values(by=['user_id:token'], kind='mergesort').reset_index(drop=True)

print(len(out_df))

user_out_df = user_df.rename(columns={'user_id':'user_id:token'})
user_out_df.head()

item_out_df = useful_meta_df.rename(columns={'item_id':'item_id:token', 'popularity':'popularity:token',
                                             'loudness':'loudness:float', 'danceability':'danceability:float',
                                             'energy':'energy:float', 'key':'key:token', 'speechiness':'speechiness:float',
                                             'acousticness':'acousticness:float','instrumentalness':'instrumentalness:float',
                                             'valence':'valence:float', 'tempo':'tempo:float', 
                                             'general_genre_id':'general_genre_id:token', 'duration':'duration:token',
                                             'F0final_sma_amean':'F0final_sma_amean:float', 'F0final_sma_stddev':'F0final_sma_stddev:float',
                                             'audspec_lengthL1norm_sma_stddev':'audspec_lengthL1norm_sma_stddev:float',
                                             'pcm_RMSenergy_sma_stddev':'pcm_RMSenergy_sma_stddev:float',
                                             'pcm_fftMag_psySharpness_sma_amean':'pcm_fftMag_psySharpness_sma_amean:float',
                                             'pcm_fftMag_psySharpness_sma_stddev':'pcm_fftMag_psySharpness_sma_stddev:float',
                                             'pcm_zcr_sma_amean':'pcm_zcr_sma_amean:float', 'pcm_zcr_sma_stddev':'pcm_zcr_sma_stddev:float'
                                        })
item_out_df['popularity:token'] = item_out_df['popularity:token'].astype(int)
item_out_df['key:token'] = item_out_df['key:token'].astype(int)
item_out_df['duration:token'] = item_out_df['duration:token'].astype(int)
item_out_df = item_out_df[['item_id:token', 'popularity:token', 'loudness:float', 'danceability:float',
                          'energy:float', 'key:token', 'speechiness:float', 'acousticness:float',
                          'instrumentalness:float', 'valence:float', 'tempo:float',
                          'general_genre_id:token', 'duration:token', 'F0final_sma_amean:float',
                          'F0final_sma_stddev:float', 'audspec_lengthL1norm_sma_stddev:float',
                          'pcm_RMSenergy_sma_stddev:float', 'pcm_fftMag_psySharpness_sma_amean:float',
                            'pcm_fftMag_psySharpness_sma_stddev:float', 'pcm_zcr_sma_amean:float', 'pcm_zcr_sma_stddev:float']]
item_out_df = item_out_df.drop_duplicates()
item_out_df = item_out_df.sort_values(by=['item_id:token'], kind='mergesort').reset_index(drop=True)
item_out_df['item_id:token'] = item_out_df['item_id:token'].astype(int)

print(item_out_df.columns)


# ### Randomly split for 10 seeds and save files from stage 2 everywhere

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Iterate through the random seeds
for seed in range(101, 111):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Split the DataFrame by user_id
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for _, group in out_df.groupby('user_id:token'):
        # Shuffle the group
        group = group.sample(frac=1).reset_index(drop=True)

        # Calculate the indices for the splits
        train_idx = int(len(group) * train_ratio)
        val_idx = int(len(group) * (train_ratio + val_ratio))

        # Split the group into train, validation, and test sets
        train = group[:train_idx]
        val = group[train_idx:val_idx]
        test = group[val_idx:]

        # Append the splits to the respective lists
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

    # Concatenate the DataFrames using pandas.concat
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    # Create a folder for the current seed
    folder_name = f"./setting2-{seed}"
    os.makedirs(folder_name, exist_ok=True)
    
    folder_name_case1 = f"./setting1-{seed}"
    os.makedirs(folder_name_case1, exist_ok=True)
    
    folder_name_case3 = f"./setting3-{seed}"
    os.makedirs(folder_name_case3, exist_ok=True)
    

    # Save the DataFrames to their respective files
    train_filename = f"setting2-{seed}.train.inter"
    val_filename = f"setting2-{seed}.valid.inter"
    test_filename = f"setting2-{seed}.test.inter"
    
    test_filename_case1 = f"setting1-{seed}.test.inter"
    
    train_filename_case3 = f"setting3-{seed}.train.inter"
    val_filename_case3 = f"setting3-{seed}.valid.inter"

    #Save train, valid, test of Stage 2 to the case 2 folders 
    train_df.to_csv(os.path.join(folder_name, train_filename), index=False, sep='\t')
    val_df.to_csv(os.path.join(folder_name, val_filename), index=False, sep='\t')
    test_df.to_csv(os.path.join(folder_name, test_filename), index=False, sep='\t')
    
    #Save test of Stage 2 to the case 1 folder 
    test_df.to_csv(os.path.join(folder_name_case1, test_filename_case1), index=False, sep='\t')
    
    #Save train and valid of Stage 2 to the case 3 folder
    train_df.to_csv(os.path.join(folder_name_case3, train_filename_case3), index=False, sep='\t')
    val_df.to_csv(os.path.join(folder_name_case3, val_filename_case3), index=False, sep='\t')

    print(f"DataFrames saved for seed {seed}")

#Saving user and item files everywhere

for seed in range(101, 111):
    #No case specifications means case2
    folder_name = f"./setting2-{seed}" 
    folder_name_case1 = f"./setting1-{seed}"   
    folder_name_case3 = f"./setting3-{seed}"
    
    item_filename = f"setting2-{seed}.item"
    
    user_filename = f"setting2-{seed}.user"
    user_filename_case1 = f"setting1-{seed}.user"
    user_filename_case3 = f"setting3-{seed}.user"

    
    #saving to case2 folders 
    item_out_df.to_csv(os.path.join(folder_name, item_filename), index=False, sep='\t')
    user_out_df.to_csv(os.path.join(folder_name, user_filename), index=False, sep='\t')
    
    #saving to case1 folders 
    user_out_df.to_csv(os.path.join(folder_name_case1, user_filename_case1), index=False, sep='\t')
    
    #saving to case3 folders 
    user_out_df.to_csv(os.path.join(folder_name_case3, user_filename_case3), index=False, sep='\t')
    
    print(f"DataFrames saved for seed {seed}")


# ## Online study (Stage 3)

online_inter = pd.read_csv(os.path.join(online_study_dir, "interactions.csv"))
online_inter_df = online_inter[['inter_id', 'user_id', 'item_id', 'timestamp', 'rating',
                               'emo_pre_valence', 'emo_pre_arousal', 'emo_post_valence', 
                                'emo_post_arousal']]


#Loading the bracelet data 
online_bracelet = json.load(open(os.path.join(online_study_dir,"env.json")))
e = pd.DataFrame.from_dict(online_bracelet).T.reset_index()
e.columns = ["inter_id", "time", "weather", "GPS"]
e['inter_id'] = e['inter_id'].astype(int)
online_inter_df = online_inter_df.merge(d, on=["inter_id"], how="inner")

print(len(online_inter_df))

#Loading the wrist data
online_wrist_data = np.load(os.path.join(online_study_dir,"wrist.npy"), allow_pickle=True)

#encode wrist data activity_type field 
encoding_rules = {'still': 0, 'act2still': 1, 'none': 3, 'running': 5, 'sleep': 4, 'walking': 2}
encoded_wrist_data = online_wrist_data.copy()

for activity, code in encoding_rules.items():
    encoded_wrist_data[..., 3] = np.where(online_wrist_data[..., 3] == activity, code, encoded_wrist_data[..., 3])
wrist_data = encoded_wrist_data

feature_columns_AT = []
feature_columns_av = []
dominant_activity_types = []
feature_columns_av = []
feature_columns_std = []


for item in wrist_data:
    zipped_item = zip(*item)
    first_four_columns = list(itertools.islice(zipped_item, 4))
    
    column_means = [sum(col) / len(col) for col in first_four_columns[:3]]
    column_stds = [statistics.stdev(col) for col in first_four_columns[:3]]
    
    try:
        fourth_column_mode = statistics.mode(first_four_columns[3])
    except statistics.StatisticsError:
        # Handle the case where there is no unique mode
        fourth_column_mode = first_four_columns[3][0]
    
    feature_columns_av.append(column_means)
    feature_columns_std.append(column_stds)
    dominant_activity_types.append(fourth_column_mode)


online_means_df = pd.DataFrame(feature_columns_av, columns=['relative_HB_mean', 'activity_intensity_mean', 'activity_step_mean'])
online_stds_df = pd.DataFrame(feature_columns_std, columns=['relative_HB_std', 'activity_intensity_std', 'activity_step_std'])
online_act_type_df = pd.DataFrame(dominant_activity_types, columns=['activity_type'])

online_inter_df = pd.concat([online_inter_df, online_means_df, online_stds_df, online_act_type_df], axis=1)

# Split the "weather" and "GPS" columns into separate columns
online_inter_df[['weather1', 'weather2', 'weather3', 'weather4']] = pd.DataFrame(online_inter_df['weather'].to_list(), index=online_inter_df.index)
online_inter_df[['GPS1', 'GPS2', 'GPS3']] = pd.DataFrame(online_inter_df['GPS'].to_list(), index=online_inter_df.index)
online_inter_df.drop('weather', axis=1, inplace=True)
online_inter_df.drop('GPS', axis=1, inplace=True)
online_inter_df.head()


#One-hot encode the weather1 column
encoded_weather = pd.get_dummies(online_inter_df['weather1'], prefix='weather1')

# Rename the new columns
encoded_weather.columns = ['weather_sunny', 'weather_cloudy', 'weather_rainy']

insert_position = online_inter_df.columns.get_loc('weather2')
for idx, column_name in enumerate(encoded_weather.columns[::-1]):
    online_inter_df.insert(insert_position, column_name, encoded_weather[column_name])
    
online_inter_df = online_inter_df.drop('weather1', axis=1)
online_inter_df.head()

print(online_inter_df.columns)


#One-hot encode the activity_type column
encoded_activity_type = pd.get_dummies(online_inter_df['activity_type'], prefix='activity_type')

# Rename the new columns
encoded_activity_type.columns = ['still', 'act2still', 'walking', 'none', 'sleeping', 'running']

insert_position = online_inter_df.columns.get_loc('weather_sunny')
for idx, column_name in enumerate(encoded_activity_type.columns[::-1]):
    online_inter_df.insert(insert_position, column_name, encoded_activity_type[column_name])
    
online_inter_df = online_inter_df.drop('activity_type', axis=1)

#One-hot encode the time column
encoded_time = pd.get_dummies(online_inter_df['time'], prefix='time')

# Rename the new columns
encoded_time.columns = ['morning', 'afternoon', 'evening']
#print(encoded_time)

insert_position = online_inter_df.columns.get_loc('relative_HB_mean')
for idx, column_name in enumerate(encoded_time.columns[::-1]):
    online_inter_df.insert(insert_position, column_name, encoded_time[column_name])
    
online_inter_df = online_inter_df.drop('time', axis=1)

print(online_inter_df.columns)
print(inter_df.isnull().values.any())


#Only retain items that appear in interaction data
online_useful_meta_df = music_features[music_features['item_id'].isin(online_inter_df['item_id'])].reset_index(drop=True)
all_items = set(online_useful_meta_df['item_id'].values.tolist())

print(len(online_useful_meta_df))
online_useful_meta_df.head()


# ### Transform to recbole format
online_out_df = online_inter_df.rename(columns={'user_id':'user_id:token', 'item_id': 'item_id:token', 
                                         'rating':'rating:float','timestamp':'timestamp:float',
                                         
                                         'morning':'morning:float', 'afternoon':'afternoon:float',
                                         'evening':'evening:float', 
                                         
                                         'relative_HB_mean':'relative_HB_mean:float', 'activity_intensity_mean':'activity_intensity_mean:float', 
                                         'activity_step_mean':'activity_step_mean:float', 
                                         
                                         'relative_HB_std':'relative_HB_std:float', 'activity_intensity_std':'activity_intensity_std:float',
                                         'activity_step_std':'activity_step_std:float', 'activity_type_std':'activity_type_std:float', 
                                         
                                         'still':'still:float', 'act2still':'act2still:float', 'walking':'walking:float', 
                                         'none':'none:float', 'sleeping':'sleeping:float', 'running':'running:float',
                                         
                                         'weather_sunny':'weather_sunny:float',
                                         'weather_cloudy':'weather_cloudy:float', 'weather_rainy':'weather_rainy:float',
                                         'weather2':'weather2:float','weather3':'weather3:float', 'weather4':'weather4:float',
                                         
                                         'GPS1':'GPS1:float','GPS2':'GPS2:float','GPS3':'GPS3:float',
                                                
                                         'emo_pre_valence':'emo_pre_valence:float', 'emo_pre_arousal':'emo_pre_valence:float',
                                         'emo_post_valence':'emo_post_valence:float', 'emo_post_arousal':'emo_post_arousal:float'
                                        })

online_out_df = online_out_df[['user_id:token', 'item_id:token', 'rating:float', 
                 'morning:float', 'afternoon:float', 'evening:float',
                 'relative_HB_mean:float', 'activity_intensity_mean:float',
                 'activity_step_mean:float', 'relative_HB_std:float', 
                 'activity_intensity_std:float', 'activity_step_std:float', 
                 'still:float', 'act2still:float', 'walking:float',
                 'none:float', 'sleeping:float', 'running:float',
                 'weather_sunny:float', 'weather_cloudy:float', 'weather_rainy:float',
                 'weather2:float', 'weather3:float', 'weather4:float', 
                 
                 'GPS1:float', 'GPS2:float', 'GPS3:float', 'timestamp:float',
                               
                 'emo_pre_valence:float', 'emo_pre_valence:float',
                 'emo_post_valence:float', 'emo_post_arousal:float']]
online_out_df = online_out_df.drop_duplicates()
online_out_df = online_out_df.sort_values(by=['user_id:token'], kind='mergesort').reset_index(drop=True)


online_out_df.columns


len(online_out_df)



online_item_out_df = online_useful_meta_df.rename(columns={'item_id':'item_id:token', 'popularity':'popularity:token',
                                             'loudness':'loudness:float', 'danceability':'danceability:float',
                                             'energy':'energy:float', 'key':'key:token', 'speechiness':'speechiness:float',
                                             'acousticness':'acousticness:float','instrumentalness':'instrumentalness:float',
                                             'valence':'valence:float', 'tempo':'tempo:float', 
                                             'general_genre_id':'general_genre_id:token', 'duration':'duration:token',
                                             'F0final_sma_amean':'F0final_sma_amean:float', 'F0final_sma_stddev':'F0final_sma_stddev:float',
                                             'audspec_lengthL1norm_sma_stddev':'audspec_lengthL1norm_sma_stddev:float',
                                             'pcm_RMSenergy_sma_stddev':'pcm_RMSenergy_sma_stddev:float',
                                             'pcm_fftMag_psySharpness_sma_amean':'pcm_fftMag_psySharpness_sma_amean:float',
                                             'pcm_fftMag_psySharpness_sma_stddev':'pcm_fftMag_psySharpness_sma_stddev:float',
                                             'pcm_zcr_sma_amean':'pcm_zcr_sma_amean:float', 'pcm_zcr_sma_stddev':'pcm_zcr_sma_stddev:float'
                                        })
online_item_out_df['popularity:token'] = online_item_out_df['popularity:token'].astype(int)
online_item_out_df['key:token'] = online_item_out_df['key:token'].astype(int)
online_item_out_df['duration:token'] = online_item_out_df['duration:token'].astype(int)
online_item_out_df = online_item_out_df[['item_id:token', 'popularity:token', 'loudness:float', 'danceability:float',
                          'energy:float', 'key:token', 'speechiness:float', 'acousticness:float',
                          'instrumentalness:float', 'valence:float', 'tempo:float',
                          'general_genre_id:token', 'duration:token', 'F0final_sma_amean:float',
                          'F0final_sma_stddev:float', 'audspec_lengthL1norm_sma_stddev:float',
                          'pcm_RMSenergy_sma_stddev:float', 'pcm_fftMag_psySharpness_sma_amean:float',
                            'pcm_fftMag_psySharpness_sma_stddev:float', 'pcm_zcr_sma_amean:float', 'pcm_zcr_sma_stddev:float']]
online_item_out_df = online_item_out_df.drop_duplicates()
online_item_out_df = online_item_out_df.sort_values(by=['item_id:token'], kind='mergesort').reset_index(drop=True)
online_item_out_df['item_id:token'] = online_item_out_df['item_id:token'].astype(int)



print(len(online_item_out_df))
online_item_out_df.head()


# ### Save across 10 seeds in case3 folders as a test split
# Iterate through the random seeds
for seed in range(101, 111):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    folder_name_case3 = f"./setting3-{seed}"
    test_filename_case3 = f"setting3-{seed}.test.inter"
    item_filename_case3 = f"setting3-{seed}.item"
    
    online_item_out_df.to_csv(os.path.join(folder_name_case3, item_filename_case3), index=False, sep='\t')
    online_out_df.to_csv(os.path.join(folder_name_case3, test_filename_case3), index=False, sep='\t')
    print(f"DataFrames saved for seed {seed}")


# ## Lab study (Stage 1)

lab_inter = pd.read_csv(os.path.join(lab_study_dir, "interactions.csv"))
lab_inter_df = lab_inter[['user_id', 'item_id', 'timestamp', 'rating',
                         'emo_valence', 'emo_arousal']]
# Add a new column named "inter_id" filled with continuous integers
insert_index = lab_inter_df.columns.get_loc('user_id')
inter_id_column = range(1, len(lab_inter_df) + 1)
lab_inter_df.insert(insert_index, 'inter_id', inter_id_column)


print(len(lab_inter_df))

lab_inter_df.head()

#Only retain items that appear in interaction data
lab_useful_meta_df = music_features[music_features['item_id'].isin(lab_inter_df['item_id'])].reset_index(drop=True)
all_items = set(online_useful_meta_df['item_id'].values.tolist())

print(len(lab_useful_meta_df))
lab_useful_meta_df.head()


# ### Transform to RecBole format

lab_out_df = lab_inter_df.rename(columns={'user_id':'user_id:token', 'item_id': 'item_id:token', 
                                         'rating':'rating:float','timestamp':'timestamp:float',
                                         'emo_valence':'emo_valence:float', 'emo_arousal':'emo_arousal:float'})

lab_out_df = lab_out_df[['user_id:token', 'item_id:token', 'rating:float','timestamp:float', 'emo_valence:float', 'emo_arousal:float']]

lab_out_df = lab_out_df.drop_duplicates()
lab_out_df = lab_out_df.sort_values(by=['user_id:token'], kind='mergesort').reset_index(drop=True)

lab_out_df.head()

print(lab_out_df)

lab_item_out_df = lab_useful_meta_df.rename(columns={'item_id':'item_id:token', 'popularity':'popularity:token',
                                             'loudness':'loudness:float', 'danceability':'danceability:float',
                                             'energy':'energy:float', 'key':'key:token', 'speechiness':'speechiness:float',
                                             'acousticness':'acousticness:float','instrumentalness':'instrumentalness:float',
                                             'valence':'valence:float', 'tempo':'tempo:float', 
                                             'general_genre_id':'general_genre_id:token', 'duration':'duration:token',
                                             'F0final_sma_amean':'F0final_sma_amean:float', 'F0final_sma_stddev':'F0final_sma_stddev:float',
                                             'audspec_lengthL1norm_sma_stddev':'audspec_lengthL1norm_sma_stddev:float',
                                             'pcm_RMSenergy_sma_stddev':'pcm_RMSenergy_sma_stddev:float',
                                             'pcm_fftMag_psySharpness_sma_amean':'pcm_fftMag_psySharpness_sma_amean:float',
                                             'pcm_fftMag_psySharpness_sma_stddev':'pcm_fftMag_psySharpness_sma_stddev:float',
                                             'pcm_zcr_sma_amean':'pcm_zcr_sma_amean:float', 'pcm_zcr_sma_stddev':'pcm_zcr_sma_stddev:float'
                                        })
lab_item_out_df['popularity:token'] = lab_item_out_df['popularity:token'].astype(int)
lab_item_out_df['key:token'] = lab_item_out_df['key:token'].astype(int)
lab_item_out_df['duration:token'] = lab_item_out_df['duration:token'].astype(int)
lab_item_out_df = lab_item_out_df[['item_id:token', 'popularity:token', 'loudness:float', 'danceability:float',
                          'energy:float', 'key:token', 'speechiness:float', 'acousticness:float',
                          'instrumentalness:float', 'valence:float', 'tempo:float',
                          'general_genre_id:token', 'duration:token', 'F0final_sma_amean:float',
                          'F0final_sma_stddev:float', 'audspec_lengthL1norm_sma_stddev:float',
                          'pcm_RMSenergy_sma_stddev:float', 'pcm_fftMag_psySharpness_sma_amean:float',
                            'pcm_fftMag_psySharpness_sma_stddev:float', 'pcm_zcr_sma_amean:float', 'pcm_zcr_sma_stddev:float']]
lab_item_out_df = lab_item_out_df.drop_duplicates()
lab_item_out_df = lab_item_out_df.sort_values(by=['item_id:token'], kind='mergesort').reset_index(drop=True)
lab_item_out_df['item_id:token'] = lab_item_out_df['item_id:token'].astype(int)


print(len(lab_item_out_df))
lab_item_out_df.head()


# ### Randomly split into train and valid and save over 10 seeds into case1 folders

# Define the split ratios
train_ratio = 0.8
val_ratio = 0.2

# Iterate through the random seeds
for seed in range(101, 111):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Split the DataFrame by user_id
    train_dfs = []
    val_dfs = []

    for _, group in lab_out_df.groupby('user_id:token'):
        # Shuffle the group
        group = group.sample(frac=1).reset_index(drop=True)

        # Calculate the index for the split
        train_idx = int(len(group) * train_ratio)

        # Split the group into train and validation sets
        train = group[:train_idx]
        val = group[train_idx:]

        # Append the splits to the respective lists
        train_dfs.append(train)
        val_dfs.append(val)

    # Concatenate the DataFrames using pandas.concat
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)

    # Save the DataFrames to their respective files
    folder_name = f"./setting1-{seed}"
    item_filename = f"setting1-{seed}.item"
    train_filename = f"setting1-{seed}.train.inter"
    val_filename = f"setting1-{seed}.valid.inter"
    
    lab_item_out_df.to_csv(os.path.join(folder_name, item_filename), index=False, sep='\t')
    train_df.to_csv(os.path.join(folder_name, train_filename), index=False, sep='\t')
    val_df.to_csv(os.path.join(folder_name, val_filename), index=False, sep='\t')

    print(f"DataFrames saved for seed {seed}")
