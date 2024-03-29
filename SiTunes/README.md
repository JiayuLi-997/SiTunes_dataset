# Dataset

We provide all data in *SiTunes* here. The dataset provides situational music recommendation records in a three-stage user study.

We describe the format and meaning of all data provided in each stage as follows (All user ids and item ids are aligned through data in three stages and item metadata):

### Stage 1
**interactions.csv**
- Format: ``user_id, item_id, No., timestamp, rating, duration, emo_valence, emo_arousal``
- The ``No.`` column is numbered separately for each user from 1 to 20.
- ``rating`` is five-point Likert scale. ``emo_valence`` and ``emo_arousal`` are emotions after music listening in two dimensions (range: [-1,1]).

### Stage 2
**interactions.csv**
- Format: ``inter_id, user_id, item_id, timestamp, rating, emo_pre_valence, emo_pre_arousal, emo_post_valence, emo_post_arousal, duration, rec_type``
- ``emo_pre_XXX`` are emotions before music listening, and ``emo_post_XXX`` are emotions after music listening, both in range [-1,1].
- ``rec_type`` are missing for first three users due to system errors and ``duration`` are missing for a few records of first 8 users becuase incorrect recording of start time or end time.

**env.json**
- Format: ``{inter_id:{'time': time_period, 'weather':[weather type, pressure, temperature, humidity], 'GPS': [longitude, latitude, speed] } }``
- `time_period`: Represents different times of day, encoded as follows:
  - `1`: 8 a.m. - 12 p.m.
  - `2`: 12 p.m. - 6 p.m.
  - `3`: 6 p.m. - 2 a.m.
  - *Note*: No records are collected during 2 a.m. to 8 a.m.
- `weather_type`: Indicates the type of weather, categorized as:
  - `0`: Sunny
  - `1`: Cloudy
  - `2`: Rainy
 - ``longitude`` and ``latitude`` are both re-scaled for privacy protection.

**wrist.npy**
- Format: ``[heart rate, activity intensity, activity step, activity type]``
- A matrix of size ``30 x 4`` is aligned to each record with record id ``inter_id``
- ``heart rate`` is normalized for each user for privacy protection.
- ``activity type`` is encoded by:
  - `0`: Still (User is stationary)
  - `1`: Act2Still (Transition from activity to stillness)
  - `2`: Walking
  - `3`: Missing Type (Activity type is not recorded or unavailable)
  - `4`: Lying (User is lying down)
  - `5`: Running

### Stage 3
**interactions.csv**
- Format: ``inter_id, user_id, item_id, timestamp, rating, preference, emo_pre_valence, emo_pre_arousal, emo_post_valence, emo_post_arousal, duration, rec_type``
- ``rating`` is five-point Likert scale. ``preference`` is a continous bar for preference in range [0,100],

**env.json & wrist.npy**
- The format and meaning is the same as in Stage 2.

### music_metadata
**music_info.csv**
- includes all music metadata information with ``item_id``.
