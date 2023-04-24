# Psychological Satisfaction

Modeling user satisfaction is a core task in recommender systems. 
Previous research on satisfaction focused on users’ likes, dislikes, or ratings of the items. 
However, user satisfaction has a broader meaning, e.g., the users’ emotion changes by receiving recommendations, which we call psychological satisfaction. 
It was impossible to be explored as existing public datasets have no emotion annotations after item consuming. 
With our dataset *SiTunes*, we take the first step to understand users’ psychological satisfaction with music recommendation by predicting their emotion changes.


### Structure
- ``dataset\``:
  - ``data_generator.py``: generates data from the dataset for situational recommendation.
- ``src\``
  - ``main.py``: serves as the entrance of all predictors
  - ``models\``: integrates all traditioanl classifiers into our framework to run with unified configs.
  - ``helpers\``: includes ``BaseReader.py`` to read the dataset, ``configs.py`` to define constant configerations in the dataset (modify it to try different feature combinations), ``utils.py`` to include some basic functions.
- ``script\``: final running commands for each model used in the paper, named with the model names.
