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

### Task Settings

As an initial exploration, we define the psychological satisfaction prediction as a classification task: Given user ID, music metadata, and situations, we aim to predict if users' emotions will lift, drop or keep after music listening.

For classification ground truth labels, we select to predict the changes in the valence dimension, since positive valence usually indicates psychological satisfaction, while arousal is not always consistent with satisfaction. The dividing point between emotion lifting, keeping, and dropping is ±0.125, 0.125 is chosen since it is half a grid in the two-dimensional diagram used for participants' annotation. We consider such an interval to be distinguishable to the participants.

Three different feature combinations are applied as the situation information during the experiments. 
1. The first is *Objective* (Obj.), which utilizes all objective information as situation features, including time, weather, location, and physiological features collected. The features and pre-process strategies are the same as in Section ``SiTunes\Experiments\Situational_recommendation``.
2. The second is *Subjective* (Sub.), including users' annotated valence and arousal values before listening to music. The subjective features might be helpful since emotions statistically tend to become neutral, i.e., it is unlikely to lift the emotion after listening to music when a user is already in high valence.
3. The third is to concatenate objective and subjective features as a whole, referred to as Obj.+Sub.

### Baseline Models and Evaluation Settings

Four classical classifiers are adopted as baseline models for the experiment, including (1) a linear model, Logistic Regression (LR), (2) the Support Vector Machine (SVM), (3) an ensemble classifier, Random Forest (RF), and (4) a basic neural network, Multi-layer Perceptron (MLP). For all classifiers, user ID, item metadata, and situation features are concatenated as input of the model.

For dataset split, we follow the second and third settings in Section ``SiTunes\Experiments\Situational_recommendation`` for situational music recommendation to evaluate the offline and online performance on the emotion prediction task, respectively. In each setting, we repeat the experiments 10 times and report the average results on the test set. The first setting in Section ``SiTunes\Experiments\Situational_recommendation`` is unavailable for this task because users' emotions ahead of music listening are unavailable in Stage 1.

We adopt accuracy (Acc.), Macro F1 score, Macro Average Precision score (Macro AP), and Micro Average Precision score (Micro AP) as evaluation metrics. We implemented all models with the Scikit-learn toolkit, and tuned the hyper-parameters to find results with the highest accuracy on the validation set. 

Results for the psychological satisfaction prediction task, i.e., mood~(valence) change classification of all three situation combinations in two settings. 
The relative t-test is conducted between adjacent situations of the same model, i.e., Obj. vs. Sub. and Sub. vs. Obj.+Sub. `*`/`**` indicates p-value<0.05/0.01, and the best results are shown in **bold**.

![Experiments results situation](/log/_static/Psychological_satisfaction_prediction_task.png)

### Experimental Results

Comparing results with different kinds of situation features, we find that Obj. situation helps predict mood changes after music listening, as the results are better than random predictions. Classification results with Sub. situations are significantly better than objective ones, which confirms the close relationship between users' emotion before and after music listening, and shows the importance of collecting psychological signals. Finally, Obj. + Sub. situation features achieve the best results on Stage 2, but not Stage 3. The reason might be that the models are trained on Stage 2 data, and Obj. features are effective indicators for datasets with the same distribution as training set, but not as effective as Sub. features for out-of-distribution inference.

