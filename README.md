# SiTunes_dataset
This repository is the open source for *SiTunes*,  a situational music recommendation feedback dataset with Physiological and Psychological signals.

## Dataset Introduction
*SiTunes* is a music recommendation dataset that contains rich physiological situation information and psychological feedback signals based on a real-world user study. Data and explanations about the dataset are detailed in ``SiTunes\``.

### Collection Process
*SiTunes* is collected with a three-stage user study, which includes a lab study to collect users’ inherent preference (Stage 1), and two field studies to record situations, preference, and psychological feedback in users’ daily life with traditional recommenders (Stage 2) and situational recommenders (Stage 3).
An illustration of the data collection process is shown below.


![Data Collection Process](./log/_static/Experiment_flow.png)


### Basic Statistics

|         | #User | #Item | #Interaction | Phy. Signals | Psy. Signals | Rating | Psy. Feedback |
|---------|-------|-------|--------------|:------------:|:------------:|:------:|:-------------:|
| Stage 1 | 30    | 25    | 600          |              |       √      |    √   |       √       |
| Stage 2 | 30    | 105   | 897          |       √      |       √      |    √   |       √       |
| Stage 3 | 10    | 217   | 509          |       √      |       √      |    √   |       √       |
| All     | 30    | 307   | 2,006        |       √      |       √      |    √   |       √       |

## Experiments
we propose two tasks and corresponding baseline results on *SiTunes*: Situational music recommendation and psychological satisfaction as recommendation feedback in the resource paper.
Details about the codes and configs of the experiments are shown in ``Experiments\``.

### Situational music recommendation
Integrating physiological and environmental situations, such as weather and activities, enables the design of recommenders that adapt to users' preference shifts in situations. 
For instance, a recommender system can suggest different music when the user is exercising on a sunny day or relaxing at home during a rainstorm.
Here we conducted preliminary experiments with *SiTunes* to explore how it can help with situational music recommendation.

We evaluate the effectiveness of incorporating situational information into recommender systems using three evaluation metrics: AUC for assessing model discrimination ability regardless of class imbalance, MAE for easy interpretability and identifying large discrepancies, and RMSE for sensitivity to extreme errors. 

In our study, we chose to evaluate a set of models, 
including Factorization Machines (FM), Wide\&Deep,  AutoInt, and Deep \& Cross Network Version 2 (DCN V2).
Wide\&Deep and FM are selected as they represent classical and popular baseline approaches in the contextual recommendation.
And AutoInt and DCN V2 represent recent advances by which we aim to assess the possibility of employing state-of-the-art approaches on.
In experiments, we adopted the popular RecBole framework to implement and evaluate all results.

The results of the experiments are shown below 

![Experiments results](./log/_static/Situational_recommendation_experiments_results.png)

The experimental results presented in the table offer compelling evidence for the importance of situational information in recommender systems. Comparing with recommendation results without situation, significant performance improvements are achieved for almost all metrics for all models when situational data is incorporated. Therefore, the situational information provided in SiTnues is significantly helpful for music recommendation tasks.

The observed improvement in AUC, MAE, and RMSE metrics when situational data is used highlights the significance of this information in enhancing the accuracy of rating predictions.

Comparing results in three settings,
the superior performance of models in Setting 2 to Setting 1 illustrates the necessity of real-world data to better understand the impact of situational factors on user preferences, which confirms the need to involve a field study. 
Performance in Setting 3 is also worse than in Setting 2.
The performance decrease may be caused by the distribution discrepancy between Stage 2 and Stage 3 with different backbone recommenders. 
Nevertheless, it worth noting that they are not so comparable as the test sets are distinct.

Furthermore, we observe no significant performance difference between the four models. 
The limited scale of our dataset may cause AutoInt and DCN V2 to not have outstanding performances compared with traditional models.
However, these methods are all used for general context-aware recommendation, and we believe models designed explicitly for situational recommendation will lead to better performance in the future.
suggesting that our dataset might not be sufficiently large to exploit the capabilities of advanced machine learning methods fully. 
We believe that a larger dataset with rich context and situational information could better support these methods and yield more insightful results.
