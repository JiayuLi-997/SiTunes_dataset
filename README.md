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



