# Predicting Heart Disease

For my third module project for Flatiron School, I chose to build a binary classifier to predict [heart disease](https://www.kaggle.com/danimal/heartdiseaseensembleclassifier). I explore a variety of relatively simple classifiers (read: no neural networks)––Support Vector Machines, Decision Trees and Random Forests, AdaBoost and XGBoost, KNN––and fine tune each to upwards of 85% accuracy on test data. My final model, an ensemble [Voting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier), combines some of the best models to achieve 89% accuracy on test data.

## Getting Started
### Contents of Repository

* **analysis.ipynb** is a Jupyter Notebook containing all my analysis and visualizations for the project.
* **Heart_Disease_Data.csv** contains all data from Kaggle's [Heart Disease Ensemble Classifier](https://www.kaggle.com/danimal/heartdiseaseensembleclassifier).
* **images** is a directory containing images used in this README, as well as a Decision Tree visualization (tree.png) produced in my analysis.
* **presentation.pdf** contains my powerpoint presentation for a non-technical audience.

### Prerequisites

The standard packages for data analysis are required–[NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Matplotlib](https://matplotlib.org/)––as well as [pydotplus](https://pypi.org/project/pydotplus/) and [Graphviz](https://graphviz.org/) to make a visualization of a decision tree, [scikit-learn](https://scikit-learn.org/stable/index.html) for a number of classifiers, and [XGBoost](https://xgboost.readthedocs.io/en/latest/). Below are examples of their installations using Anaconda.

```
$ conda install -c anaconda numpy
$ conda install pandas
$ conda install -c conda-forge matplotlib
$ conda install scikit-learn
$ conda install -c conda-forge xgboost
$ conda install -c conda-forge pydotplus
$ conda install -c anaconda graphviz
```


## Built With

[Jupyter Notebook](https://jupyter.org) - Documents containing live code and visualizations.

## Contributing

Due to the nature of the assignment, this project is not open to contributions. If, however, after looking at the project you'd like to give advice to someone new to the field and eager to learn, please reach out to me at [stephen.t.lanier@gmail.com]

## Author

**Stephen Lanier** <br/>
[GitHub](https://github.com/stlanier) <br/>
[Datalingo](https://datalingo.wordpress.com)



## Acknowledgments

<a href="https://flatironschool.com"><img src="images/flatiron.png" width="80" height="40"  alt="Flatiron School Logo"/></a>
Special thanks to Jacob Eli Thomas and Victor Geislinger, my instructors at [Flatiron School](https://flatironschool.com), for their encouragement, instruction, and guidance.

<a href="https://www.kaggle.com"><img src="images/kaggle.png" width="80" height="40"  alt="Kaggle Logo"/></a>
Thanks to [Kaggle](https://www.kaggle.com) for access to data found in [Heart Disease Ensemble Classifier](https://www.kaggle.com/danimal/heartdiseaseensembleclassifier), and particular thanks to [Nathan S. Robinson](https://www.kaggle.com/iamkon/ml-models-performance-on-risk-prediction) for his work on the same dataset: it was beautifully organized, instructive, and a constant source of clarity and inspiration. 
