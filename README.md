# [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning)

![](images/deeplearning-ai.png)

[![Build Status](https://travis-ci.org/bhishanpdl/Deep_Learning_Specialization_Coursera.svg?branch=master)](https://travis-ci.org/bhishanpdl/Deep_Learning_Specialization_Coursera)

Instructor: [Andrew Ng](http://www.andrewng.org/)

# Importrant Notes
1. In course 2, week1, init_utils.py, the function `plot_decision_bounday` has a bug.
```python
From: plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
to: plt.scatter(X[0, :], X[1, :], c=y[0,:], cmap=plt.cm.Spectral)
```

2. Using HTML in markdown cells
In Github markdown cell supports HTML rendering, but Google Colaboraroty does not support it. So use markdown
syntax instead of html syntax to create tables and other markdown codes.
