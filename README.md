Table of Contents
=================
   * [<a href="https://www.coursera.org/specializations/deep-learning" rel="nofollow">Deep Learning Specialization on Coursera</a>](#deep-learning-specialization-on-coursera)
   * [BUG Fix](#bug-fix)
   * [Tips](#tips)
   * [Download data from coursera](#download-data-from-coursera)
   * [Course 1. <a href="https://www.coursera.org/learn/neural-networks-deep-learning" rel="nofollow">Neural Networks and Deep Learning</a>](#course-1-neural-networks-and-deep-learning)
   * [Course 2. <a href="https://www.coursera.org/learn/deep-neural-network" rel="nofollow">Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization</a>](#course-2-improving-deep-neural-networks-hyperparameter-tuning-regularization-and-optimization)
   * [Course 3. <a href="https://www.coursera.org/learn/machine-learning-projects" rel="nofollow">Structuring Machine Learning Projects</a>](#course-3-structuring-machine-learning-projects)
   * [Course 4. <a href="https://www.coursera.org/learn/convolutional-neural-networks" rel="nofollow">Convolutional Neural Networks</a>](#course-4-convolutional-neural-networks)
   * [Course 5. <a href="https://www.coursera.org/learn/nlp-sequence-models" rel="nofollow">Sequential Models</a>](#course-5-sequential-models)
   * [Disclaimer](#disclaimer)

# [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning)
Instructor: [Andrew Ng](http://www.andrewng.org/)
Syllabus: [Syllabus](https://www.coursera.org/specializations/deep-learning)

# BUG Fix
- In course 2, week1, init_utils.py, the function plot_decision_bounday has a bug.
```python
From: plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
To  : plt.scatter(X[0, :], X[1, :], c=y[0,:], cmap=plt.cm.Spectral)
```

# Tips
- In some markdown cells of Coursera notebook, HTML is used to render image instead of plain markdown.
  While jupyter-notebook displayed in gihtub supports HTML but Google Colaboratory does not support it.
  So, its best to resort to simple markdown format rather than using direct HTML syntax to render images.
- We do not need to run the whole notebook which may take hours to train the model to submit the assignment,
  we can just finish the assignments and then submit the result.

# Download data from coursera
- Click on File icon on top left corner. Click Open. Then download required data.



# Course 1. [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) 
https://www.coursera.org/learn/neural-networks-deep-learning/home/info


Foundations of Deep Learning:
* Understand the major technology trends driving Deep Learning
* Be able to build, train and apply fully connected deep neural networks 
* Know how to implement efficient (vectorized) neural networks 
* Understand the key parameters in a neural network's architecture 

  
# Course 2. [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network) 
https://www.coursera.org/learn/deep-neural-network/home/info

* Understand industry best-practices for building deep learning applications. 
* Be able to effectively use the common neural network "tricks", including initialization, L2 and dropout regularization, Batch normalization, gradient checking, 
* Be able to implement and apply a variety of optimization algorithms, such as mini-batch gradient descent, Momentum, RMSprop and Adam, and check for their convergence. 
* Understand new best-practices for the deep learning era of how to set up train/dev/test sets and analyze bias/variance
* Be able to implement a neural network in TensorFlow. 

# Course 3. [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects) 
https://www.coursera.org/learn/machine-learning-projects/home/info  

- Understand how to diagnose errors in a machine learning system, and 
- Be able to prioritize the most promising directions for reducing error
- Understand complex ML settings, such as mismatched training/test sets, and comparing to and/or surpassing human-level performance
- Know how to apply end-to-end learning, transfer learning, and multi-task learning

# Course 4. [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks) 
https://www.coursera.org/learn/convolutional-neural-networks/home/info 

* Understand how to build a convolutional neural network, including recent variations such as residual networks.
* Know how to apply convolutional networks to visual detection and recognition tasks.
* Know to use neural style transfer to generate art.
* Be able to apply these algorithms to a variety of image, video, and other 2D or 3D data.

# Course 5. [Sequential Models](https://www.coursera.org/learn/nlp-sequence-models) 
https://www.coursera.org/learn/nlp-sequence-models/home/info

* Understand how to build and train Recurrent Neural Networks (RNNs), and commonly-used variants such as GRUs and LSTMs. 
* Be able to apply sequence models to natural language problems, including text synthesis. 
* Be able to apply sequence models to audio applications, including speech recognition and music synthesis.

# Disclaimer
- All the notebooks are downloaded from Coursera and solved by me. I do not own any ownership or anything on them
  only the exercises in the notebooks are solved my me. All the copyrights are reserved to Coursera or therein 
  mentioned parties.
- These materials involves quizzes and notebooks solutions. These are for my own backups and for future readings.
  No any other readers are encourased to copy from them and dishonor the Coursera Honor Codes. The solutions are
  purely my opinions and may not be true and may fail in future versions of test and future versions of python libraries.
- All the readme files for these deep learning coursera are taken from [Mahmoud Badry et. al.](https://github.com/mbadry1/DeepLearning.ai-Summary)
- 
