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

2. Rendering HTML in markdown  
In Jupyter-notebook (e.g. in Google Colaboratory), HTML rendering is not supported as like Github Markdown,
so use markdown tables instead of HTML tables to show the summary results.

# Troubleshoots
File: https://github.com/bhishanpdl/Deep_Learning_Specialization_Coursera/blob/master/01_Neural_Networks_and_Deep_Learning/week4/Deep_Neural_Network_Application_Image_Classification/Deep_Neural_Network_Application_v8.ipynb

In First course "Neural_Networks_and_Deep_Learning", in week 4 topic "Deep_Neural_Network_Application_Image_Classification/Deep_Neural_Network_Application_v8.ipynb" at the end part of the notebook, we may get the error `'scipy.ndimage' has no attribute 'imread'` to fix this we need to install pillow library.

**Code**
```
## START CODE HERE ##
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
```

**Error**
```
AttributeError: module 'scipy.ndimage' has no attribute 'imread'
```

**Fix**
```
pip install pillow # using pip
conda install -n myenv conda-forge pillow # for conda environment called myenv
```
