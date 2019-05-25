# Course 1: week4 Deep_Neural_Network_Application_Image_Classification Deep_Neural_Network_Application
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

# Then,
from scipy.misc import imread
image = imread(fname, flatten=False)
```



# Course 2: week 1  init_utils.py
There is bug in function `plot_decision_boundary(model,X,y)`. It gives `TypeError: unhashable type: 'numpy.ndarray'`.
To fix this:
```python
plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral) # gives error

plt.scatter(X[0, :], X[1, :], c=y[0,:], cmap=plt.cm.Spectral) # fixes the bug
```
