# Course 1: week4 Deep_Neural_Network_Application_Image_Classification Deep_Neural_Network_Application
**Code**
```
NOTE: This code works for scipy 1.1 but does not work for scipy 1.3.
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

**Error for scipy >=1.3**
```
AttributeError: module 'scipy.ndimage' has no attribute 'imread'
```

**Fix**
```
from matplotlib.pyplot import imread
from skimage.transform import resize

## START CODE HERE ##
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = imread(fname)
num_px = 64
my_image = resize(image, (num_px,num_px), order=1, preserve_range=True).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) 
       + ", your L-layer model predicts a \""
       + classes[int(np.squeeze(my_predicted_image)),]
       .decode("utf-8") +  "\" picture.")
```



# Course 2: week 1  init_utils.py
There is bug in function `plot_decision_boundary(model,X,y)`. It gives `TypeError: unhashable type: 'numpy.ndarray'`.
To fix this:
```python
plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral) # gives error

plt.scatter(X[0, :], X[1, :], c=y[0,:], cmap=plt.cm.Spectral) # fixes the bug
```
