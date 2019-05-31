# scipy.ndimage.imread
File: DeepLearning1/week4/Deep_Neural_Network_Application_Image_Classification/Deep_Neural_Network_Application_v8.ipynb
Part: Part7 Test with your own image
Problem: scipy1.3 does not have function scipy.ndimage.imread

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



# plt.scatter keyword c needs flattened array
File: DeepLearning2/week1/Initialization/init_utils.py
File: DeepLearning2/week2/Optimization/opt_utils.py
Function: plot_decision_boundary

**Error**:  
```python
ValueError: 'c' argument has 1 elements, which is not acceptable for use with 'x' with size 300, 'y' with size 300.
```

**Fix**:  
```python
From: plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral) 

To: plt.scatter(X[0, :], X[1, :], c=y[0,:], cmap=plt.cm.Spectral)
```
