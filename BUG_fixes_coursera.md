# Course 2: week 1  init_utils.py
There is bug in function `plot_decision_boundary(model,X,y)`. It gives `TypeError: unhashable type: 'numpy.ndarray'`.
To fix this:
```python
plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral) # gives error

plt.scatter(X[0, :], X[1, :], c=y[0,:], cmap=plt.cm.Spectral) # fixes the bug
```
