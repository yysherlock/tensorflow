"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    # x: M x N, each row: score, each column: example
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
# score1 is vary from -2.0 to 6.0, score2 is fixed as 1, score3 is fixed as 0.2
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

#b = np.array([[1,2,3],[4,5,6]])
#plt.plot(b, softmax(b), linewidth=2)
#plt.show()

#print x.shape, scores.shape, softmax(scores).shape
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
