import matplotlib.pyplot as plt
import numpy as np

filename = "---.npy"

ear_history = np.load(filename)

print(ear_history)

plt.plot(ear_history)
plt.show()
