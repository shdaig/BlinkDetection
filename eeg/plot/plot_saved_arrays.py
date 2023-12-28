import matplotlib.pyplot as plt
import numpy as np


f_name_1 = "conv_ae32_1"
f_name_2 = "ae128.npy"

with open(f'saved_arrays/{f_name_1}', 'rb') as f:
    a = np.load(f)

with open(f'saved_arrays/{f_name_2}', 'rb') as f:
    b = np.load(f)

latent_dims = [2, 3, 5, 7, 10, 12, 15, 17, 20, 23, 25, 27, 30, 32]

exp_name = "cp_0"
plt.plot(latent_dims, a, label='Convolutional AE')
plt.plot(latent_dims, b, label='FC AE')
plt.xlabel("latent dim")
plt.ylabel("mse")
plt.legend()
plt.grid(True)
plt.savefig(f'saved_plots/{exp_name}.png')
plt.close()