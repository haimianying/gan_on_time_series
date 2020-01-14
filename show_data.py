import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fps = glob.glob('gen_signal/*.npy')
fp = fps[-1]

arr = np.load(fp)
print(arr.shape)
for one_arr in arr:
    one_arr = one_arr.ravel()
    plt.plot(one_arr, '-b.')
    plt.show()
