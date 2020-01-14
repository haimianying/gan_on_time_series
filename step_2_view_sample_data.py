import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def view_sample_rri_data(arr):
    for rri in arr:
        hr = 60000 / rri

        fig, axs = plt.subplots(2, 1, figsize=(9, 5))
        axs = axs.ravel()

        ax0 = axs[0]
        ax0.plot(rri, '-b.', label='rri')
        ax0.set_ylim([300, 1500])
        ax0.legend()
        ax0.grid()

        ax1 = axs[1]
        ax1.plot(hr, '-r.', label='hr')
        ax1.set_ylim([40, 150])
        ax1.legend()
        ax1.grid()

        fig.suptitle('sample rri show')
        plt.show()


def main():
    fp = 'nsrdb_rri.pickle'
    df = pd.read_pickle(fp)

    rri = np.vstack(df['rri'].values * 1000)
    train_rri = rri.reshape(rri.shape[0], rri.shape[1], 1)
    print('train rri shape: ', train_rri.shape)
    view_sample_rri_data(rri[10:20])


if __name__ == "__main__":
    main()
