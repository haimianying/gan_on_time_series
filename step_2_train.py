import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



def view_sample_rri_data(arr):
    for rri in arr:
        hr = 60000 / rri

        fig, ax = plt.subplots()
        ax1 = ax.twinx()

        ax.plot(rri, '-b.', label='rri')
        ax.set_ylim([300, 1500])
        ax1.plot(hr, '-r.', label='hr')
        ax1.set_ylim([40, 120])

        plt.legend()
        plt.suptitle('sample rri show')
        plt.show()


def gen_model():
    pass


def main():
    fp = 'nsrdb_rri.pickle'
    df = pd.read_pickle(fp)

    rri = np.vstack(df['rri'].values)
    train_rri = rri.reshape(rri.shape[0], rri.shape[1], 1)
    print('train rri shape: ', train_rri.shape)
    view_sample_rri_data(rri[10:20])



if __name__ == "__main__":
    main()
