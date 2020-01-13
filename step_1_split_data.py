import glob
import os

import numpy as np
import pandas as pd


def split_nsrdb_data():
    os.makedirs('nsrdb_rri', exist_ok=True)

    df = pd.DataFrame()

    length = 300
    fps = glob.glob('nsrdb_gen/rri/*.txt')
    for fp in fps:
        df_one = pd.read_csv(fp, sep='\t')
        df_one.columns = ['left_index', 'left_type', 'rri', 'right_type', 'right_index']

        rri = df_one['rri'].values
        arr_num = np.size(rri) // length
        # arr_list = np.array_split(rri[:arr_num*length], arr_num)
        arr_list = rri[:arr_num * length].reshape(-1, length)
        for arr in arr_list:
            df = df.append({'rri': arr}, ignore_index=True)

    df.to_pickle('nsrdb_rri.pickle')


if __name__ == "__main__":
    split_nsrdb_data()
