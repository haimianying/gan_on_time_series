import wfdb
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def trans_raw_data():
    for data_dir in ['mitdb', 'nsrdb']:
        dir_rri = '%s_gen/rri' % data_dir
        dir_atr = '%s_gen/atr' % data_dir
        dir_sum = '%s_gen/summary' % data_dir
        dir_desc = '%s_gen/desc' % data_dir

        os.makedirs(dir_rri, exist_ok=True)
        os.makedirs(dir_atr, exist_ok=True)
        os.makedirs(dir_sum, exist_ok=True)
        os.makedirs(dir_desc, exist_ok=True)

        fps = glob.glob(f"{data_dir}/*.hea")
        for fp in fps:
            data_name = fp.replace('.hea', '').split('/')[-1]
            to_save = data_name + '.txt'

            cmd = 'ann2rr -r %s/%s -a atr -i s3 -v -V  -w -W > %s/%s' % (data_dir, data_name, dir_rri, to_save)
            os.system(cmd)

            cmd = 'rdann -r %s/%s -a atr -v -e > %s/%s' % (data_dir, data_name, dir_atr, to_save)
            os.system(cmd)

            cmd = 'sumann -r %s/%s -a atr> %s/%s' % (data_dir, data_name, dir_sum, to_save)
            os.system(cmd)

            cmd = 'wfdbdesc %s/%s > %s/%s' % (data_dir, data_name, dir_desc, to_save)
            os.system(cmd)


if __name__ == "__main__":
    trans_raw_data()
