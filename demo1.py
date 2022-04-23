import random
from typing import List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import glob2
import loguru
import pathlib
from tqdm import tqdm
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

'''simple def impl'''


# read csv
def read_csvs():
    result_dfs_list = []
    root = pathlib.Path('csv')
    files_glob = root.rglob('*.csv')
    for file in files_glob:
        df = pd.read_csv(file)
        result_dfs_list.append(df)
        # print(df)
    return result_dfs_list


# 拆出一定数量的航班
def split_files(total_need=50):
    df = read_csvs()[0]
    df = df.dropna()
    df_gb = df.groupby(['icao24', 'callsign']).size().reset_index().rename(columns={0: '数量'})
    df_gb['数量'] = df_gb['数量'].astype('int')
    df_choose = df_gb[(df_gb['数量'] > 200) & (df_gb['数量'] < 1000)]
    df_choose = df_choose[~df_choose['callsign'].str.contains('  ')]
    df_choose = df_choose[~df_choose['callsign'].str.contains('00000000')]
    df_choose_list = df_choose[['icao24', 'callsign']].to_numpy().tolist()
    df_choose_list = [df_choose_list.pop(random.choice(range(len(df_choose_list)))) for _ in range(total_need * 5)]
    has_finish = 0
    for i, (icao, callsign) in tqdm(enumerate(df_choose_list)):
        if has_finish >= total_need:
            break
        df_i = df[(df['icao24'] == icao) & (df['callsign'] == callsign)]
        # nan_size = len(df_i[['lat', 'lon', 'baroaltitude', 'geoaltitude']].isna())
        # if nan_size / len(df_i) >= 0.2:
        #     continue
        df_i.to_excel('doc/' + '轨迹' + str(i + 1) + '.xlsx')
        has_finish += 1


# draw 3d lines
def draw_3d_lines(df_list: List[Union[DataFrame]]):
    # mandarin support
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #
    plt.rcParams['agg.path.chunksize'] = 1000000
    #
    ax = plt.figure().add_subplot(projection='3d')
    for df in df_list:
        df_np = df[['lat', 'lon', 'geoaltitude']].to_numpy()
        x = df_np[:, 0]
        y = df_np[:, 0]
        z = df_np[:, 0]
        ax.plot(x, y, z, label=u'航迹示意图')
    ax.legend()
    plt.show()


# draw 3d scatters
def draw_3d_scatters(df_list: List[Union[DataFrame]]):
    # 将点云concat

    pass


if __name__ == '__main__':
    split_files()
    df_list = read_csvs()
    draw_3d_lines(df_list)
    pass
