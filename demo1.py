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
def read_csvs(suffix='csv'):
    result_dfs_list = []
    root = pathlib.Path(suffix)
    files_glob = root.rglob('*.' + suffix)
    for file in tqdm(files_glob):
        if suffix == 'csv':
            df = pd.read_csv(file)
        elif suffix == 'xlsx':
            df = pd.read_excel(file)
        else:
            raise Exception()
        result_dfs_list.append(df)
    return result_dfs_list


# 拆出一定数量的航班
def split_files(total_need=50):
    df = read_csvs('csv')[0]
    # df = df.dropna()
    df_gb = df.groupby(['icao24', 'callsign']).size().reset_index().rename(columns={0: '数量'})
    df_gb['数量'] = df_gb['数量'].astype('int')
    df_choose = df_gb[(df_gb['数量'] > 200) & (df_gb['数量'] < 1000)]
    df_choose = df_choose[~df_choose['callsign'].str.contains('  ')]
    df_choose = df_choose[~df_choose['callsign'].str.contains('00000000')]
    df_choose_list = df_choose[['icao24', 'callsign']].to_numpy().tolist()
    print(f'共有{len(df_choose_list)}个可待选航迹')
    i = 0
    while i < total_need:
        icao, callsign = df_choose_list.pop(random.choice(range(len(df_choose_list))))
        df_i = df[(df['icao24'] == icao) & (df['callsign'] == callsign)]
        df_i = df_i[['lat', 'lon', 'baroaltitude', 'geoaltitude']]
        nan_size = len(df_i[df_i.isna().any(1)])
        if nan_size / len(df_i) >= 0.1:
            continue
        df_i = df_i.interpolate()
        df_i = df_i.reset_index(drop=True)
        df_i.to_excel('xlsx/' + '轨迹' + str(i + 1) + '.xlsx')
        i += 1
        print(f'已保存{i}')


# draw 3d lines
def draw_3d_lines(df_list: List[Union[DataFrame]]):
    # mandarin support
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #
    # plt.rcParams['agg.path.chunksize'] = 1000000
    #
    ax = plt.figure().add_subplot(projection='3d')
    for df in df_list:
        df_np = df[['lat', 'lon', 'geoaltitude']].to_numpy()
        x = df_np[:, 0]
        y = df_np[:, 1]
        z = df_np[:, 2]
        # ax.plot(x, y, z, label=u'航迹示意图')
        ax.plot(x, y, z)
    # ax.legend()
    plt.show()


# draw 3d scatters
def draw_3d_scatters(df_list: List[Union[DataFrame]]):
    # 将点云concat

    pass


if __name__ == '__main__':
    split_files(150)
    df_list = read_csvs('xlsx')
    draw_3d_lines(df_list)
    pass
