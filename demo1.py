from typing import List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import glob2
import loguru
import pathlib

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

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


# draw 3d lines
def draw_3d_lines(df_list: List[Union[DataFrame]]):
    # mandarin support
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
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
    pass


if __name__ == '__main__':
    df_list = read_csvs()
    draw_3d_lines(df_list)
    pass
