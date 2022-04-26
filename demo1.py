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
def read_csvs(suffix='csv', file_root=''):
    if not file_root:
        file_root = suffix
    result_dfs_list = []
    root = pathlib.Path(file_root)
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
        ax.plot(x, y, z, lw=0.5)
    # ax.legend()
    plt.show()


# 点图航迹带标签
def draw_line_scatters(df_list: List[DataFrame]):
    # mandarin support
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    expand = 2
    ax = plt.figure(figsize=[6.4 * expand, 4.8 * expand], dpi=100.0 / expand).add_subplot(projection='3d')
    for df in df_list:
        df_np = df[['lat', 'lon', 'geoaltitude']].to_numpy()
        x = df_np[:, 0]
        y = df_np[:, 1]
        z = df_np[:, 2]
        # ax.plot(x, y, z, label=u'航迹示意图')
        ax.scatter(x, y, z, marker='o', s=0.05)
    # ax.legend()
    ax.set_xlabel("lat")
    ax.set_ylabel("lon")
    ax.set_zlabel("geoaltitude")
    ax.set_title("航迹点云")
    plt.show()


# draw 3d scatters
def draw_3d_scatters(df_list: List[Union[DataFrame]]):
    # 将点云concat
    df = pd.concat(df_list)
    df = df.dropna()
    X = df[['lat', 'lon', 'geoaltitude']].to_numpy()
    # todo 聚类时加入航向和航速特征并归一化
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    X = np.column_stack([X, labels])
    X = X[X[:, 3] != -1]
    # 类的数量 -> 取出排名前N个类
    ord_dic_list = [{'cluster_index': i, 'amt': len(X[X[:, 3] == i])} for i in range(n_clusters_)]
    ord_dic_list = sorted(ord_dic_list, key=lambda x: x['amt'], reverse=True)
    topk_cluster = [ord_dic_list[i]['cluster_index'] for i in range(50)]  # todo hyperparam
    # 开始绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for cluster_index in topk_cluster:
        temp_nd = X[X[:, 3] == cluster_index]
        xs = temp_nd[:, 0]
        ys = temp_nd[:, 1]
        zs = temp_nd[:, 2]
        m = 'o'
        ax.scatter(xs, ys, zs, marker=m)
    plt.show()


if __name__ == '__main__':
    ############# todo 如果是第一次跑放开下面这行方法注释 ###########
    # split_files(150)  # 切分文件到xlsx目录
    ##########################################################
    # df_list = read_csvs('csv', 'D:\Repo\dbscan_absd\爬取数据')
    df_list = read_csvs('csv', 'D:\Repo\dbscan_absd\爬取数据\M-C17运输-AE1450')
    # draw_3d_lines(df_list)  # 折现航迹
    draw_line_scatters(df_list)  # 散点航迹
    # draw_3d_scatters(df_list) # 聚类散点
