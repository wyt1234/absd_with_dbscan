# absd_with_dbscan
基于dbscan的ads-b三维航迹可视化数据分析

时间
icao24
纬度
速度
航向
垂直
呼号
地面
警报
spi
应答机
气压
高度
地理高度
最后一次更新
最后一次联络

time	
icao24	
lat	
lon	
velocity	
heading	
vertrate	
callsign	
onground	
alert	
spi	
squawk	
baroaltitude	
geoaltitude	
lastposupdate	
lastcontact	
hour


#操作步骤：
#### 1、安装依赖：pip install -r requirements.txt
#### 2、把全量数据（6.csv）放入csv目录下（如果没有则在absd_with_dbscan文件夹下新建一个）
#### 3、创建一个xlsx空目录在absd_with_dbscan文件夹下（如果没有）
#### 4、运行demo1.py（如果是第一次跑放开最后一段split_files方法的注释）