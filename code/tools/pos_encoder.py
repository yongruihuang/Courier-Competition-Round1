# encoding: utf-8
'''
Created on Mar 15, 2020

@author: Yongrui Huang
'''

import numpy as np

def haversine_array(lat1, lng1, lat2, lng2):
    """
    Haversine距离
    """
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371 # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    """
    曼哈顿距离
    """
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    """
    两个经纬度之间的方位
    """
    AVG_EARTH_RADIUS = 6371 # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def mid_pos(lat1, lng1, lat2, lng2):
    """
    两个经纬度的中位点
    """
    return (lat1 + lat2) / 2, (lng1 + lng2) / 2

def mutual2pos(lat1, lng1, lat2, lng2, prefix=''):
    """
    计算2个经纬度相互信息
    Returns:
        d1 : haversine_array
        d2 : dummy_manhattan_distance
        d3 : bearing_array
        mlat, mlng : mid_pos
    """
    mlat, mlng = mid_pos(lat1, lng1, lat2, lng2)
    
    return {
        '%s_haversine' % (prefix) : haversine_array(lat1, lng1, lat2, lng2),
        '%s_manhattan' % (prefix) : dummy_manhattan_distance(lat1, lng1, lat2, lng2),
        '%s_bearing' % (prefix) : bearing_array(lat1, lng1, lat2, lng2),
        '%s_midlat' % (prefix) : mlat,
        '%s_midlng' % (prefix) : mlng
    }

if __name__ == '__main__':
    pass