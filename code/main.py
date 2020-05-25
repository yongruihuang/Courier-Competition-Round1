#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import datetime
import pickle
import random
import os
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(level=logging.INFO, filename='log_Model', format=LOG_FORMAT)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

from IPython.display import display, HTML
def show_df(df):
    display(HTML(df.to_html()))

VAL_POINT = 65623


# In[2]:


from tools.pos_encoder import *
from tools.time_encoder import *


# In[ ]:


logging.info('start prediction')


# # data sort out

# In[3]:


testB_dataset_path = '../data/eleme_round1_testB/'
def read_a_day(dataset_path, read_date):
    df_action = pd.read_csv('%saction/action_%s.txt' % (dataset_path, read_date.strftime('%Y%m%d')))
    df_courier = pd.read_csv('%scourier/courier_%s.txt' % (dataset_path, read_date.strftime('%Y%m%d')))
    df_distance = pd.read_csv('%sdistance/distance_%s.txt' % (dataset_path, read_date.strftime('%Y%m%d')))
    df_order = pd.read_csv('%sorder/order_%s.txt' % (dataset_path, read_date.strftime('%Y%m%d')))
    return df_action, df_courier, df_distance, df_order

def read_days(start_date, end_date, data_dir):
    cur_date = start_date
    actions = []
    couriers = []
    distances = []
    orders = []
    while cur_date < end_date:
        df_action, df_courier, df_distance, df_order = read_a_day(data_dir, cur_date)
        df_action['date'] = cur_date
        df_courier['date'] = cur_date
        df_distance['date'] = cur_date
        df_order['date'] = cur_date
        
        actions.append(df_action)
        couriers.append(df_courier)
        distances.append(df_distance)
        orders.append(df_order)
        cur_date += datetime.timedelta(days = 1)
    
    df_actions = pd.concat(actions, axis = 0, ignore_index = True)
    df_couriers = pd.concat(couriers, axis = 0, ignore_index = True)    
    df_distances = pd.concat(distances, axis = 0, ignore_index = True)    
    df_orders = pd.concat(orders, axis = 0, ignore_index = True) 
    return df_actions, df_couriers, df_distances, df_orders

df_actions_testB, df_couriers_testB, df_distances_testB, df_orders_testB = read_days(datetime.datetime(2020, 3, 1), datetime.datetime(2020, 3, 7), testB_dataset_path)


# In[4]:


def get_map(df):
    """
    将数据映射成字典，可通过
        mp[date][courier_id][wave_index]
    访问

    """
    last_date, last_courier, last_wave = df.at[0, 'date'], df.at[0, 'courier_id'], df.at[0, 'wave_index']
    last_i = 0
    mp_date, mp_courier, mp_wave = {}, {}, {}
    key_list = []
    for i in range(1, df.shape[0]):
        cur_wave = df.at[i, 'wave_index']
        cur_courier = df.at[i, 'courier_id']
        cur_date = df.at[i, 'date']

        if last_wave != cur_wave or (last_wave == cur_wave and last_courier != cur_courier):
            mp_wave[last_wave] = df.iloc[last_i : i]
            last_i = i
            key_list.append([last_date, last_courier, last_wave])
            last_wave = cur_wave

        if last_courier != cur_courier or (last_courier == cur_courier and last_date != cur_date):
            mp_courier[last_courier] = mp_wave.copy()
            mp_wave = {}
            last_courier = cur_courier

        if last_date != cur_date:
            mp_date[last_date] = mp_courier.copy()
            mp_courier = {}
            last_date = cur_date
    
    mp_wave[last_wave] = df.iloc[last_i : ]
    key_list.append([last_date, last_courier, last_wave])
    mp_courier[last_courier] = mp_wave.copy()
    mp_date[last_date] = mp_courier.copy()
    mp_courier = {}
    
    return mp_date, key_list

def get_map_distance_detail(mp_distance, key_list):
    mp_distance_detail = {}
    for (date, courier, wave_idx) in key_list:
        if date not in mp_distance_detail:
            mp_distance_detail[date] = {}
        if courier not in mp_distance_detail[date]:
            mp_distance_detail[date][courier] = {}
        if wave_idx not in mp_distance_detail[date][courier]:
            mp_distance_detail[date][courier][wave_idx] = {}
        
        df_a_distance = mp_distance[date][courier][wave_idx]
        
        gby_tracking = df_a_distance.groupby('tracking_id')
        for tracking_id, df_track_id in gby_tracking:
            mp_distance_detail[date][courier][wave_idx][tracking_id] = {}
            gby_target_tracking = df_track_id.groupby('target_tracking_id')
            for target_trackingid, df_target_tracking in gby_target_tracking:
                mp_distance_detail[date][courier][wave_idx][tracking_id][target_trackingid] = df_target_tracking
            
    return mp_distance_detail

start = time.time()
mp_action_testB, action_key_list_testB = get_map(df_actions_testB)
mp_distance_testB, distance_key_list_testB = get_map(df_distances_testB)
mp_order_testB, order_key_list_testB = get_map(df_orders_testB)
mp_distance_detail_testB = get_map_distance_detail(mp_distance_testB, action_key_list_testB)
time.time() - start


# In[5]:


testB_dataset_mp = {
    'key_list' : action_key_list_testB,
    'mp_action' : mp_action_testB,
    'mp_distance' : mp_distance_testB,
    'mp_order' : mp_order_testB,
    'mp_distance_detail' : mp_distance_detail_testB
}


# # build dataset

# In[6]:


logging.info('start')
key_list_testB = testB_dataset_mp['key_list']
testB_know_lens, testB_lens, testB_impossible_idxs = [], [], []
for (date, courier, wave) in key_list_testB:
    df_action = mp_action_testB[date][courier][wave]
    testB_lens.append(df_action.shape[0])
    df_konw = df_action.query('expect_time != 0')
    df_unkonw = df_action.query('expect_time == 0')
    testB_know_lens.append(df_konw.shape[0])
    impossible_idx = []
    konw_tracking_id = set(df_konw['tracking_id'])
    for i, idx in enumerate(df_unkonw.index):
        if df_unkonw.at[idx, 'tracking_id'] not in konw_tracking_id and df_unkonw.at[idx, 'action_type'] == 'DELIVERY':
            impossible_idx.append(i)
    testB_impossible_idxs.append(impossible_idx)

df_testB_info = pd.DataFrame()
df_testB_info['know_lens'] = testB_know_lens
df_testB_info['lens'] = testB_lens
df_testB_info['impossible_idxs'] = testB_impossible_idxs
logging.info('end')


# # generate_test_data_sample

# In[7]:


courier_delay_features = ['pickup_delay_rate', 'delivery_delay_rate', 'pickup_delay_time_avg', 'delivery_delay_time_avg',
                         'delivery_delay_count', 'pickup_delay_count']

pickle_path = '../user_data/generate_train_test_courier_feature/mp_couriers_features_testB.pickle'
with open(pickle_path, 'rb') as f:
    mp_couriers_features_testB = pickle.load(f)


# In[8]:



mp_info_dict_testB = {
    'action' : mp_action_testB,
    'distance' : mp_distance_detail_testB,
    'order' : mp_order_testB,
    'couriers' : mp_couriers_features_testB
}

data_info_dict_testB = {
    'konw_lens' : df_testB_info['know_lens'].values,
    'full_lens' : df_testB_info['lens'].values,
    'impossible_idxs' : df_testB_info['impossible_idxs'].values
}

mp_action_type = {'PICKUP' : 0, 'DELIVERY' : 1}
mp_weather = {'正常天气' : 0, '轻微恶劣天气' : 1, '恶劣天气' : 2, '极恶劣天气' : 3}
def get_static_action_feature_dict(prefix, se_a_action, df_a_order, df_a_distance):
    feature_dict = {}
    feature_dict[prefix + '_action_type'] = mp_action_type[se_a_action.action_type]
    feature_dict[prefix + '_weather'] = mp_weather[df_a_order.loc[se_a_action.tracking_id]['weather_grade']]
    se_lng_lat = df_a_order.loc[se_a_action.tracking_id][['pick_lng', 'pick_lat', 'deliver_lng', 'deliver_lat']]
    se_lng_lat.index = prefix + '_' + se_lng_lat.index
    feature_dict.update(se_lng_lat.to_dict())
    
    self_row = df_a_distance[se_a_action.tracking_id][se_a_action.tracking_id].query('source_type == "PICKUP" & target_type == "DELIVERY"')
    feature_dict[prefix + '_self_p_d_distance'] = float(self_row['grid_distance'])
    return feature_dict

mp_row = {'ASSIGN' : 0, 'DELIVERY' : 1, 'PICKUP' : 2}
def get_cross_action_feature_dict(last_action, cur_action, df_a_order, df_a_distance, speed):
    features_dict = {}
    #action
    features_dict['same_tracking_id'] = (last_action.tracking_id == cur_action.tracking_id)
    #order
    features_dict['cur_pd_sub_last_time'] = df_a_order.loc[cur_action.tracking_id]['promise_deliver_time'] - last_action['expect_time']
    features_dict['cur_ep_sub_last_time'] = df_a_order.loc[cur_action.tracking_id]['estimate_pick_time'] - last_action['expect_time']
    features_dict['cur_assigned_sub_last_time'] = last_action['expect_time'] - df_a_order.loc[cur_action.tracking_id]['assigned_time']
    features_dict['last_assigned_sub_last_time'] = last_action['expect_time'] - df_a_order.loc[last_action.tracking_id]['assigned_time']
    
    features_dict['cur_create_sub_last_time'] = last_action['expect_time'] - df_a_order.loc[cur_action.tracking_id]['create_time']
    features_dict['last_create_sub_last_time'] = last_action['expect_time'] - df_a_order.loc[last_action.tracking_id]['create_time']
    features_dict['cur_confirm_sub_last_time'] = last_action['expect_time'] - df_a_order.loc[cur_action.tracking_id]['confirm_time']
    features_dict['last_confirm_sub_last_time'] = last_action['expect_time'] - df_a_order.loc[last_action.tracking_id]['confirm_time']

    #distance
    df_a_distance_relation = df_a_distance[last_action.tracking_id][cur_action.tracking_id]
    df_a_distance_relation = df_a_distance_relation.sort_values(by = ['source_type', 'target_type'])    

    if features_dict['same_tracking_id']:
        idx = mp_row[last_action.action_type] * 2 + mp_row[cur_action.action_type]
        features_dict['distance_a_a'] = 1.
        features_dict['distance_a_d'] = df_a_distance_relation.iloc[0]['grid_distance'] 
        features_dict['distance_a_p'] = df_a_distance_relation.iloc[1]['grid_distance'] 
        features_dict['distance_d_a'] = df_a_distance_relation.iloc[2]['grid_distance'] 
        features_dict['distance_d_d'] = 1.
        features_dict['distance_d_p'] = df_a_distance_relation.iloc[3]['grid_distance'] 
        features_dict['distance_p_a'] = df_a_distance_relation.iloc[4]['grid_distance'] 
        features_dict['distance_p_d'] = df_a_distance_relation.iloc[5]['grid_distance'] 
        features_dict['distance_p_p'] = 1.
    else:
        idx = mp_row[last_action.action_type] * 3 + mp_row[cur_action.action_type]
        features_dict['distance_a_a'] = df_a_distance_relation.iloc[0]['grid_distance'] 
        features_dict['distance_a_d'] = df_a_distance_relation.iloc[1]['grid_distance'] 
        features_dict['distance_a_p'] = df_a_distance_relation.iloc[2]['grid_distance'] 
        features_dict['distance_d_a'] = df_a_distance_relation.iloc[3]['grid_distance'] 
        features_dict['distance_d_d'] = df_a_distance_relation.iloc[4]['grid_distance'] 
        features_dict['distance_d_p'] = df_a_distance_relation.iloc[5]['grid_distance'] 
        features_dict['distance_p_a'] = df_a_distance_relation.iloc[6]['grid_distance'] 
        features_dict['distance_p_d'] = df_a_distance_relation.iloc[7]['grid_distance'] 
        features_dict['distance_p_p'] = df_a_distance_relation.iloc[8]['grid_distance'] 

    a_distance_row = df_a_distance_relation.iloc[idx]
    features_dict['grid_distance'] = a_distance_row['grid_distance']
    
    pos_dict = a_distance_row[['source_lng', 'source_lat', 'target_lng', 'target_lat']].to_dict()
    pos_mutual_dict = mutual2pos(pos_dict['source_lat'], pos_dict['source_lng'], pos_dict['target_lat'], pos_dict['target_lng'], 'last_cur_position')
    features_dict.update(pos_dict)
    features_dict.update(pos_mutual_dict)
    
    #estimate time
    features_dict['cur_action_estimate_time'] = features_dict['grid_distance'] / speed
#     if cur_action.action_type == "PICKUP":
    time_diff_pickup =  features_dict['cur_ep_sub_last_time'] - features_dict['cur_action_estimate_time']
#     else:
    time_diff_delivery = features_dict['cur_pd_sub_last_time'] - features_dict['cur_action_estimate_time']
    
    features_dict["estimate_time_diff_pickup"] = time_diff_pickup
    features_dict["pickup_estimate_time_in_0min"] = time_diff_pickup < 0
    features_dict["pickup_estimate_time_in_5min"] = time_diff_pickup >= 0 and time_diff_pickup < 60 * 5
    features_dict["pickup_estimate_time_in_15min"] = time_diff_pickup >= 60 * 5 and time_diff_pickup < 60 * 15 
    features_dict["pickup_estimate_time_in_45min"] = time_diff_pickup >= 60 * 15 and time_diff_pickup < 60 * 45
    features_dict["pickup_estimate_time_in_120min"] = time_diff_pickup >= 60 * 45 and time_diff_pickup < 60 * 120
    features_dict["pickup_estimate_time_exceed_120min"] = time_diff_pickup >= 120 * 60
    
    features_dict["estimate_time_diff_delivery"] = time_diff_delivery
    features_dict["delivery_estimate_time_in_0min"] = time_diff_delivery < 0
    features_dict["delivery_estimate_time_in_5min"] = time_diff_delivery >= 0 and time_diff_delivery < 60 * 5
    features_dict["delivery_estimate_time_in_15min"] = time_diff_delivery >= 60 * 5 and time_diff_delivery < 60 * 15 
    features_dict["delivery_estimate_time_in_45min"] = time_diff_delivery >= 60 * 15 and time_diff_delivery < 60 * 45
    features_dict["delivery_estimate_time_in_120min"] = time_diff_delivery >= 60 * 45 and time_diff_delivery < 60 * 120
    features_dict["delivery_estimate_time_exceed_120min"] = time_diff_delivery >= 120 * 60

    
    return features_dict

def softmax_np(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def get_agg_action_feature_dict(tracking_ids, action_types, cur_action, last_time,                                   df_a_action, df_a_order, df_a_distance, prefix):
    
    cur_tracking_id = cur_action.tracking_id
    cur_action_type = cur_action.action_type
    
    agg_feature_list = []
    for i in range(len(tracking_ids)):
        agg_feature_dict = {}
        df_a_distance_relation = df_a_distance[cur_tracking_id][tracking_ids[i]]
        next_action_type = action_types[i]
        query_row = df_a_distance_relation.query('source_type == @cur_action_type & target_type == @next_action_type')
        if query_row.shape[0] == 0:
            continue
        else:
            se_query_row = query_row.iloc[0]
        pos_dict = se_query_row[['source_lng', 'source_lat', 'target_lng', 'target_lat']].to_dict()
        pos_mutual_dict = mutual2pos(pos_dict['source_lat'], pos_dict['source_lng'], pos_dict['target_lat'], pos_dict['target_lng'], prefix)
        agg_feature_dict.update(pos_mutual_dict)
        agg_feature_dict[prefix + '_grid_distance'] = se_query_row['grid_distance']
        agg_feature_dict[prefix + '_create_time_sub_last'] = last_time - df_a_order.loc[tracking_ids[i]].create_time 
        agg_feature_dict[prefix + '_confirm_time_sub_last'] = last_time - df_a_order.loc[tracking_ids[i]].confirm_time
        agg_feature_dict[prefix + '_assigned_time_sub_last'] = last_time - df_a_order.loc[tracking_ids[i]].assigned_time
        agg_feature_dict[prefix + '_promise_deliver_time_sub_last'] = df_a_order.loc[tracking_ids[i]].promise_deliver_time - last_time
        agg_feature_dict[prefix + '_estimate_pick_time_sub_last'] = df_a_order.loc[tracking_ids[i]].estimate_pick_time - last_time

        agg_feature_list.append(agg_feature_dict)
    
    df_agg_feature = pd.DataFrame(agg_feature_list)
        
    feature_dict = {}
    se_mean = df_agg_feature.mean()
    se_mean.index = se_mean.index + '_mean'
    feature_dict.update(se_mean.to_dict())
    
    se_min = df_agg_feature.min()
    se_min.index = se_min.index + '_min'
    feature_dict.update(se_min.to_dict())

    se_max = df_agg_feature.max()
    se_max.index = se_max.index + '_max'
    feature_dict.update(se_max.to_dict())
    
#     se_dist_weight = df_agg_feature.mul(softmax_np(1./df_agg_feature[prefix + '_haversine']), axis=0).sum()
#     se_dist_weight.index = se_dist_weight.index + '_haversine_weight'
#     feature_dict.update(se_dist_weight.to_dict())
    
#     se_time_weight = df_agg_feature.mul(softmax_np(1./df_agg_feature[prefix + '_promise_deliver_time_sub_last']), axis=0).sum()
#     se_time_weight.index = se_time_weight.index + '_pd_time_weight'
#     feature_dict.update(se_time_weight.to_dict())
    
    return feature_dict

def a_feature_dict(i, j, df_a_action, start_action_expect_time, cur_action, last_action, know_len, full_len,                    se_a_couier, df_a_order, df_a_distance, last_load, unknow_tracking_ids, unknow_action_types,                  know_tracking_ids, know_action_types):
    
    cur_action = df_a_action.iloc[j]
    features_dict = {}            

    #generate features(can not be used)
    features_dict['origin_i'] = i
    features_dict['target_position'] = j
    features_dict['start_action_expect_time'] = start_action_expect_time
    features_dict['last_action_expect_time'] = last_action.expect_time
    features_dict['cur_action_expect_time'] = cur_action.expect_time

    #time feature
    last_action_time_feature_dict = time_vector(last_action.expect_time, 'last_action_time')
    features_dict.update(last_action_time_feature_dict)


    #i fix features
    features_dict['know_lens'] = know_len
    features_dict['full_lens'] = full_len
    features_dict.update(se_a_couier[['level', 'speed', 'max_load'] + courier_delay_features])

    #last action features
    last_action_static_feature = get_static_action_feature_dict('last', last_action, df_a_order, df_a_distance)
    features_dict.update(last_action_static_feature)
    features_dict['last_load'] = last_load

    #cur action features
    cur_action_static_feature = get_static_action_feature_dict('cur', cur_action, df_a_order, df_a_distance)
    features_dict.update(cur_action_static_feature)
    if cur_action.action_type == "PICKUP":
        features_dict['cur_load'] = last_load + 1
    else:
        features_dict['cur_load'] = last_load - 1
    features_dict['over_load'] = features_dict['cur_load'] > features_dict['max_load']

    #cross feature
    cross_features = get_cross_action_feature_dict(last_action, cur_action, df_a_order, df_a_distance, features_dict['speed'])
    features_dict.update(cross_features)
    
    #agg feature
    #future feature
    future_features = get_agg_action_feature_dict(unknow_tracking_ids, unknow_action_types, cur_action, last_action.expect_time,                                                      df_a_action, df_a_order, df_a_distance, 'future_agg')
#     pass feature
    pass_features = get_agg_action_feature_dict(know_tracking_ids, know_action_types, cur_action, last_action.expect_time,                                                     df_a_action, df_a_order, df_a_distance, 'pass_agg')
    
    #all feature
    all_features = get_agg_action_feature_dict(list(df_a_action.tracking_id), list(df_a_action.action_type), cur_action, last_action.expect_time,                                                     df_a_action, df_a_order, df_a_distance, 'all_agg')
    
    features_dict.update(future_features)
    features_dict.update(pass_features)
    features_dict.update(all_features)
  
    return features_dict


def generate_gbdt_df(key_list, mp_info_dict, data_info_dict):
    mp_action, mp_distance, mp_order, mp_couriers = mp_info_dict['action'],  mp_info_dict['distance'],  mp_info_dict['order'],  mp_info_dict['couriers']
    know_lens, full_lens, impossible_idxs = data_info_dict['konw_lens'], data_info_dict['full_lens'], data_info_dict['impossible_idxs']
    features_dict_list = []
    for i in range(len(key_list)):
        if (i+1) % 1000 == 0:
            logging.info('%d' % i)
        
        date, courier, wave_idx = key_list[i]
        df_a_action = mp_action[date][courier][wave_idx]
        df_a_distance = mp_distance[date][courier][wave_idx]
        df_a_order = mp_order[date][courier][wave_idx]
        df_a_order.index = df_a_order.tracking_id
        se_a_couier = mp_couriers[courier][date]        
        impossible_idx_set = set(impossible_idxs[i])
        
        start_action_expect_time = df_a_action.iloc[0].expect_time
        last_action = df_a_action.iloc[know_lens[i] - 1]
        pickup_num = df_a_action.query('action_type == "PICKUP"').shape[0] 
        delivery_num = know_lens[i] - pickup_num
        last_load = pickup_num - delivery_num 
        
        df_unknow_action = df_a_action.iloc[know_lens[i] : full_lens[i]]
        unknow_tracking_ids, unknow_action_types = list(df_unknow_action.tracking_id), list(df_unknow_action.action_type)
        
        df_know_action = df_a_action.iloc[: know_lens[i]]
        know_tracking_ids, know_action_types = list(df_know_action.tracking_id), list(df_know_action.action_type)

        for j in range(know_lens[i], full_lens[i]):
            if j - know_lens[i] in impossible_idx_set:
                continue
                
            cur_action = df_a_action.iloc[j]
            know_len, full_len = know_lens[i], full_lens[i]
            features_dict = a_feature_dict(i, j, df_a_action, start_action_expect_time, cur_action, last_action, know_len, full_len,                   se_a_couier, df_a_order, df_a_distance, last_load, unknow_tracking_ids, unknow_action_types, know_tracking_ids, know_action_types)
            features_dict_list.append(features_dict)
    
    df_features = pd.DataFrame(features_dict_list)
    return df_features


# In[9]:


df_features_testB = generate_gbdt_df(key_list_testB, mp_info_dict_testB, data_info_dict_testB)


# # makepair_gbdt

# In[10]:


weight_names = []
pass_names = []
all_agg_names = []
for name in df_features_testB.columns:
    if '_weight' in name:
        weight_names.append(name)
    if 'pass_agg' in name:
        pass_names.append(name)
    if 'all_agg' in name:
        all_agg_names.append(name)

        
not_features = ['origin_i', 'target_position', 'start_action_expect_time', 'last_action_expect_time', 'cur_action_expect_time',
                'last_action_time_hour', 'last_action_time_minute', 'last_action_time_second', 'last_action_time_weekday', 
                'last_action_time_is_weekend', 'last_action_time_is_worktime', 'know_lens', 'full_lens', 'level', 'speed', 'max_load',
                'pickup_delay_rate', 'delivery_delay_rate', 'pickup_delay_time_avg', 'delivery_delay_time_avg', 'delivery_delay_count', 'pickup_delay_count',
                'last_action_type', 'last_weather', 'last_pick_lng', 'last_pick_lat', 'last_deliver_lng', 'last_deliver_lat', 'last_self_p_d_distance',
                'last_load',
               ] + weight_names + pass_names

features_name = sorted( list( set(df_features_testB.columns) - set(not_features)) )
len_features_name = len(features_name)
len_pair_features = 0

pair_feature_name = list(map(lambda x : 'left_' + x, features_name) ) + list(map(lambda x : 'right_' + x, features_name) ) 

cat_feature_name = ['cur_action_type', 'cur_weather', 'delivery_estimate_time_exceed_120min', 'delivery_estimate_time_in_0min',
                    'delivery_estimate_time_in_120min', 'delivery_estimate_time_in_15min', 'delivery_estimate_time_in_45min', 
                    'delivery_estimate_time_in_5min', 'same_tracking_id']

cat_feature_name = list(set(cat_feature_name) - set(not_features))
pair_cat_feature_name = list(map(lambda x : 'left_' + x, cat_feature_name) ) + list(map(lambda x : 'right_' + x, cat_feature_name) )

logging.info('origin feature lens: %d' % len_features_name)
 

def build_pair(se_1, se_2):
    
    pair_np = np.zeros((se_1.shape[0] * 2 + len_pair_features))
    pair_np[ : se_1.shape[0] * 2] = np.concatenate([se_1.values, se_2.values])

    return pair_np

def apply_pairs_train(df):
#     show_df(df)
    df_feature = df[features_name]
    right_se = df_feature.iloc[0]
    n_sample = (df_feature.shape[0] - 1) * 2
    n_feature = len_features_name * 2 + len_pair_features

    sample_np = np.zeros((n_sample * 2, n_feature))
    labels = np.zeros((n_sample * 2,))
    p = 0
    for i in range(1, df.shape[0]):
        wrong_se = df_feature.iloc[i]
        
        sample_np[p] = build_pair(right_se, wrong_se)
        labels[p] = 1
        p += 1
        
        sample_np[p] = build_pair(wrong_se, right_se)
        labels[p] = 0
        p += 1
        
        
    return sample_np[:p], labels[:p]

def apply_pairs_train_data_arugment(df_arugment):
    
    gby_know_len = df_arugment.groupby('know_lens')
    n_sample = 0
    n_feature = len_features_name * 2 + len_pair_features

    for know_len, df in gby_know_len:
        n_sample += df.shape[0] - 1
    n_pair = n_sample * (n_sample + 1)
    
    sample_np = np.zeros((n_pair, n_feature))
    labels = np.zeros((n_pair,))
    p = 0

    for know_len, df in gby_know_len:
        
        df_feature = df[features_name]
        right_se = df_feature.iloc[0]
                
        for i in range(1, df.shape[0]):
            wrong_se = df_feature.iloc[i]

            sample_np[p] = build_pair(right_se, wrong_se)
            labels[p] = 1
            p += 1

            sample_np[p] = build_pair(wrong_se, right_se)
            labels[p] = 0
            p += 1
            
        
    return sample_np[:p], labels[:p]


def apply_pair_test(df):
    df_feature = df[features_name]
    n_sample = int(df_feature.shape[0] * (df_feature.shape[0] - 1))
    n_feature = df_feature.shape[1] * 2 + len_pair_features
    samples_np = np.zeros((n_sample, n_feature))
    positions = np.zeros((n_sample, 2))
    p = 0
    
    for i in range(df.shape[0]):
        for j in range(i + 1, df.shape[0]):
            samples_np[p] = build_pair(df_feature.iloc[i], df_feature.iloc[j])
            positions[p] = np.array([df.iloc[i].target_position, df.iloc[j].target_position])
            p += 1
            
            samples_np[p] = build_pair(df_feature.iloc[j], df_feature.iloc[i])
            positions[p] = np.array([df.iloc[j].target_position, df.iloc[i].target_position])
            p += 1

    return samples_np, positions

#about 4 min
logging.info('start building se_pairs')
se_pairs_testB = df_features_testB.groupby('origin_i').apply(apply_pair_test)
logging.info('finish building se_pairs')


# In[11]:


def get_choose_idx(df):
    mp_cnt_win = {}
    for pos_1, pos_2, score in df[['pos_1', 'pos_2', 'score']].values:
        if(pos_1 not in mp_cnt_win):
            mp_cnt_win[pos_1] = 0
        mp_cnt_win[pos_1] += score

    return max(mp_cnt_win, key=lambda x:mp_cnt_win[x])

def test_with_se_pair(model, se_pairs):
    cnt_sample = 0
    features_len = se_pairs.iloc[0][0].shape[1]
    for (samples, labels) in se_pairs:
        cnt_sample += samples.shape[0]
    test_x = np.zeros((cnt_sample, features_len))   
    positions = np.zeros((cnt_sample, 2))
    origin_idxs = np.zeros((cnt_sample, ))
    head, tail = 0, None
    
    se_pairs_index = se_pairs.index
    for i, (samples, position) in enumerate(se_pairs):
        tail = samples.shape[0] + head
        test_x[head : tail] = samples
        positions[head : tail] = position
        origin_idxs[head : tail] = se_pairs_index[i]
        head = tail
    
    scores = model.predict(test_x)
    df_idx_score = pd.DataFrame()
    df_idx_score['origin_i'] = origin_idxs
    df_idx_score['pos_1'] = positions[:, 0]
    df_idx_score['pos_2'] = positions[:, 1]    
    df_idx_score['score'] = scores  
    se_choose_pos = df_idx_score.groupby('origin_i').apply(get_choose_idx)
    return se_choose_pos


# In[12]:


import lightgbm as lgb

model_final = lgb.Booster(model_file='../user_data/makepair_gbdt/model_clf_pair.txt')


# In[13]:


testB_se_choose_pos = test_with_se_pair(model_final, se_pairs_testB)
testB_choose_idxs = list(testB_se_choose_pos)


# # regress_task

# In[14]:


df_features_testB['choose_position'] = df_features_testB['origin_i'].map(lambda x : testB_choose_idxs[x])


# In[38]:


pickle_path = '../user_data/regress_task/reg_features_name.pickle'
with open(pickle_path, 'rb') as f:
    reg_features_name = pickle.load(f)
# reg_features_name


# In[39]:


# not_features = ['origin_i', 'target_position', 'cur_action_expect_time', 'choose_position']
# reg_features_name = list( set(df_features_testB.columns) - set(not_features))

def apply_regress_sample(df):
    df_choose_rows = df.query('target_position == choose_position')
    df_choose_rows['label'] = df_choose_rows['cur_action_expect_time'] - df_choose_rows['last_action_expect_time']
    return df_choose_rows[reg_features_name + ['label']]
df_regress_testB = df_features_testB.groupby('origin_i').apply(apply_regress_sample)


# In[40]:


# not_features = ['origin_i', 'target_position', 'cur_action_expect_time', 'choose_position']
# reg_features_name = list( set(df_features_testB.columns) - set(not_features))

def apply_regress_sample(df):
    df_choose_rows = df.query('target_position == choose_position')
    df_choose_rows['label'] = df_choose_rows['cur_action_expect_time'] - df_choose_rows['last_action_expect_time']
    return df_choose_rows[reg_features_name + ['label']]

def test_process_reg(model, df_regress_testB):
    pre = model.predict(df_regress_testB[reg_features_name])
    return pre + df_regress_testB['last_action_expect_time'] 

def get_reg_result(testB_choose_idxs, df_features_testB, model):
    df_features_testB['choose_position'] = df_features_testB['origin_i'].map(lambda x : testB_choose_idxs[x])
    df_regress_testB = df_features_testB.groupby('origin_i').apply(apply_regress_sample)
    se_testB_expect_times = test_process_reg(model, df_regress_testB)
    return se_testB_expect_times

model_reg = lgb.Booster(model_file='../user_data/regress_task/model_reg.txt')
se_testB_expect_times = get_reg_result(testB_choose_idxs, df_features_testB, model_reg)


# In[41]:


now_str = str(datetime.datetime.now()).replace(':', '_')
test_expect_times_list = list(se_testB_expect_times)

submit_dir = '../action_predict/'
if os.path.exists(submit_dir) == False:
    os.mkdir(submit_dir)
cur_date_str = None
for i, (date, courier, wave_idx) in enumerate(key_list_testB):
    df_a_action = mp_action_testB[date][courier][wave_idx]
    choose_idx = int(testB_choose_idxs[i])
    choose_row = df_a_action.iloc[choose_idx]
    date_str = date.strftime('%Y%m%d')

    if cur_date_str != date_str:
        if cur_date_str != None:
            f.close()
        cur_date_str = date_str
        f = open(submit_dir + 'action_' + cur_date_str + '.txt', 'w+')
        f.write('courier_id,wave_index,tracking_id,courier_wave_start_lng,courier_wave_start_lat,action_type,expect_time\n')
    a_line = '%d,%d,%d,%.6f,%.6f,%s,%f\n'% (choose_row['courier_id'], choose_row['wave_index'], choose_row['tracking_id'],                                            choose_row['courier_wave_start_lng'], choose_row['courier_wave_start_lat'],                                             choose_row['action_type'], test_expect_times_list[i],)
    f.write(a_line)

f.close()


# In[ ]:


logging.info('finish prediction')

