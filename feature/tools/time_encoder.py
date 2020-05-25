# encoding: utf-8
'''
Created on Mar 15, 2020

@author: Yongrui Huang
'''

import time

def time_vector(time_stamp, prefix=''):
    """
    时，分，秒，周几，是否周末，是否营业时间
    # TODO:
    时间特征One-hot
    上午、下午、晚上、是否饭点
    """
    time_subject = time.localtime(time_stamp)
    
    return {
        '%s_hour' % (prefix) : time_subject.tm_hour,
        '%s_minute' % (prefix) : time_subject.tm_min,
        '%s_second' % (prefix) : time_subject.tm_sec,
        '%s_weekday' % (prefix) : time_subject.tm_wday,
        '%s_is_weekend' % (prefix) : time_subject.tm_wday in set([5, 6]),
        '%s_is_worktime' % (prefix) :  (time_subject.tm_hour >= 8 and time_subject.tm_hour < 22)
    }

if __name__ == '__main__':
    pass