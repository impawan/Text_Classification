# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:24:44 2018

@author: paprasad
"""

import MySQLdb
import pandas as pd
import numpy as np


def conn(hostname, port,username,password,schema):
    if type(port) is not 'int':
        port = int(port)
    DBconn = MySQLdb.connect(host=hostname, port=port,user=username, passwd=password, db=schema)
    return DBconn


def dataframe_from_db(query, DBConn):
    df = pd.read_sql(query,con = DBConn)
    return df

def load_DBconfig(name='MYSQLDB.config'):
    config_file = open(name,'r')
    lines = config_file.readlines()
    
    
    TempDict = {}
    for line in lines:
        line = line.rstrip('\n')
        param,value = line.split(":")
        TempDict[param] = value
        
    config_file.close()    
    return TempDict