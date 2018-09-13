# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:24:44 2018

@author: paprasad
"""
import datetime
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


def insert_dict_to_db(myDict,conn,table,StartTime=None):
    
    if StartTime is not None:
        EndTime = datetime.datetime.now().replace(microsecond=0)
        Total_Training_Time = EndTime - StartTime
        Total_Training_Time = Total_Training_Time.total_seconds()
        myDict['Total_Training_Time'] = int(Total_Training_Time)
        
    cursor = conn.cursor()
    placeholders = ', '.join(['%s'] * len(myDict))
    columns = ', '.join(myDict.keys())
    sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % (table, columns, placeholders)
    cursor.execute(sql, list(myDict.values()))
    conn.commit()


def fetch_score_from_db():
    
    ConnParam = load_DBconfig()
    DbConn = conn(ConnParam['hostname'],ConnParam['port'],ConnParam['username'],ConnParam['password'],ConnParam['schema'])
    sql = 'select MultinomialNB_clf, BernoulliNB_clf, LinearSVC_clf,SGDClassifier_clf,LogisticRegression_clf,RandomForestClassifier_clf from '+ConnParam['build_stats']+' where id = (select max(id) from '+ConnParam['build_stats']+')' 
    
    df = pd.read_sql(sql,DbConn)
    models = list(df.columns.values)
    df = df.T
    score = list(df.iloc[:,0])
    MyDict = {}
    
    for i in range(0,len(model)):
        MyDict[model[i]] = score[i]
        
    return MyDict       
    

    
    