#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:37:09 2020

@author: lichen
"""



import numpy as np
from numpy import array
from random import sample,seed
import time
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
import pandas as pd
#import numpy as np
from Bio import SeqIO
import h5py
import seaborn as sns
from scipy.stats import wilcoxon,pearsonr
from re import search
import math

#from numpy import argmax
from sklearn.metrics import roc_curve,auc,f1_score,recall_score,precision_score,accuracy_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle


import keras
from keras.models import Sequential,load_model,Model,clone_model,save_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional,Flatten,BatchNormalization 
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers import Input,Embedding, GlobalAveragePooling1D, Dense, concatenate
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras import regularizers

import tensorflow as tf









def onehot(fafile):
    x=[]
    for seq_record in SeqIO.parse(fafile, "fasta"):
        #print(seq_record.id)
        #print(seq_record.seq)
        #get sequence into an array
        seq_array = array(list(seq_record.seq))
        #integer encode the sequence
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)
        #one hot the sequence
        onehot_encoder = OneHotEncoder(sparse=False)
        #reshape because that's what OneHotEncoder likes
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
        x.append(onehot_encoded_seq)        
    x = array(x)
    return x




def onehot2(seqs):
    x=np.zeros((len(seqs),len(seqs[0]),4))
    allchs=array(['A','C','G','T'])
    for iseq in range(len(seqs)):
        #print(iseq)
        label_encoder = LabelEncoder()
        seq=array(list(seqs[iseq]))
        integer_encoded_seq = label_encoder.fit_transform(seq)
        #one hot the sequence
        onehot_encoder = OneHotEncoder(sparse=False)
        #reshape because that's what OneHotEncoder likes
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
        chs=np.unique(seq)
        if len(chs)<4:
            #id=np.take_along_axis(allchs,chs,0)
            id=np.in1d(allchs,chs)
            id=np.where(id==True)
            id=id[0].tolist()
            #tmp=np.zeros((len(seqs[0]),4))
            #tmp[:,array(id)]=onehot_encoded_seq     
            x[iseq][:,id]=onehot_encoded_seq
        else:
            x[iseq]=onehot_encoded_seq      
    return x





def split_stratified_into_train_test(x,y,frac_test=0.2,
                                         random_state=None):
    

    # Split original dataframe into train and temp dataframes.
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                          stratify=y,
                                                          test_size=frac_test,
                                                          random_state=random_state)

    assert len(x) == len(x_train) + len(x_test)
    return x_train, y_train,x_test,y_test






def combinedata(files):
    x=None
    y=None
    for i in range(len(files)):      
        hf1 = h5py.File(files[i], 'r')
        x1= hf1.get('x')
        y1= hf1.get('y')
        x1=array(x1)
        y1=array(y1)
        hf1.close()
        if i==0:
            x=x1
            y=y1
        else:
            x=np.concatenate((x,x1),axis=0)
            y=np.concatenate((y,y1),axis=0)
    return x,y



def get_balance_data(file=None,x=None,y=None,frac_test=0.2,frac_val=0.5,seeda=1,random_state=1,only_val=False):
    if file!=None:
        hf = h5py.File(file, 'r')
        x= hf.get('x')
        y= hf.get('y')
        x=array(x)
        y=array(y)
    np.random.seed(seeda)
    p = np.random.permutation(len(y))
    x=x[p]
    y=y[p]
    id1=np.where(y==1);id1=id1[0];#print(len(id1))
    id0=np.where(y==0);id0=id0[0];#print(len(id0))
    seed(a=seeda)
    id2=sample(list(id0),len(id1));#print(len(id2))
    #id2[:10]
    x=np.concatenate((x[id1],x[id2]),axis=0)
    y=np.concatenate((y[id1],y[id2]),axis=0)
    #print(x.shape)
    #print(y.shape)
    #np.concatenate((, ))
    x_train,xx,y_train,yy=train_test_split(x, y,stratify=y,test_size=frac_test,random_state=random_state)
    if only_val==False:
        x_test,x_val,y_test,y_val=train_test_split(xx, yy,stratify=yy,test_size=frac_val,random_state=random_state)
        print('train: ',x_train.shape,'test: ',x_test.shape,'val: ',x_val.shape)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_val = to_categorical(y_val)
        return  x_train, y_train,x_test,y_test,x_val,y_val
    elif only_val==True:
        print('train: ',x_train.shape,'val: ',xx.shape)
        y_train = to_categorical(y_train)
        yy = to_categorical(yy)
        return  x_train, y_train,xx,yy

       
        
    



def get_down_balance_data(file,frac_out=0,frac_val=0.3,seeda=1,random_state=1):
    hf = h5py.File(file, 'r')
    x= hf.get('x')
    y= hf.get('y')
    x=array(x)
    y=array(y)
    np.random.seed(seeda)
    p = np.random.permutation(len(y))
    x=x[p]
    y=y[p]
    id1=np.where(y==1);id1=id1[0];#print(len(id1))
    id0=np.where(y==0);id0=id0[0];#print(len(id0))
    seed(a=seeda)
    id2=sample(list(id0),len(id1));#print(len(id2))
    #id2[:10]
    x=np.concatenate((x[id1],x[id2]),axis=0)
    y=np.concatenate((y[id1],y[id2]),axis=0)
    x,x2,y,y2=train_test_split(x, y,stratify=y,test_size=frac_out,random_state=random_state)
    x_train,x_val,y_train,y_val=train_test_split(x, y,stratify=y,test_size=frac_val,random_state=random_state)
    print(x_train.shape,y_train.shape,x_val.shape,y_val.shape)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    return  x_train, y_train,x_val,y_val
   
    



def get_data(file=None,x=None,y=None,frac_test=0.2,frac_val=0.5,random_state=1):
    if file!=None:
        hf = h5py.File(file, 'r')
        x= hf.get('x')
        y= hf.get('y')
        x=array(x)
        y=array(y)
    x_train,xx,y_train,yy=train_test_split(x, y,stratify=y,test_size=frac_test,random_state=random_state)
    x_test,x_val,y_test,y_val=train_test_split(xx, yy,stratify=yy,test_size=frac_val,random_state=random_state)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_val.shape,y_val.shape)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    return  x_train, y_train,x_test,y_test,x_val,y_val



def converttoh5(prefix,bgtype):
    x=onehot(bgtype+'/'+prefix+'.'+bgtype+'.fasta')
    dat = pd.read_csv(bgtype+'/'+prefix+'.'+bgtype+'.label', sep="\t")
    y=dat['label']
    y=array(y)
    hf = h5py.File(bgtype+'/'+prefix+'.'+bgtype+'.h5', 'w')
    hf.create_dataset('x', data=x,compression='gzip')
    hf.create_dataset('y', data=y,compression='gzip')
    hf.close()



def get_data2(prefix,bgtype,frac_test=0.2,frac_val=0.5,random_state=1):
    x=onehot('../'+bgtype+'/'+prefix+'.'+bgtype+'.fasta')
    dat = pd.read_csv('../'+bgtype+'/'+prefix+'.'+bgtype+'.label', sep="\t")
    y=dat['label']
    y=array(y)
    x_train,xx,y_train,yy=train_test_split(x, y,stratify=y,test_size=frac_test,random_state=random_state)
    x_test,x_val,y_test,y_val=train_test_split(xx, yy,stratify=yy,test_size=frac_val,random_state=random_state)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_val.shape,y_val.shape)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    return  x_train, y_train,x_test,y_test,x_val,y_val



