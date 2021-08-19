#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:23:16 2021

@author: lichen
"""


from __future__ import print_function
import os
import sys
from model import *
from lib import *
import argparse

import warnings
warnings.filterwarnings('ignore')

import time



def parse_arguments(parser):
    
    
    parser.add_argument('--comparison', dest='comparison', action='store_true', help='Use this option for comparing TL model')
    parser.set_defaults(comparison=False)
    
    parser.add_argument('--pretrain', dest='pretrain', action='store_true',help='Use this option for pretrain model')
    parser.set_defaults(pretrain=False)
    
    parser.add_argument('--jobname', type=str,
                        help='Job name for model, figure and result')
       
    # for files arguments
    parser.add_argument('--pretrain_model', type=str,
                        help='Pretrained model h5 file')
    
    
    parser.add_argument('--data_file', type=str,
                        help='h5 data file for x y')
    
    parser.add_argument('--seq_file', type=str,
                        help='fasta sequence for SNPs')
    
    parser.add_argument('--snp_file', type=str,
                        help='label for SNPs')
                        
    parser.add_argument('--seq_files', nargs='+' ,type=str,
                        help='fasta sequence for SNPs for multiple diseases')
    
    parser.add_argument('--snp_files', nargs='+' ,type=str,
                        help='label for SNPs for multiple diseases')
    

    # for model arguments
    
    parser.add_argument('--frac_pretrain', type=float, default=1,
                        help='Fractions of pre-training samples')
    
    parser.add_argument('--frac_trains', nargs='+' ,type=float, default=[0.1,0.3,0.7,1],
                        help='Fractions of training samples')
        
    parser.add_argument('--frac_test', type=float, default=0.2,
                        help='Fraction of testing samples')
    
    parser.add_argument('--frac_val', type=float, default=0.2,
                        help='Fraction of validation_split')
    
    parser.add_argument('--methods', nargs='+' ,type=str, default=['tl','self','base','tl0'],
                        help='Methods used in comparison')
  
    parser.add_argument('--nrep', type=int, default=10,
                        help='Number for expeirments')
    
    parser.add_argument('--batch_size', type=int, default=32*2,
                        help='The batch size for training')

    parser.add_argument('--epochs', type=int, default=50,
                        help='The max epoch for training')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='The dropout rate for training')
    
    parser.add_argument('--ilayer', type=int, default=2,
                        help='ith layer starting transfer learning')
    
    parser.add_argument('--nodes', nargs='+' ,type=float, default=[64,32],
                        help='Number of nodes in customized layers')
    
    args = parser.parse_args()

    return args






def main(args):

    print(args)
    
    if args.pretrain:
        if args.seq_files is None or args.snp_files is None:
          sys.exit("seq_files and snp_files should be both provided!")
        for i in range(len(args.seq_files)):
          if i==0:
            x=onehot(args.seq_files[i])
            dat = pd.read_csv(args.snp_files[i], sep="\t")
            y=dat['label']
            y=np.asarray(y)
            if args.frac_pretrain==0:
                pretrain_model=build_model_seq(x,y)
                model_file=args.jobname+str(args.frac_pretrain)+'.seq.h5'
                pretrain_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                pretrain_model.save(model_file)
                break
            else:
                id1=np.where(y==1);id1=id1[0]
                id0=np.where(y==0);id0=id0[0]
                n=round(len(id1)*args.frac_pretrain)
                id1_sub=sample(list(id1),n)
                id0_sub=sample(list(id0),n)
                x=np.concatenate((x[id1_sub],x[id0_sub]),axis=0)
                y=np.concatenate((y[id1_sub],y[id0_sub]),axis=0)
                p = np.random.permutation(len(y))
                x=x[p]
                y=y[p]
          else:
            xx=onehot(args.seq_files[i])
            dat = pd.read_csv(args.snp_files[i], sep="\t")
            yy=dat['label']
            yy=np.asarray(yy)
            id1=np.where(yy==1);id1=id1[0]
            id0=np.where(yy==0);id0=id0[0]
            id1_sub=sample(list(id1),round(len(id1)*args.frac_pretrain))
            id0_sub=sample(list(id0),round(len(id1)*args.frac_pretrain))
            xx=np.concatenate((xx[id1_sub],xx[id0_sub]),axis=0)
            yy=np.concatenate((yy[id1_sub],yy[id0_sub]),axis=0)
            p = np.random.permutation(len(yy))
            xx=xx[p]
            yy=yy[p]
            x=np.concatenate((x,xx),axis=0)
            y=np.concatenate((y,yy))
        

        if args.frac_pretrain!=0:
            model_file=args.jobname+str(args.frac_pretrain)+'.seq.h5'
            pretrain_model=build_model_seq(x,y)
            history=fit_model_seq(pretrain_model,model_file,x,y,args.frac_val,args.epochs,args.batch_size,verbose=1)
            

    if args.comparison:
      
        print('Comparing with TL methods')

        # load data
        if args.data_file is not None:
          x,y=combinedata([args.data_file])
          
        if args.seq_file is not None:
          x=onehot(args.seq_file)
          
        if args.snp_file is not None:
          dat = pd.read_csv(args.snp_file, sep="\t")
          y=dat['label']
          y=np.asarray(y)
        
        
        #load pretrain model
        basemodel = load_model(args.pretrain_model)
        
        #load tf model
        tlmodel=reset_tlmodel_seq(basemodel,ilayer=args.ilayer,nodes=args.nodes,dropout=args.dropout)
    
    
        print('Model trainable\n')
        for i,layer in enumerate(tlmodel.layers):
            print(i,layer.name,layer.trainable) 
    
        
        test_tl,test_self,test_base,test_tl0=eval_tlmodel_seq(
                    basemodel=args.pretrain_model,jobname=args.jobname,methods=args.methods,
                    x=x,y=y,
                    epochs=args.epochs, batch_size = args.batch_size, nrep=args.nrep,
                    frac_trains=args.frac_trains,frac_test=args.frac_test,frac_val=args.frac_val,
                    ilayer=args.ilayer,nodes=args.nodes,dropout=args.dropout,
                    verbose1=0,verbose2=0,balance=True)


        print('Compare AUC:\n')

        compresult_sam(test_tl,test_self,test_base,test_tl0,args.frac_trains,args.methods,args.jobname,metric='auc')

        print('Compare R:\n')

        compresult_sam(test_tl,test_self,test_base,test_tl0,args.frac_trains,args.methods,args.jobname,metric='R')

        saveresult_sam(test_tl,test_self,test_base,test_tl0,args.jobname)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tranfser learning')
    args = parse_arguments(parser)
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))




    
    
    
