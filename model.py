#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 01:35:05 2020

@author: lichen
"""


from lib import *



def build_model_seq(x_train, y_train):
   #n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
    inputt=Input(shape=(x_train.shape[1],x_train.shape[2]))
    output=Conv1D(filters=32, kernel_size=32, activation='relu')(inputt)
    output=MaxPooling1D(pool_size=4)(output)
    output=Conv1D(filters=32, kernel_size=32, activation='relu')(inputt)
    output=MaxPooling1D(pool_size=4)(output)
    output=Flatten()(output)
    output = Dense(128,activation='relu')(output)
    output=Dropout(0.5)(output)
    output=Dense(64,activation='softmax')(output)
    output=Dropout(0.5)(output)
    output=Dense(2,activation='softmax')(output)
    model = Model(inputs=inputt, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def fit_model_seq(model,model_file, x_train, y_train,frac_val,epochs, batch_size,verbose):
    if y_train.ndim==1:
        y_train = to_categorical(y_train)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=5)
    mc = ModelCheckpoint(model_file, monitor='val_accuracy', mode='max', verbose=verbose, save_best_only=True)
    history=model.fit(x_train, y_train, validation_split=frac_val,epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=[es, mc])
    if verbose==1:
        plt.clf()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        plt.savefig('train.png')
    return history




def eval_model(model,x_test,y_test,verbose=0):    
    y_test_prob = model.predict(x_test)
    y_test_classes=np.argmax(model.predict(x_test), axis=-1)
    fpr, tpr, thresholds = roc_curve(y_test[:,0], y_test_prob[:,0])
    auc_test = auc(fpr, tpr)
    acc_test=accuracy_score(y_test_classes, np.argmax(y_test, axis=-1))
    f1_test = f1_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    recall_test = recall_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    precision_test = precision_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    R_test=pearsonr(y_test[:,0], y_test_prob[:,0])[0]
    
    acc_test=round(acc_test,3)
    auc_test=round(auc_test,3)
    f1_test=round(f1_test,3)
    precision_test=round(precision_test,3)
    recall_test=round(recall_test,3)
    R_test=round(R_test,3)
    
    if verbose==1:
        print('Test: acc %.3f, auc %.3f, f1 %.3f, precision %.3f, recall %.3f,R %.3f\n' % (acc_test, auc_test, f1_test, precision_test, recall_test,R_test))
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc_test)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
    return [acc_test, auc_test, f1_test, precision_test, recall_test,R_test]





def reset_tlmodel_seq(basemodel,ilayer,nodes,dropout):
    iflatten=0
    for i in range(len(basemodel.layers)):
        if search('flatten',basemodel.layers[i].name) is not None:
            iflatten=i
            break    
    riflatten=len(basemodel.layers)-iflatten
    x = basemodel.layers[-riflatten].output        

    if ilayer=='2layer':
        x = Dense(nodes[0], activation='relu', trainable=True)(x)
        x=Dropout(dropout)(x)
        x = Dense(nodes[1], activation='relu', trainable=True)(x)
        x=Dropout(dropout)(x)  
    elif ilayer=='1layer':   
        x = Dense(nodes[0], activation='relu', trainable=True)(x)
        #x=Dropout(0.5)(x)
    x=Dense(2, activation='softmax', trainable=True)(x)
    tlmodel = Model(basemodel.inputs, x)
    for layer in tlmodel.layers[:(iflatten+1)]:
        layer.trainable = False
    # for i,layer in enumerate(tlmodel.layers):
    #     print(i,layer.name,layer.trainable)    
    return tlmodel 
        




def eval_tlmodel_seq(basemodel,jobname,methods,ilayer,nodes,dropout,
                     data_file=None,x=None,y=None,
                epochs=50, batch_size = 32, nrep=10,frac_trains=[0.1,0.5,1],frac_test=0.2,frac_val=0.2,
                verbose1=0, verbose2=0,balance=True):    
    
    tlmodel_file=jobname+'.tlmodel.h5'
    selfmodel_file=jobname+'.selfmodel.h5'
    
    basemodel = load_model(basemodel)

    
    shape=(len(frac_trains),nrep)
    
    test_tl_auc=np.zeros(shape)
    test_tl_R=np.zeros(shape)
    
    test_self_auc=np.zeros(shape)
    test_self_R=np.zeros(shape)
    
    test_base_auc=np.zeros(shape)
    test_base_R=np.zeros(shape)
   
    test_tl0_auc=np.zeros(shape)
    test_tl0_R=np.zeros(shape)  
    
    
    
    for ifrac in range(len(frac_trains)):
        
        frac_train=frac_trains[ifrac]
        print('frac_train: ',frac_trains[ifrac])  


        for irep in range(nrep):
                
            if data_file!=None:
              hf = h5py.File(data_file, 'r')
              x= hf.get('x')
              y= hf.get('y')
              x=array(x)
              y=array(y)
              
              
            if balance==True:
              np.random.seed(irep)
              id1=np.where(y==1);id1=id1[0];#print(len(id1))
              id0=np.where(y==0);id0=id0[0];#print(len(id0))
              #seed(a=irep)
              id2=sample(list(id0),len(id1));#print(len(id2))
              x=np.concatenate((x[id1],x[id2]),axis=0)
              y=np.concatenate((y[id1],y[id2]),axis=0)
              p = np.random.permutation(len(y))
              x=x[p]
              y=y[p]
              
                        
            x_train,x_test,y_train,y_test=train_test_split(x, y,stratify=y,test_size=frac_test,random_state=irep)
            if frac_train!=1:
                x_train,_,y_train,_=train_test_split(x_train, y_train,stratify=y_train,test_size=1-frac_train,random_state=irep)
            if irep==0:
                print('Train size:',int(x_train.shape[0]*(1-frac_val)),'Validation size:',int(x_train.shape[0]*frac_val),'Test size:',x_test.shape[0])
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
        

            #case1 tf without tuning
            if 'tl0' in methods:
                tlmodel0=reset_tlmodel_seq(basemodel,nodes,ilayer,dropout) #need reset otherwise tlmodel from last loop will be reuse
                test=eval_model(tlmodel0,x_test,y_test,verbose=verbose1)
                test_tl0_auc[ifrac,irep]=test[1]
                test_tl0_R[ifrac,irep]=test[5]  
              
            
            #case2 tf with tuning
            if 'tl' in methods:
                tlmodel=reset_tlmodel_seq(basemodel,nodes,ilayer,dropout) #need reset otherwise tlmodel from last loop will be reuse                
                history=fit_model_seq(tlmodel,tlmodel_file, x_train, y_train,frac_val, epochs, batch_size,verbose1)
                if verbose2==1:
                    plt.plot(history.history['loss'], label='train')
                    plt.plot(history.history['val_loss'], label='test')
                    plt.legend()
                    plt.show()
                tlmodel = load_model(tlmodel_file)
                test=eval_model(tlmodel,x_test,y_test,verbose1)
                test_tl_auc[ifrac,irep]=test[1]
                test_tl_R[ifrac,irep]=test[5]
            
                    
            #case3 base model
            if 'base' in methods:
                test=eval_model(basemodel,x_test,y_test,verbose1)
                test_base_auc[ifrac,irep]=test[1]
                test_base_R[ifrac,irep]=test[5]
           
            
            #case4 self model
            if 'self' in methods:
                selfmodel=build_model_seq(x_train,y_train)
                history=fit_model_seq(selfmodel,selfmodel_file, x_train, y_train,frac_val, epochs, batch_size,verbose1)
                if verbose2==1:
                    plt.plot(history.history['loss'], label='train')
                    plt.plot(history.history['val_loss'], label='test')
                    plt.legend()
                    plt.show()
                selfmodel = load_model(selfmodel_file)
                test=eval_model(selfmodel,x_test,y_test,verbose=verbose1)
                test_self_auc[ifrac,irep]=test[1]
                test_self_R[ifrac,irep]=test[5]

        
    test_tl={'test_tl_auc':test_tl_auc,'test_tl_R':test_tl_R}
    test_self={'test_self_auc':test_self_auc,'test_self_R':test_self_R}
    test_base={'test_base_auc':test_base_auc,'test_base_R':test_base_R}
    test_tl0={'test_tl0_auc':test_tl0_auc,'test_tl0_R':test_tl0_R}
    return [test_tl,test_self,test_base,test_tl0]





def saveresult_sam(test_tl,test_self,test_base,test_tl0,jobname):
    mode='test'
    hf = h5py.File(jobname+'.result.h5', 'w')
    g1 = hf.create_group('tl')
    g1.create_dataset(mode+'_tl_auc', data=test_tl[mode+'_tl_auc'])
    g1.create_dataset(mode+'_tl_R', data=test_tl[mode+'_tl_R'])
    g2 = hf.create_group('self')
    g2.create_dataset(mode+'_self_auc', data=test_self[mode+'_self_auc'])
    g2.create_dataset(mode+'_self_R', data=test_self[mode+'_self_R'])
    g3 = hf.create_group('base')
    g3.create_dataset(mode+'_base_auc', data=test_base[mode+'_base_auc'])
    g3.create_dataset(mode+'_base_R', data=test_base[mode+'_base_R'])
    g4 = hf.create_group('tl0')
    g4.create_dataset(mode+'_tl0_auc', data=test_tl0[mode+'_tl0_auc'])
    g4.create_dataset(mode+'_tl0_R', data=test_tl0[mode+'_tl0_R'])
    hf.close()




#CAGI_train.sample.result.h5
def loadresult_sam(save_file,mode='test'):
    hf = h5py.File(save_file, 'r')
    g1 = hf.get('tl')
    test_tl_auc = g1.get(mode+'_tl_auc')
    test_tl_R = g1.get(mode+'_tl_R')
    test_tl_auc=np.array(test_tl_auc)
    test_tl_R=np.array(test_tl_R)
    g2 = hf.get('self')
    test_self_auc = g2.get(mode+'_self_auc')
    test_self_R = g2.get(mode+'_self_R')
    test_self_auc=np.array(test_self_auc)
    test_self_R=np.array(test_self_R)
    g3 = hf.get('base')
    test_base_auc = g3.get(mode+'_base_auc')
    test_base_R = g3.get(mode+'_base_R')
    test_base_auc=np.array(test_base_auc)
    test_base_R=np.array(test_base_R)
    g4 = hf.get('tl0')
    test_tl0_auc = g4.get(mode+'_tl0_auc')
    test_tl0_R = g4.get(mode+'_tl0_R')
    test_tl0_auc=np.array(test_tl0_auc)
    test_tl0_R=np.array(test_tl0_R)
    hf.close()
    test_tl={'test_tl_auc':test_tl_auc,'test_tl_R':test_tl_R}
    test_self={'test_self_auc':test_self_auc,'test_self_R':test_self_R}
    test_base={'test_base_auc':test_base_auc,'test_base_R':test_base_R}
    test_tl0={'test_tl0_auc':test_tl0_auc,'test_tl0_R':test_tl0_R}
    return [test_tl,test_self,test_base,test_tl0]



    
    

def compresult_sam(test_tl,test_self,test_base,test_tl0,frac_trains,methods,jobname,metric):
    #adapt boxplot for grant application
    test_tl2 = pd.DataFrame(np.transpose(test_tl['test_tl_'+metric]),  columns=frac_trains)
    a=pd.melt(test_tl2)
    a['method']=['tl']*a.shape[0]
    test_self2=pd.DataFrame(np.transpose(test_self['test_self_'+metric]),  columns=frac_trains)
    b=pd.melt(test_self2)
    b['method']=['self']*b.shape[0]
    test_base2=pd.DataFrame(np.transpose(test_base['test_base_'+metric]),  columns=frac_trains)
    c=pd.melt(test_base2)
    c['method']=['base']*c.shape[0]
    test_tl02=pd.DataFrame(np.transpose(test_tl0['test_tl0_'+metric]),  columns=frac_trains)
    d=pd.melt(test_tl02)
    d['method']=['tl0']*d.shape[0]
    abcd=np.concatenate((a,b,c,d),axis=0)
    abcd=pd.DataFrame(abcd,columns=['scenario', 'value', 'method'])
    abcd.to_csv(jobname+'.'+metric+'.csv')
    tmp=np.repeat(False,abcd.shape[0])
    for i in range(len(methods)):
        tmp=tmp | np.asarray(abcd['method']==methods[i])
    tmp=pd.Series(tmp)
    boxplot(abcd[tmp],metric,jobname)   
    
    print('tl\n',list(np.mean(test_tl2, axis = 0)) )
    print('self\n',list(np.mean(test_self2, axis = 0)) )
    # print('base\n',np.mean(test_base2, axis = 0))
    # print('tl0\n',np.mean(test_tl02, axis = 0)) 


    for col in range(len(frac_trains)):
        print('\nfrac_train:',frac_trains[col])
        print(' self:', wilcoxon(np.asarray(test_tl2)[:,col],np.asarray(test_self2)[:,col],alternative ='greater'))
        #print('base: ',wilcoxon(np.asarray(test_tl2)[:,col],np.asarray(test_base2)[:,col],alternative ='greater'))
        #print('tl0: ',wilcoxon(np.asarray(test_tl2)[:,col],np.asarray(test_tl02)[:,col],alternative ='greater'))


def boxplot(dat,metric,jobname):
    dat['value'] = dat['value'].astype(float)
    bp=sns.boxplot(y='value', x='scenario',
                 data=dat,
                 palette="colorblind",
                  hue='method').set_title(metric)
    bp=sns.stripplot(y='value', x='scenario',
                   data=dat,
                   jitter=True,
                   dodge=True,
                   marker='o',
                   alpha=0.5,
                   hue='method',
                   color='grey')
    handles, labels = bp.get_legend_handles_labels()
    l = plt.legend(handles[0:4], labels[0:4])
    #plt.savefig('fig.png')
    bp.get_figure().savefig(jobname+'.'+metric+'.png')
    plt.clf()
    #l = plt.legend(handles, labels)
    






          

