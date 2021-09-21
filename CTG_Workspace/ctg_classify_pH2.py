import pandas as pd
import numpy as np
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import ltc2_model as ltc
from ctrnn2_model import CTRNN, NODE, CTGRU
import argparse
import datetime as dt

class CtgData:  
    def data_frame(self):
        #Anotation Set
        ann_name = os.path.join("database","ann_db_read.csv")
        #print(ann_name)
        ann = pd.read_csv(ann_name,header=0, index_col=0)
        # "database/ann_db_read.csv", header=0, index_col=0

        #CTG Sets
        ctg_name = sorted([os.path.join("database/signals",d) for d in os.listdir("database/signals") if d.endswith(".csv")])
        #print(ctg_name)
        ctg = [pd.read_csv(c, header=0) for c in ctg_name]

        #Name of file exclude path
        ctg_file_name = []
        for name in ctg_name:
            file_name = name.replace("database/signals/", "")
            file_name = file_name.replace(".csv", "")
            file_name = int(file_name)
            ctg_file_name.append(file_name)
        #print(ctg_file_name)

        #Insert Name Column
        for i in range(len(ctg)):
            name = ctg_file_name[i]
            ctg[i].insert(0,"Name",name)

        df = []
        min_df = 0
        #Trim
        for i in range(len(ctg)):
            for j in range(len(ctg[i])):
                if((ctg[i].at[j,'FHR']!=0) or (ctg[i].at[j,'UC']!=0)):
                    result = j
            if min_df==0 or min_df > result:
                min_df = result
        #print("Minimum Signal Length: ", min_df+1)
        for i in range(len(ctg)):
            #Df without merge
            #df.append(ctg[i].iloc[:min_df+1])
            #Merge 2 ann and ctg
            df.append(pd.merge(ctg[i].iloc[:min_df+1], ann, on="Name"))
        #print(df[1])
        return df, ann, min_df+1

    def Normalization(self,feature,n):
        feature = feature.reshape((len(feature), 1))
        scaler = MinMaxScaler(feature_range=(0, 1))

        scaler_fit = scaler.fit(feature)
        #print("Frame: ", n)
        #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        return scaler.transform(feature)

    def Normalize_X(self,X, name):
        #X = X.replace(0, 1)
        n = name
        for i in range(1,3,1):
            if i ==1 :
                j= 1;
            else:
                j = 0.01

            X.iloc[:,i] = X.iloc[:,i].replace(0, j)
            feature = X.iloc[:,i].values
            X.iloc[:,i] = self.Normalization(feature,n)
        
        return X

    def Normalize_y(self,y, name):
        n = name
        #feature = y.iloc[:,:].values
        #y.iloc[:,:] = self.Normalization(feature,n)
        feature = y.values
        y = self.Normalization(feature,n)
        return y

    def iterate_train(self,batch_size=16):
        total_seqs = self.X_train.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.X_train[:,permutation[start:end]]
            batch_y = self.y_train[:,permutation[start:end]]
            yield (batch_x,batch_y)

    def batching_X(self, X,signal_length,window):
        #print("X rows: ",X.shape[0])
        sn = signal_length
        w = window
        X = X.to_numpy()
        X = X.reshape(X.shape[0]//w,w, X.shape[1])
        return X

    def batching_y(self, y, signal_length, window):
        #print("y rows: ", y.shape[0])
        sn = signal_length
        w = window
        y = y.to_numpy()
        y = np.where(y <= 7, 1, y)
        y = np.where(y != 1, 0, y)
        y = y.reshape(y.shape[0]//w,w)
        return y
        #print("Y rows: ",y.shape[0])
        #y = y.to_numpy()
        #y = y.reshape(1,y.shape[0])
        #return y
    
    def __init__(self,w):
        #CTG Set initialize
        ctg = []
        ctg, ann, signal_length = self.data_frame()
        self.window = w
        #Split 
        train, test_valid = train_test_split(ctg, test_size = 0.3, random_state=42)
        test, valid = train_test_split(test_valid, test_size = 0.66, random_state=42)
        #print(len(train),len(test),len(valid),len(ctg))

        #Without Merge
        #y = pd.DataFrame(ann.iloc[:,0:1])
        X=[]
        y=[]
        for i in ctg:
            X.append(i.iloc[:,1:4])
            y.append(i.iloc[:,4])
        #Split train, test and validation
        X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.66, random_state=42)
        #Concat
        X_train = pd.concat(X_train)
        X_test = pd.concat(X_test)
        X_valid = pd.concat(X_valid)
        y_train = pd.concat(y_train)
        y_test = pd.concat(y_test)
        y_valid = pd.concat(y_valid)
        #Normalize X
        self.X_train = self.Normalize_X(X_train,'X_train')
        self.X_test = self.Normalize_X(X_test,'X_test')
        self.X_valid = self.Normalize_X(X_valid,'X_valid')
        #Normalize Y
        #self.y_train = self.Normalize_y(y_train,'y_train')
        #self.y_test = self.Normalize_y(y_test,'y_test')
        #self.y_valid = self.Normalize_y(y_valid,'y_valid')
        #Batching X
        self.X_train = self.batching_X(X_train,signal_length,self.window)
        self.X_test = self.batching_X(X_test,signal_length,self.window)
        self.X_valid = self.batching_X(X_valid,signal_length,self.window)
        #Batching Y without Merge
        #self.y = self.batching_y(y)
        #self.y_train = self.batching_y(y_train)
        #self.y_test = self.batching_y(y_test)
        #self.y_valid = self.batching_y(y_valid)

        #Batching Y with Merge
        self.y_train = self.batching_y(y_train,signal_length,self.window)
        self.y_test = self.batching_y(y_test,signal_length,self.window)
        self.y_valid = self.batching_y(y_valid,signal_length,self.window)


class TrainingModel:
    #Similar - Person
    #Similar loss, acc - Binary Cross Entropy, Reduce Mean
    #Learning Rate: 0.01-0.02 for LTC, 0.001 for all other models.
    def __init__(self,window, model_type,model_size,sparsity_level=0.0,learning_rate = 0.001):
        self.model_type = model_type
        self.window = window
        self.constrain_op = []
        self.sparsity_level = sparsity_level
        self.X = tf.placeholder(dtype=tf.float32,shape=[None,None,3])
        self.target_y = tf.placeholder(dtype=tf.int32,shape=[None,None])

        self.model_size = model_size
        head = self.X
        
        #Print Shape 1
        print("Head Shape 1",head.shape)
        if(model_type == "lstm"):
            #unstacked_signal = tf.unstack(self.X,axis=0)
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op.extend(self.wm.get_param_constrain_op())
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))
        target_y = tf.expand_dims(self.target_y,axis=-1)
        print(target_y.shape)
        
        #Print Shape 2
        print("Head Shape 2",head.shape)
        if(self.sparsity_level > 0):
            self.constrain_op.extend(self.get_sparsity_ops())
        #Change Logit shape
        self.y = tf.layers.Dense(2,activation=None)(head)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # self.result_file = os.path.join("results","ctg","{}_{}_{:02d}.csv".format(model_type,model_size,int(100*self.sparsity_level)))
        self.result_file = os.path.join("results","ctg_class","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/ctg_class")):
            os.makedirs("results/ctg_class")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("window size, best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","ctg_class","{}".format(model_type))
        if(not os.path.exists("tf_sessions/ctg_class")):
            os.makedirs("tf_sessions/ctg_class")
            
        self.saver = tf.train.Saver()

    def get_sparsity_ops(self):
        tf_vars = tf.trainable_variables()
        op_list = []
        for v in tf_vars:
            # print("Variable {}".format(str(v)))
            if(v.name.startswith("rnn")):
                if(len(v.shape)<2):
                    # Don't sparsity biases
                    continue
                if("ltc" in v.name and (not "W:0" in v.name)):
                    # LTC can be sparsified by only setting w[i,j] to 0
                    # both input and recurrent matrix will be sparsified
                    continue
                op_list.append(self.sparse_var(v,self.sparsity_level))
                
        return op_list
        
    def sparse_var(self,v,sparsity_level):
        mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
        v_assign_op = tf.assign(v,v*mask)
        print("Var[{}] will be sparsified with {:0.2f} sparsity level".format(
            v.name,sparsity_level
        ))
        return v_assign_op

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,ctg_data,epochs,verbose=True,log_period=50):

        best_valid_accuracy = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        print("Entering training loop")
        #print("self.X: ",self.X.shape)
        #print("ctg_data.X_test: ",ctg_data.X_test.shape)
        #print("self.target_y",self.target_y.shape)
        #print("ctg_data.y_test",ctg_data.y_test.shape)
        #print("self.accuracy",self.accuracy)
        #print("self.loss",self.loss)
        for e in range(epochs):
            if(verbose and e%log_period == 0):
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.X:ctg_data.X_test,self.target_y: ctg_data.y_test})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.X:ctg_data.X_valid,self.target_y: ctg_data.y_valid})
                if(valid_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

            #Training
            print("Epoch: ",e)
            losses = []
            accs = []
            for batch_x,batch_y in ctg_data.iterate_train(batch_size=32):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.X:batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)
                #print("loss: " + str(loss))
                #print("acc: " + str(acc))

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            self.window,
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))

ctg_data = CtgData(i)
print("Window: ", i)
tf.reset_default_graph()
model = TrainingModel(window=ctg_data.window, model_type = "lstm", model_size=32, sparsity_level=0.0)

model.fit(ctg_data ,epochs=200,log_period=1)
