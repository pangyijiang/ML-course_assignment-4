# -*- coding: UTF-8 -*-

import numpy as np
import random
from sympy import *


class LR_customized():
    featuresDic = {}
    adamDic = {}
    def __init__(self, dim_in, max_iter = 1000, tol = 1e-4, batch_size = 64, lr = 1e-2, class_weight = None):
        self.dim_in  = dim_in
        self.featuresDic["lr"] = lr    
        self.featuresDic["tol"] = tol
        self.featuresDic["max_iter"] = max_iter 
        self.featuresDic["batch_size"] = batch_size 
        self.featuresDic["class_weight"] = class_weight 
        self.featuresDic["lambda"] = 0
        self.featuresDic["lr_max"] = lr  
        self.featuresDic["lr_min"] = lr/50
        self.adamDic["b1"] = 0.9
        self.adamDic["b2"] = 0.999
        self.adamDic["e"] = 0.00000001
        self.adamDic["mt"] = np.zeros((dim_in, 1))
        self.adamDic["vt"] = np.zeros((dim_in, 1))
        # self.theta_Set = np.random.random((dim_in, 1)) - 0.5
        self.theta_Set = np.zeros((dim_in, 1))

    def fit(self, X_in, Y_in, verbose = False):
        loss_trace = []
        epoch = 0
        while(True):
            #sort data randomly
            c = list(zip(X_in, Y_in))
            random.shuffle(c)
            X_in, Y_in = zip(*c)
            X_in = np.array([i for i in X_in])
            Y_in = np.array([i for i in Y_in])
            #SGD
            loss_epoch = []
            for i_batch in range(int(len(Y_in)/self.featuresDic["batch_size"])):
                try:
                    X_in_sub = X_in[i_batch*self.featuresDic["batch_size"]:(i_batch+1)*self.featuresDic["batch_size"]]
                    Y_in_sub = Y_in[i_batch*self.featuresDic["batch_size"]:(i_batch+1)*self.featuresDic["batch_size"]]
                except:
                    X_in_sub = X_in[i_batch*self.featuresDic["batch_size"]:]
                    Y_in_sub = Y_in[i_batch*self.featuresDic["batch_size"]:]
                Y_linear, Y_pred = self.forward(X_in_sub)
                loss = self.loss_func(Y_pred, Y_in_sub)
                self.backward(X_in_sub, Y_linear, Y_pred, Y_in_sub, i_batch)   #update theta_Set
                loss_epoch.append(loss)
            loss_epoch = np.mean(loss_epoch)
            loss_trace.append(loss_epoch)
            if epoch%20 ==0: print("epoch = %s, loss = %.4f" %(epoch, loss_epoch))
            self.lr_decline(epoch)
            epoch = epoch + 1
            if epoch > self.featuresDic["max_iter"]: break
        return loss_trace

    def forward(self, X_in):
        Y_linear = X_in@self.theta_Set
        Y_pred = self.sigmodFunction(Y_linear)
        return Y_linear, Y_pred

    def backward(self, X_in_sub, Y_linear, Y_pred, Y_label, i_batch):
        grads = np.zeros((self.dim_in, 1))
        # grads = 2*(Y_pred - Y_label)*self.sigmodFunction(Y_linear)*(1 - self.sigmodFunction(Y_linear))*(X_in_sub)
        grads = (-Y_label/Y_pred + (1-Y_label)/(1 - Y_pred))*self.sigmodFunction(Y_linear)*(1 - self.sigmodFunction(Y_linear))*(X_in_sub)
        grads = np.mean(grads, axis = 0)[:, np.newaxis]
        self.theta_Set = self.theta_Set - self.featuresDic["lr"]*grads


    def loss_func(self, Y_pred, Y_label):
        # err_all = np.linalg.norm((Y_pred - Y_label), ord = 2, axis = 1)   #L2 loss
        err_all = -(Y_label*np.log(Y_pred) + (1-Y_label)*np.log(1 - Y_pred))
        return np.mean(err_all)

    def lr_decline(self, epoch):
        self.featuresDic["lr"] = self.featuresDic["lr_min"] + 0.5*(self.featuresDic["lr_max"] -self.featuresDic["lr_min"])*(1 + np.cos(epoch/self.featuresDic["max_iter"]*np.pi))

    def predict(self, X_in):
        _, Y_pred = self.forward(X_in)
        return np.round(Y_pred)

    def sigmodFunction(self, z):
        y=1/(1 + np.exp(-z))
        return y

