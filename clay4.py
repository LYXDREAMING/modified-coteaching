import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

import sys
pd.set_option('display.width', None) # 设置字符显示无限制
pd.set_option('display.max_rows', None) # 设置行数显示无限制
np.set_printoptions(threshold=sys.maxsize)
class Clay(Dataset):#适用方案-1-4
    def __init__(self, train_f_dataset = True,train_g_dataset = True,
                 noise_type='unclearn'):

        names = ['no', 'su.s', 'k.s', 'L/D.s', 'a.s', 'x.s', 'h.s', 'langle.s', 'tk.s', 'f', 'p','type']
        data1 = pd.read_csv("experimentclay3.csv", names=names)
        data2_1 = pd.read_csv('clean-514-numclay.csv',names=names)
        #按照设定的比例进行抽样，这个模型抽样的是35个num数据
        sample_percentage =1# 给定抽取比例，获得参数结果，从数值数据中抽样的比例
        sampled_Data = data2_1.sample(frac=sample_percentage,random_state=args.seed)

        X1 = data1[names[1:9]]
        y1 = data1[names[10]]
        t1 = data1[names[11]]
        X2 = sampled_Data[names[1:9]]
        y2 = sampled_Data[names[10]]
        t2 = sampled_Data[names[11]]
        #可以解释为X1=(1-test_size1)*len(experimentclay3.csv)+(1-test_size2)*len(clay_num_514clean.csv)
        #X2=(test_size1)*len(experimentclay3.csv)+(test_size2)*len(clay_num_514clean.csv)
        X1_f,X1_g,y1_f,y1_g = train_test_split(X1,y1,test_size = 0.33,
                                               random_state = args.seed)#X1_f,y1_f占X1,y1的0.999
        X1_f, X1_g, t1_f, t1_g = train_test_split(X1, t1, test_size=0.33,
                                                  random_state=args.seed)  # X1_f,y1_f占X1,y1的0.999
        #

        X2_g = X2
        y2_g = y2
        t2_g = t2

        Xg_combined = np.concatenate((X1_g,X2_g),axis=0)
        yg_combined = np.concatenate((y1_g,y2_g),axis=0)
        tg_combined = np.concatenate((t1_g, t2_g), axis=0)

        column_names = names[1:9] + ['target']+ ['type']

        data_f = np.hstack((X1_f,y1_f.values.reshape(-1, 1)))
        data_f2 = np.hstack((data_f, t1_f.values.reshape(-1, 1)))
        df_data_f = pd.DataFrame(data_f2, columns=column_names)

        data_g = np.hstack((Xg_combined, yg_combined.reshape(-1, 1)))
        data_g2 = np.hstack((data_g, tg_combined.reshape(-1, 1)))
        dg_data_g = pd.DataFrame(data_g2, columns=column_names)

        #分别设定modelf 和modelg的训练集和测试集
        X_f = df_data_f.iloc[:, 0:8]
        y_f = df_data_f['target']
        X_f_train = X_f
        y_f_train = y_f

        X_g = dg_data_g.iloc[:, 0:8]
        y_g = dg_data_g['target']
        X_g_train= X_g
        y_g_train = y_g

        X_f_train_indices = X_f_train.index.tolist()  # Get the indices of the train set samples
        X_g_train_indices = X_g_train.index.tolist()  # Get the indices of the train set samples
        print("Indices of samples in the train X_f_train_indices:",X_f_train_indices)
        print("Indices of samples in the train X_g_train_indices:",X_g_train_indices)
        print("Indices of samples in the train df_data_f", df_data_f)  # 输出抽取得到的F模型的所有数据
        print("Indices of samples in the train dg_data_g", dg_data_g)  # 输出抽取得到的G模型的所有数据
        # Standardize the features
        scaler = StandardScaler()
        X_f_train = scaler.fit_transform(X_f_train)

        X_g_train = scaler.fit_transform(X_g_train)

        # Convert data to tensors
        X_f_train = torch.from_numpy(X_f_train).float()
        y_f_train_numpy = y_f_train.values
        y_f_train_numpy = y_f_train_numpy.astype(float)
        y_f_train = torch.from_numpy(y_f_train_numpy).float().unsqueeze(1)

        X_g_train = torch.from_numpy(X_g_train).float()
        y_g_train_numpy = y_g_train.values
        y_g_train_numpy = y_g_train_numpy.astype(float)
        y_g_train = torch.from_numpy(y_g_train_numpy).float().unsqueeze(1)

        self.train_f_dataset = train_f_dataset
        self.train_g_dataset = train_g_dataset

        self.noise_type = noise_type
        if self.train_f_dataset:
            self.X_f_train = X_f_train
            self.y_f_train = y_f_train
            self.noise_or_not1 = np.transpose(self.y_f_train) == np.transpose(self.y_f_train)
        else:
            self.X_g_train = X_g_train
            self.y_g_train = y_g_train
            self.noise_or_not2 = np.transpose(self.y_g_train) == np.transpose(self.y_g_train)

    def __getitem__(self, index):
        if self.train_f_dataset:
            X = self.X_f_train[index]
            y = self.y_f_train[index]
        else:
            X = self.X_g_train[index]
            y = self.y_g_train[index]


        return X, y, index

    def __len__(self):
        if self.train_f_dataset:
            return len(self.X_f_train)
        else:
            return len(self.X_g_train)

class Clay1(Dataset):#适用方案-2-3-8
    def __init__(self, train_f_dataset = True,train_g_dataset = True,
                 noise_type='unclearn'):

        names = ['no', 'su.s', 'k.s', 'L/D.s', 'a.s', 'x.s', 'h.s', 'langle.s', 'tk.s', 'f', 'p','type']
        data1 = pd.read_csv("experimentclay3.csv", names=names)
        data2_1 = pd.read_csv('clean-514-numclay.csv',names=names)
        #按照设定的比例进行抽样，这个模型抽样的是35个num数据
        sample_percentage =1# 给定抽取比例，获得参数结果，从数值数据中抽样的比例
        sampled_Data = data2_1.sample(frac=sample_percentage,random_state=args.seed)

        X1 = data1[names[1:9]]
        y1 = data1[names[10]]
        t1 = data1[names[11]]
        X2 = sampled_Data[names[1:9]]
        y2 = sampled_Data[names[10]]
        t2 = sampled_Data[names[11]]
        # 除了5
        # 可以解释为X1=(1-test_size1)*len(experimentclay3.csv)+(1-test_size2)*len(clay_num_514clean.csv)
        # X2=(test_size1)*len(experimentclay3.csv)+(test_size2)*len(clay_num_514clean.csv)
        X1_f, X1_g, y1_f, y1_g = train_test_split(X1, y1, test_size=0.5,
                                                  random_state=args.seed)  # X1_f,y1_f占X1,y1的0.999
        X1_f, X1_g, t1_f, t1_g = train_test_split(X1, t1, test_size=0.5,
                                                  random_state=args.seed)  # X1_f,y1_f占X1,y1的0.999
        X2_f, X2_g, y2_f, y2_g = train_test_split(X2, y2, test_size=0.5,
                                                  random_state=args.seed)  # X2_f,y2_f占X2,y2的0.01
        X2_f, X2_g, t2_f, t2_g = train_test_split(X2, t2, test_size=0.5,
                                                  random_state=args.seed)  # X2_f,y2_f占X2,y2的0.01
        Xf_combined = np.concatenate((X1_f, X2_f), axis=0)
        yf_combined = np.concatenate((y1_f, y2_f), axis=0)
        print('lenth of XF', len(Xf_combined))
        Xg_combined = np.concatenate((X1_g, X2_g), axis=0)
        yg_combined = np.concatenate((y1_g, y2_g), axis=0)
        tf_combined = np.concatenate((t1_f, t2_f), axis=0)
        print('lenth of TF', len(tf_combined))
        tg_combined = np.concatenate((t1_g, t2_g), axis=0)
        X_f1 = X1_f.index.tolist()  # Get the indices of the train set samples
        X_f2 = X2_f.index.tolist()  # Get the indices of the train set samples
        X_g1 = X1_g.index.tolist()  # Get the indices of the train set samples
        X_g2 = X2_g.index.tolist()  # Get the indices of the train set samples
        print("Indices of samples in the train X_f1_indices:", X_f1)
        print("Indices of samples in the train X_f2_indices:", X_f2)
        print("Indices of samples in the train X_g1_indices:", X_g1)
        print("Indices of samples in the train X_g2_indices:", X_g2)
        data_f = np.hstack((Xf_combined, yf_combined.reshape(-1, 1)))
        data_f2 = np.hstack((data_f, tf_combined.reshape(-1, 1)))
        column_names = names[1:9] + ['target'] + ['type']
        df_data_f = pd.DataFrame(data_f2, columns=column_names)

        print("Indices of samples in the train df_data_f", df_data_f)  # 输出抽取得到的F模型的所有数据

        data_g = np.hstack((Xg_combined, yg_combined.reshape(-1, 1)))
        data_g2 = np.hstack((data_g, tg_combined.reshape(-1, 1)))
        dg_data_g = pd.DataFrame(data_g2, columns=column_names)

        print("Indices of samples in the train dg_data_g", dg_data_g)  # 输出抽取得到的G模型的所有数据

        # 分别设定modelf 和modelg的训练集和测试集
        X_f = df_data_f.iloc[:, 0:8]
        y_f = df_data_f['target']
        X_f_train = X_f
        y_f_train = y_f

        X_g = dg_data_g.iloc[:, 0:8]
        y_g = dg_data_g['target']
        X_g_train = X_g
        y_g_train = y_g

        X_f_train_indices = X_f_train.index.tolist()  # Get the indices of the train set samples
        X_g_train_indices = X_g_train.index.tolist()  # Get the indices of the train set samples
        print("Indices of samples in the train X_f_train_indices:", X_f_train_indices)
        print("Indices of samples in the train X_g_train_indices:", X_g_train_indices)

        # Standardize the features
        scaler = StandardScaler()
        X_f_train = scaler.fit_transform(X_f_train)

        X_g_train = scaler.fit_transform(X_g_train)

        # Convert data to tensors
        X_f_train = torch.from_numpy(X_f_train).float()
        y_f_train_numpy = y_f_train.values
        y_f_train_numpy = y_f_train_numpy.astype(float)
        y_f_train = torch.from_numpy(y_f_train_numpy).float().unsqueeze(1)

        X_g_train = torch.from_numpy(X_g_train).float()
        y_g_train_numpy = y_g_train.values
        y_g_train_numpy = y_g_train_numpy.astype(float)
        y_g_train = torch.from_numpy(y_g_train_numpy).float().unsqueeze(1)

        self.train_f_dataset = train_f_dataset
        self.train_g_dataset = train_g_dataset

        self.noise_type = noise_type
        if self.train_f_dataset:
            self.X_f_train = X_f_train
            self.y_f_train = y_f_train
            self.noise_or_not1 = np.transpose(self.y_f_train) == np.transpose(self.y_f_train)
        else:
            self.X_g_train = X_g_train
            self.y_g_train = y_g_train
            self.noise_or_not2 = np.transpose(self.y_g_train) == np.transpose(self.y_g_train)

    def __getitem__(self, index):
        if self.train_f_dataset:
            X = self.X_f_train[index]
            y = self.y_f_train[index]
        else:
            X = self.X_g_train[index]
            y = self.y_g_train[index]

        return X, y, index

    def __len__(self):
        if self.train_f_dataset:
            return len(self.X_f_train)
        else:
            return len(self.X_g_train)
class Clay2(Dataset):#适用方案-6-7
    def __init__(self, train_f_dataset = True,train_g_dataset = True,
                 noise_type='unclearn'):

        names = ['no', 'su.s', 'k.s', 'L/D.s', 'a.s', 'x.s', 'h.s', 'langle.s', 'tk.s', 'f', 'p']
        data1 = pd.read_csv("experimentclay3.csv", names=names)
        data2_1 = pd.read_csv('clay_num_514clean.csv',names=names)
        #按照设定的比例进行抽样，这个模型抽样的是35个num数据
        sample_percentage =0.082# 给定抽取比例，获得参数结果，从数值数据中抽样的比例
        sampled_Data = data2_1.sample(frac=sample_percentage,random_state=args.seed)

        X1 = data1[names[1:9]]
        y1 = data1[names[10]]
        X2 = sampled_Data[names[1:9]]
        y2 = sampled_Data[names[10]]

        #可以解释为X1=(1-test_size1)*len(experimentclay3.csv)+(1-test_size2)*len(clay_num_514clean.csv)
        #X2=(test_size1)*len(experimentclay3.csv)+(test_size2)*len(clay_num_514clean.csv)

        X1_f = X1
        y1_f = y1
        X2_f,X2_g,y2_f,y2_g = train_test_split(X2,y2,test_size = 0.4,
                                               random_state = args.seed)#X1_f,y1_f占X1,y1的0.999
        Xf_combined = np.concatenate((X1_f,X2_f),axis=0)
        yf_combined = np.concatenate((y1_f,y2_f),axis=0)


        data_f = np.hstack((Xf_combined, yf_combined.reshape(-1, 1)))
        column_names = names[1:9] + ['target']
        df_data_f = pd.DataFrame(data_f, columns=column_names)
        data_g = np.hstack((X2_g, y2_g.values.reshape(-1, 1)))

        dg_data_g = pd.DataFrame(data_g, columns=column_names)
        #分别设定modelf 和modelg的训练集和测试集
        X_f = df_data_f.iloc[:, 0:8]
        y_f = df_data_f['target']
        X_f_train = X_f
        y_f_train = y_f


        X_g = dg_data_g.iloc[:, 0:8]
        y_g = dg_data_g['target']
        X_g_train= X_g
        y_g_train = y_g

        X_f_train_indices = X_f_train.index.tolist()  # Get the indices of the train set samples
        X_g_train_indices = X_g_train.index.tolist()  # Get the indices of the train set samples
        print("Indices of samples in the train X_f_train_indices:",X_f_train_indices)
        print("Indices of samples in the train X_g_train_indices:",X_g_train_indices)


        # Standardize the features
        scaler = StandardScaler()
        X_f_train = scaler.fit_transform(X_f_train)

        X_g_train = scaler.fit_transform(X_g_train)

        # Convert data to tensors
        X_f_train = torch.from_numpy(X_f_train).float()
        y_f_train_numpy = y_f_train.values
        y_f_train = torch.from_numpy(y_f_train_numpy).float().unsqueeze(1)

        X_g_train = torch.from_numpy(X_g_train).float()
        y_g_train_numpy = y_g_train.values
        y_g_train = torch.from_numpy(y_g_train_numpy).float().unsqueeze(1)

        self.train_f_dataset = train_f_dataset
        self.train_g_dataset = train_g_dataset

        self.noise_type = noise_type
        if self.train_f_dataset:
            self.X_f_train = X_f_train
            self.y_f_train = y_f_train
            self.noise_or_not1 = np.transpose(self.y_f_train) == np.transpose(self.y_f_train)
        else:
            self.X_g_train = X_g_train
            self.y_g_train = y_g_train
            self.noise_or_not2 = np.transpose(self.y_g_train) == np.transpose(self.y_g_train)

    def __getitem__(self, index):
        if self.train_f_dataset:
            X = self.X_f_train[index]
            y = self.y_f_train[index]
        else:
            X = self.X_g_train[index]
            y = self.y_g_train[index]


        return X, y, index

    def __len__(self):
        if self.train_f_dataset:
            return len(self.X_f_train)
        else:
            return len(self.X_g_train)

class Clay3(Dataset):#适用方案-6-7
    def __init__(self, train_f_dataset = True,train_g_dataset = True,
                 noise_type='unclearn'):

        names = ['no', 'su.s', 'k.s', 'L/D.s', 'a.s', 'x.s', 'h.s', 'langle.s', 'tk.s', 'f', 'p']
        data1 = pd.read_csv("experimentclay3.csv", names=names)
        data2_1 = pd.read_csv('clay_num_514clean.csv',names=names)
        #按照设定的比例进行抽样，这个模型抽样的是35个num数据
        sample_percentage =0.082# 给定抽取比例，获得参数结果，从数值数据中抽样的比例
        sampled_Data = data2_1.sample(frac=sample_percentage,random_state=args.seed)

        X1 = data1[names[1:9]]
        y1 = data1[names[10]]
        X2 = sampled_Data[names[1:9]]
        y2 = sampled_Data[names[10]]

        #可以解释为X1=(1-test_size1)*len(experimentclay3.csv)+(1-test_size2)*len(clay_num_514clean.csv)
        #X2=(test_size1)*len(experimentclay3.csv)+(test_size2)*len(clay_num_514clean.csv)

        #

        data_f = np.hstack((X1, y1.values.reshape(-1, 1)))
        column_names = names[1:9] + ['target']
        df_data_f = pd.DataFrame(data_f, columns=column_names)

        data_g = np.hstack((X2, y2.values.reshape(-1, 1)))
        dg_data_g = pd.DataFrame(data_g, columns=column_names)
        #分别设定modelf 和modelg的训练集和测试集
        X_f = df_data_f.iloc[:, 0:8]
        y_f = df_data_f['target']
        X_f_train = X_f
        y_f_train = y_f


        X_g = dg_data_g.iloc[:, 0:8]
        y_g = dg_data_g['target']
        X_g_train= X_g
        y_g_train = y_g

        X_f_train_indices = X_f_train.index.tolist()  # Get the indices of the train set samples
        X_g_train_indices = X_g_train.index.tolist()  # Get the indices of the train set samples
        print("Indices of samples in the train X_f_train_indices:",X_f_train_indices)
        print("Indices of samples in the train X_g_train_indices:",X_g_train_indices)


        # Standardize the features
        scaler = StandardScaler()
        X_f_train = scaler.fit_transform(X_f_train)

        X_g_train = scaler.fit_transform(X_g_train)

        # Convert data to tensors
        X_f_train = torch.from_numpy(X_f_train).float()
        y_f_train_numpy = y_f_train.values
        y_f_train = torch.from_numpy(y_f_train_numpy).float().unsqueeze(1)

        X_g_train = torch.from_numpy(X_g_train).float()
        y_g_train_numpy = y_g_train.values
        y_g_train = torch.from_numpy(y_g_train_numpy).float().unsqueeze(1)

        self.train_f_dataset = train_f_dataset
        self.train_g_dataset = train_g_dataset

        self.noise_type = noise_type
        if self.train_f_dataset:
            self.X_f_train = X_f_train
            self.y_f_train = y_f_train
            self.noise_or_not1 = np.transpose(self.y_f_train) == np.transpose(self.y_f_train)
        else:
            self.X_g_train = X_g_train
            self.y_g_train = y_g_train
            self.noise_or_not2 = np.transpose(self.y_g_train) == np.transpose(self.y_g_train)

    def __getitem__(self, index):
        if self.train_f_dataset:
            X = self.X_f_train[index]
            y = self.y_f_train[index]
        else:
            X = self.X_g_train[index]
            y = self.y_g_train[index]


        return X, y, index

    def __len__(self):
        if self.train_f_dataset:
            return len(self.X_f_train)
        else:
            return len(self.X_g_train)