from __future__ import print_function
#import torch.utils.data as data
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import numpy as np
import datetime
import shutil
from sklearn.datasets import load_boston
import itertools
from loss4 import loss_coteaching
from loss4 import loss_coteaching1
from loss4 import loss_coteaching2
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import codecs
import utils
import matplotlib.pyplot as plt
import errno
import os.path
import torch.optim as optim
from clay4 import Clay
from clay4 import Clay1
from clay4 import Clay2
from clay4 import Clay3
from model import Net
from model import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--lr',type = float,default=0.005)
parser.add_argument('--result_dir',type=str,help='dir to save result txt files',default='results/')
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = 0.05)
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--dataset', type = str, help = 'clay', default = 'clay')
args = parser.parse_args()

# Seed 代码还使用torch.manual_seed()和torch.cuda.manual_seed()设置了随机种子，以保证实验的可复现性。
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size1 = 8
batch_size2 = 8

learning_rate = args.lr

# Create result directory
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
# 打开日志文件进行写入
log_file = os.path.join(args.result_dir, 'log.txt')
log = open(log_file, 'w', encoding='utf-8')

if args.dataset == 'clay':
    args.epoch_decay_start = 10 #表示衰减开始的轮数为80。
    args.n_epoch = 500 #表示训练的总轮数为200。
    train_f_dataset = Clay(train_f_dataset=True,train_g_dataset=False,
                           noise_type='unclearn')
    train_g_dataset = Clay(train_f_dataset=False, train_g_dataset=True,
                           noise_type='unclearn')

#    all_dataset = Clay2(all_dataset=True,noise_type='unclearn')


if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

noise_or_not1 = train_f_dataset.noise_or_not1
noise_or_not1 = noise_or_not1.squeeze()
noise_or_not2 = train_g_dataset.noise_or_not2
noise_or_not2 = noise_or_not2.squeeze()

# Adjust learning rate and betas for Adam Optimizer定义了变量mom1和mom2，分别表示Adam优化器中的两个动量参数
mom1 = 0.9#通常情况下，mom1较大，用于计算梯度的指数移动平均值，
mom2 = 0.1#而mom2较小，用于计算梯度平方的指数移动平均值。
alpha_plan = [learning_rate] * args.n_epoch#用于存储每个训练轮次的学习率
beta1_plan = [mom1] * args.n_epoch#存储每个训练轮次的动量参数。

for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1

# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)

# Create result directory
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
# 打开日志文件进行写入
log_file = os.path.join(args.result_dir, 'log.txt')
log = open(log_file, 'w', encoding='utf-8')

# Train and evaluate models
best_loss = float('inf')
best_epoch = 0

start_time = datetime.datetime.now()

start_time = datetime.datetime.now()

def index_value(output, y,threshold):

    ape = (torch.abs(y-output)/y)*100
    ape = ape.squeeze().cpu().detach().numpy()
    noise_or_not = ape <= threshold
    noise_or_not_tensor = torch.from_numpy(noise_or_not)

    return noise_or_not_tensor

def remove(predictions): #Remove less than 0
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i]=0
    return predictions


def train(train_loader_f,train_loader_g,epoch,model1,optimizer1,model2,optimizer2):
    selected_indices1=[]
    selected_indices2 = []
    ind_1_update_list = []
    ind_2_update_list = []
    ind_1_sorted_list = []
    ind_2_sorted_list = []
    loss_1_sorted_list = []
    loss_2_sorted_list = []
    total_smape1 = 0.0
    total_mspe1 = 0.0
    total_mape1 = 0.0
    total_R21 = 0.0
    total_train_acc1 = 0.0
    total_samples_f = 0

    total_smape2 = 0.0
    total_mspe2 = 0.0
    total_mape2 = 0.0
    total_R22 = 0.0
    total_train_acc2 = 0.0

    total_samples_g = 0

    for i_f,(X_batch_f,y_batch_f,index_f) in enumerate(train_loader_f):
        ind_f = index_f.cpu().numpy().transpose()
        if i_f>args.num_iter_per_epoch:#如果超过指定的每轮迭代次数，则跳出循环，结束该轮训练。
            break

        X_batch_f = X_batch_f.cuda()
        y_batch_f = y_batch_f.cuda()
        mean_y_true_f = torch.mean(y_batch_f)

        # Forward + Backward + Optimize
        output1= model1(X_batch_f)
        mape1_batch = torch.mean(torch.abs((y_batch_f-output1) / y_batch_f)) * 100.0
        smape1_batch = torch.mean((torch.abs(y_batch_f - output1)) / ((torch.abs(y_batch_f) + torch.abs(output1)) / 2)) * 100
        mspe1_batch = torch.mean(((y_batch_f - output1) ** 2) / y_batch_f ** 2) * 100
        R21_batch = 1-torch.sum((y_batch_f-output1)**2)/torch.sum((y_batch_f-mean_y_true_f)**2)

        for i_g,(X_batch_g,y_batch_g,index_g) in enumerate(train_loader_g):
            ind_g = index_g.cpu().numpy().transpose()
            if i_g > args.num_iter_per_epoch:  # 如果超过指定的每轮迭代次数，则跳出循环，结束该轮训练。
                break

            X_batch_g = X_batch_g.cuda()
            y_batch_g = y_batch_g.cuda()
            mean_y_true_g = torch.mean(y_batch_g)
            output2 = model2(X_batch_g)

            mape2_batch = torch.mean(torch.abs((y_batch_g - output2) / y_batch_g)) * 100.0
            smape2_batch = torch.mean((torch.abs(y_batch_g - output2)) / ((torch.abs(y_batch_g) + torch.abs(output2)) / 2)) * 100
            mspe2_batch = torch.mean(((y_batch_g - output2) ** 2) / y_batch_g ** 2) * 100
            R22_batch = 1 - torch.sum((y_batch_g - output2) ** 2) / torch.sum((y_batch_g - mean_y_true_g) ** 2)
            if epoch <= 400:

                noise_or_not_new1 = index_value(output1, y_batch_f, 15)
                true_results_1 = [result for result in noise_or_not_new1 if result]
                true_number_1 = len(true_results_1)
                noise_or_not1[ind_f] = noise_or_not_new1

                noise_or_not_new2 = index_value(output2, y_batch_g, 15)
                true_results_2 = [result for result in noise_or_not_new2 if result]
                true_number_2 = len(true_results_2)
                noise_or_not2[ind_g] = noise_or_not_new2
                if epoch <= 300:

                    loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, \
                    loss_1_sorted_update, loss_2_sorted_update, selected_indices1, selected_indices2 \
                        = loss_coteaching(output1, output2, y_batch_f,y_batch_g, 0,
                                          ind_f,ind_g, noise_or_not1, noise_or_not2, selected_indices1, selected_indices2)

                else:
                    loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, \
                    loss_1_sorted_update, loss_2_sorted_update, selected_indices1, selected_indices2 \
                        = loss_coteaching2(output1, output2, y_batch_f,y_batch_g, rate_schedule[epoch],
                                           ind_f,ind_g, noise_or_not1, noise_or_not2, selected_indices1, selected_indices2)

            else:
                noise_or_not_new1 = index_value(output1, y_batch_f, 6)
                true_results_1 = [result for result in noise_or_not_new1 if result]
                true_number_1 = len(true_results_1)
                noise_or_not1[ind_f] = noise_or_not_new1

                noise_or_not_new2 = index_value(output2, y_batch_g, 6)
                true_results_2 = [result for result in noise_or_not_new2 if result]
                true_number_2 = len(true_results_2)
                noise_or_not2[ind_g] = noise_or_not_new2

                loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, \
                loss_1_sorted_update, loss_2_sorted_update, selected_indices1, selected_indices2 \
                    = loss_coteaching2(output1, output2, y_batch_f,y_batch_g, rate_schedule[epoch],
                                       ind_f,ind_g, noise_or_not1, noise_or_not2, selected_indices1, selected_indices2)

            ind_1_update_list.append(ind_1_update.cpu().numpy())
            ind_2_update_list.append(ind_2_update.cpu().numpy())
            loss_1_sorted_list.append(loss_1_sorted_update)
            loss_2_sorted_list.append(loss_2_sorted_update)

            #loss_1.requires_grad = True
            #loss_2.requires_grad = True
            # 清除优化器梯度
            optimizer2.zero_grad()
            # 进行反向传播
            loss_2.backward()
            # 执行优化器步骤
            optimizer2.step()

            total_samples_g += len(y_batch_g)
            total_smape2 += smape2_batch.item() * len(y_batch_g)
            total_mspe2 += mspe2_batch.item() * len(y_batch_g)
            total_mape2 += mape2_batch.item() * len(y_batch_g)
            total_R22 += R22_batch.item() * len(y_batch_g)
            total_train_acc2 += float(true_number_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        total_samples_f += len(y_batch_f)
        total_smape1 += smape1_batch.item() * len(y_batch_f)
        total_mspe1 += mspe1_batch.item() * len(y_batch_f)
        total_mape1 += mape1_batch.item() * len(y_batch_f)
        total_R21 += R21_batch.item() * len(y_batch_f)
        total_train_acc1 += float(true_number_1)

        np.save('ind_1_update.npy', np.array(ind_1_update_list))
        np.save('ind_2_update.npy', np.array(ind_2_update_list))
        np.save('loss_1_sorted.npy', np.array(loss_1_sorted_list))
        np.save('loss_2_sorted.npy', np.array(loss_2_sorted_list))


    smape1 = total_smape1 /total_samples_f
    mape1 = total_mape1/total_samples_f
    mspe1 = total_mspe1 / total_samples_f
    R21 = total_R21 / total_samples_f

    smape1 = torch.tensor(smape1)
    mape1 = torch.tensor(mape1)
    mspe1 = torch.tensor(mspe1)
    R21 = torch.tensor(R21)

    smape2 = total_smape2 /total_samples_g
    mape2 = total_mape2/total_samples_g
    mspe2 = total_mspe2 / total_samples_g
    R22 = total_R22 / total_samples_g

    smape2 = torch.tensor(smape2)
    mape2 = torch.tensor(mape2)
    mspe2 = torch.tensor(mspe2)
    R22 = torch.tensor(R22)

    train_acc1 = float(true_number_1) / float(len(y_batch_f))  # 计算模型1的训练准确率。
    train_acc2 = float(true_number_2) / float(len(y_batch_g))  # 计算模型2的训练准确率。

    return mape1,mape2,smape1,smape2,mspe1,mspe2,R21,R22,train_acc1,train_acc2,selected_indices1,selected_indices2,\
           ind_1_update,ind_2_update,loss_1_sorted_update,loss_2_sorted_update

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate(test_loader_f,test_loader_g, model1, model2):
    #print('Evaluating %s...' % model_str)
    model1.eval()

    mape1 = torch.tensor(0.0)
    mae1 = torch.tensor(0.0)
    rmse1 = torch.tensor(0.0)
    R21 = torch.tensor(0.0)
    total1 = torch.tensor(0.0)

    for inputs,targets,_ in test_loader_f:
        inputs = inputs.to(device)
        targets =targets.to(device)
        outputs1 = model1(inputs)
        mean_y_true = torch.mean(targets)

        mape1 = torch.mean(torch.abs((targets - outputs1) / targets)) * 100.0
        mae1 = torch.mean(torch.abs(targets-outputs1))
        rmse1 = torch.sqrt(torch.mean((targets-outputs1)**2))
        R21 = 1-torch.sum((targets-outputs1)**2)/torch.sum((targets-mean_y_true)**2)

    model2.eval()

    mape2 = torch.tensor(0.0)
    mae2 = torch.tensor(0.0)
    rmse2 = torch.tensor(0.0)
    R22 = torch.tensor(0.0)
    total2 =torch.tensor(0.0)

    for inputs,targets,_ in test_loader_g:
        inputs = inputs.to(device)
        targets =targets.to(device)
        outputs2 = model2(inputs)
        mean_y_true = torch.mean(targets)

        mape2 = torch.mean(torch.abs((targets - outputs2) / targets)) * 100.0
        mae2 = torch.mean(torch.abs(targets-outputs2))
        rmse2 = torch.sqrt(torch.mean((targets-outputs2)**2))
        R22 = 1-torch.sum((targets-outputs2)**2)/torch.sum((targets-mean_y_true)**2)

    return mape1, mape2,mae1,mae2,rmse1,rmse2,R21,R22

def test_clean(test_loader_f,test_loader_g, model1, model2):
    selected_indices1_test=[]
    selected_indices2_test = []
    model1.eval()
    mape1 = 0
    total1 = 0
    model2.eval()
    mape2 =0
    total2 =0
    for inputs,targets,index in test_loader_f:
        ind = index.cpu().numpy().transpose()
        inputs = inputs.to(device)
        targets =targets.to(device)

        outputs1 = model1(inputs)

        noise_or_not_new1 = index_value(outputs1, targets, 20)
        true_results_1 = [result for result in noise_or_not_new1 if result]
        true_number_1 = len(true_results_1)
        noise_or_not1[ind] = noise_or_not_new1

        for inputs, targets, index in test_loader_g:
            ind = index.cpu().numpy().transpose()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs2 = model2(inputs)

            noise_or_not_new2 = index_value(outputs2, targets, 20)
            true_results_2 = [result for result in noise_or_not_new2 if result]
            true_number_2 = len(true_results_2)
            noise_or_not2[ind] = noise_or_not_new2

        loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update,\
            loss_1_sorted_update, loss_2_sorted_update, selected_indices1_test, selected_indices2_test \
                = loss_coteaching2(outputs1, outputs2, targets, 0.05,
                                   ind, noise_or_not1, noise_or_not2, selected_indices1_test, selected_indices2_test)

    #test_acc1 = float(true_number_1) / float(len(targets))  # 计算模型1的训练准确率。
    #test_acc2 = float(true_number_2) / float(len(targets))  # 计算模型2的训练准确率。

    #return test_acc1,test_acc2,selected_indices1_test,selected_indices2_test
    return selected_indices1_test,selected_indices2_test

def alldata_clean(all_loader, model1, model2):
    model1.eval()
    model2.eval()

    for X_all,y_all,index in all_loader:
        ind = index.cpu().numpy().transpose()

        X_all = X_all.to(device)
        y_all = y_all.to(device)
        mean_y_true = torch.mean(y_all)
        # 用model1 model2进行计算
        pre_y_all1 = model1(X_all)
        pre_y_all2 = model2(X_all)

        #model1误差结果
        mape1_all = torch.mean(torch.abs((y_all - pre_y_all1) /y_all)) * 100.0
        mae1_all = torch.mean(torch.abs(y_all - pre_y_all1))
        rmse1_all = torch.sqrt(torch.mean((y_all - pre_y_all1)**2))
        R21_all = 1-torch.sum((y_all - pre_y_all1)**2)/torch.sum((y_all-mean_y_true)**2)
        #model2误差结果
        mape2_all = torch.mean(torch.abs((y_all - pre_y_all2) /y_all)) * 100.0
        mae2_all = torch.mean(torch.abs(y_all - pre_y_all2))
        rmse2_all = torch.sqrt(torch.mean((y_all - pre_y_all2)**2))
        R22_all = 1-torch.sum((y_all - pre_y_all2)**2)/torch.sum((y_all-mean_y_true)**2)


    return mape1_all, mape2_all, mae1_all, mae2_all, rmse1_all, rmse2_all, R21_all, R22_all


def smooth_curve(points, window_size=10):
    # Create a window with equal weights
    window = np.ones(window_size) / window_size
    return np.convolve(points, window, mode='valid')

def main():
    # Data Loader (Input Pipeline)
    torch.autograd.set_detect_anomaly(True)

    print('loading dataset...')
    train_loader_f = torch.utils.data.DataLoader(dataset=train_f_dataset, batch_size=batch_size1,drop_last = True,shuffle=True)
    train_loader_g = torch.utils.data.DataLoader(dataset=train_g_dataset, batch_size=batch_size2,drop_last = True,shuffle=True)

    # Define models
    print('building model...')

    '''
    这块先基于之前训练好的DNN模型参数，进行训练，将模型训练好了以后，在进行coteaching
    '''
    input_size = 8
    model1 = build_model(input_size)
    model2 = build_model(input_size)
    model1.cuda()
    model2.cuda()

    # Define optimizer
    optimizer1 = torch.optim.RMSprop(model1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=args.lr)
    epoch=0

    # evaluate models with random weights

    best_loss = float('inf')
    best_epoch = 0
    train_smape1_values = []
    train_smape2_values = []

    train_mape1_values = []
    train_mape2_values = []

    train_mspe1_values = []
    train_mspe2_values = []

    train_R21_values = []
    train_R22_values = []
    plot_interval = 50  # 设置每隔多少个epoch出一次图
    plot_counter = 0

    for epoch in range(1, args.n_epoch):
        # train models
        model1.train()
        adjust_learning_rate(optimizer1,epoch)
        model2.train()
        adjust_learning_rate(optimizer2,epoch)
        train_mape1, train_mape2, train_smape1, train_smape2, train_mspe1, train_mspe2, train_R21, train_R22, \
        train_acc1, train_acc2, selected_indices1, selected_indices2, ind_1_update, ind_2_update, loss_1_sorted, loss_2_sorted \
            = train(train_loader_f, train_loader_g,epoch, model1, optimizer1, model2, optimizer2)

        print(
            'Epoch [%d/%d],Iter [%d/%d] Iter [%d/%d] mape1: %.4F%%, mape2: %.4f%%,smape1: %.4F, smape2: %.4f,mspe1: %.4F, mspe2: %.4f,R21: %.4F, R22: %.4f,'
            'acc1: %.4f, acc2: %.4f,'
            % (epoch + 1, args.n_epoch, i + 1, len(train_f_dataset) // batch_size1,i + 1,len(train_g_dataset) // batch_size2, float(train_mape1), float(train_mape2),
               float(train_smape1), float(train_smape2), float(train_mspe1), float(train_mspe2), float(train_R21), float(train_R22)
               , float(train_acc1), float(train_acc2)))
        print(
            'Epoch [%d/%d] Model1 Test Accuracy on the %s train_F, Model2 Test Accuracy on the %s train_G' % (
                epoch + 1, args.n_epoch, len(train_f_dataset), len(train_g_dataset)))


        # 用于保存每个epoch的train和test的MAE值

        # 记录train和test的MAE值
        train_smape1_values.append(train_smape1.cpu())  # 将train_mae1从GPU上移动到CPU上
        train_smape2_values.append(train_smape2.cpu())  # 将train_mae1从GPU上移动到CPU上

        train_mape1_values.append(train_mape1.cpu())  # 将train_mae1从GPU上移动到CPU上
        train_mape2_values.append(train_mape2.cpu())  # 将train_mae1从GPU上移动到CPU上

        train_mspe1_values.append(train_mspe1.cpu())   # 将train_mae1从GPU上移动到CPU上
        train_mspe2_values.append(train_mspe2.cpu())  # 将train_mae1从GPU上移动到CPU上

        train_R21_values.append(train_R21.cpu())  # 将train_mae1从GPU上移动到CPU上
        train_R22_values.append(train_R22.cpu())  # 将train_mae1从GPU上移动到CPU上


        # 绘制train和test集的MAE曲线
    epochs = list(range(1, args.n_epoch))
    # 用test的数据导入需要进行模型训练，用model和loss得到干净数据


    # 将Tensor对象转换为NumPy数组
    train_smape1_values_np = [train_smape1.detach().numpy() for train_smape1 in train_smape1_values]
    train_smape2_values_np = [train_smape2.detach().numpy() for train_smape2 in train_smape2_values]

    train_mape1_values_np = [train_mape1.detach().numpy() for train_mape1 in train_mape1_values]
    train_mape2_values_np = [train_mape2.detach().numpy() for train_mape2 in train_mape2_values]

    train_mspe1_values_np = [train_mspe1.detach().numpy() for train_mspe1 in train_mspe1_values]
    train_mspe2_values_np = [train_mspe2.detach().numpy() for train_mspe2 in train_mspe2_values]

    train_R21_values_np = [train_R21.detach().numpy() for train_R21 in train_R21_values]
    train_R22_values_np = [train_R22.detach().numpy() for train_R22 in train_R22_values]

    window_size=10
    plt.plot(epochs[window_size - 1:], smooth_curve(train_smape1_values_np, window_size=10), marker='o', linestyle='-',
             color='red', label='Smoothed Train SMAPE1')
    plt.plot(epochs[window_size - 1:], smooth_curve(train_smape2_values_np, window_size=10), marker='o', linestyle='-',
             color='green', label='Smoothed Train SMAPE2')

    plt.xlabel('Epoch')
    plt.ylabel('SMAPE')
    plt.title('Smoothed Train and Test SMAPE over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(epochs[window_size - 1:], smooth_curve(train_mape1_values_np, window_size=10), marker='o', linestyle='-',
             color='red', label='Smoothed Train MAPE1')
    plt.plot(epochs[window_size - 1:], smooth_curve(train_mape2_values_np, window_size=10), marker='o', linestyle='-',
             color='green', label='Smoothed Train MAPE2')

    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.title('Smoothed Train and Test MAPE over Epochs-model3')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(epochs[window_size - 1:], smooth_curve(train_mspe1_values_np, window_size=10), marker='o', linestyle='-',
             color='red', label='Smoothed Train MSPE1')
    plt.plot(epochs[window_size - 1:], smooth_curve(train_mspe2_values_np, window_size=10), marker='o', linestyle='-',
             color='green', label='Smoothed Train MSPE2')

    plt.xlabel('Epoch')
    plt.ylabel('MSPE')
    plt.title('Smoothed Train and Test MSPE over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(epochs[window_size - 1:], smooth_curve(train_R21_values_np, window_size=10), marker='o', linestyle='-',
             color='red', label='Smoothed Train R21')
    plt.plot(epochs[window_size - 1:], smooth_curve(train_R22_values_np, window_size=10), marker='o', linestyle='-',
             color='green', label='Smoothed Train R22')


    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.title('Smoothed Train and Test R2 over Epochs-model3')
    plt.grid(True)
    plt.legend()
    plt.show()
    print('===================seleted==================')
    print(selected_indices1)
    print(selected_indices2)

if __name__ == '__main__':
    main()
