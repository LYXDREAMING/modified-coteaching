import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def loss_coteaching(y_1, y_2, t1, t2,forget_rate, ind1,ind2,noise_or_not1, noise_or_not2,selected_indices1,selected_indices2):
    # t，一个张量，表示目标标签；noise_or_not一个布尔数组，表示样本是否为噪声
    loss_1 = (torch.abs((y_1 - t1) / (t1))) * 100
    loss_1 = loss_1.squeeze().cpu().detach().numpy()
    ind_1_sorted = np.argsort(loss_1.data) # 对第一个模型的损失进行排序，并返回排序后的索引。
    ind_1_sorted =  torch.tensor(ind_1_sorted, device='cuda:0')
    loss_1_sorted = loss_1[ind_1_sorted]  # 根据排序后的索引，对第一个模型的损失进行排序。这是个啥

    loss_2 = (torch.abs((y_2 - t2) / (t2))) * 100
    loss_2 = loss_2.squeeze().cpu().detach().numpy()
    ind_2_sorted = np.argsort(loss_2.data) # 对第一个模型的损失进行排序，并返回排序后的索引。
    ind_2_sorted = torch.tensor(ind_2_sorted, device='cuda:0')
    loss_2_sorted = loss_2[ind_2_sorted]  # 根据排序后的索引，对第一个模型的损失进行排序。这是个啥

    remember_rate = 1 - forget_rate  # 计算记忆率（remember rate）。
    num_remember1 = int(remember_rate * len(loss_1_sorted))  # 根据记忆率计算应记住的样本数量。这个样本量是模型1的？
    num_remember2 = int(remember_rate * len(loss_2_sorted))  # 根据记忆率计算应记住的样本数量。这个样本量是模型1的？

    pure_ratio_1 = torch.sum(noise_or_not1[ind1[ind_1_sorted[:num_remember1]]]) / float(num_remember1)# 计算第一个模型在记忆样本中的纯样本比例。
    pure_ratio_2 = torch.sum(noise_or_not2[ind2[ind_2_sorted[:num_remember2]]]) / float(num_remember2)# 计算第二个模型在记忆样本中的纯样本比例。

    #选择数据量小的数据
    min_numremember = min(num_remember1,num_remember2)
    ind_1_update = ind_1_sorted[:num_remember1]  # 获取第一个模型应更新的样本索引。
    ind_2_update = ind_2_sorted[:num_remember2]  # 获取第二个模型应更新的样本索引。

    selected_indices1.extend(torch.tensor(ind1[ind_1_update], device='cuda:0').cpu().tolist())
    selected_indices2.extend(torch.tensor(ind2[ind_2_update], device='cuda:0').cpu().tolist())

    # exchange
    loss_1_update = (torch.abs((y_1[ind_1_update] - t1[ind_1_update]) / (t1[ind_1_update]))) * 100
    loss_2_update = (torch.abs((y_2[ind_2_update] - t2[ind_2_update]) / (t2[ind_2_update]))) * 100
    loss_1_final = torch.sum(loss_1_update) / num_remember1
    loss_2_final = torch.sum(loss_2_update) / num_remember2
    return loss_1_final, loss_2_final, pure_ratio_1, pure_ratio_2,\
           ind_1_update.cpu(),ind_2_update.cpu(),loss_1[ind_1_update],loss_2_sorted[ind_2_update],selected_indices1,selected_indices1



def loss_coteaching1(y_1, y_2, t1,t2,forget_rate, ind1,ind2,noise_or_not1, noise_or_not2,selected_indices1,selected_indices2):
    # t，一个张量，表示目标标签；noise_or_not一个布尔数组，表示样本是否为噪声

    loss_1 = (torch.abs((y_1 - t1) / (t1))) * 100
    loss_1 = loss_1.squeeze().cpu().detach().numpy()
    ind_1_sorted = np.argsort(loss_1.data) # 对第一个模型的损失进行排序，并返回排序后的索引。
    ind_1_sorted =  torch.tensor(ind_1_sorted, device='cuda:0')
    loss_1_sorted = loss_1[ind_1_sorted]  # 根据排序后的索引，对第一个模型的损失进行排序。这是个啥

    loss_2 = (torch.abs((y_2 - t2) / (t2))) * 100
    loss_2 = loss_2.squeeze().cpu().detach().numpy()
    ind_2_sorted = np.argsort(loss_2.data) # 对第一个模型的损失进行排序，并返回排序后的索引。
    ind_2_sorted = torch.tensor(ind_2_sorted, device='cuda:0')
    loss_2_sorted = loss_2[ind_2_sorted]  # 根据排序后的索引，对第一个模型的损失进行排序。这是个啥

    remember_rate = 1 - forget_rate  # 计算记忆率（remember rate）。
    num_remember1 = int(remember_rate * len(loss_1_sorted))  # 根据记忆率计算应记住的样本数量。这个样本量是模型1的？
    num_remember2 = int(remember_rate * len(loss_2_sorted))  # 根据记忆率计算应记住的样本数量。这个样本量是模型1的？

    pure_ratio_1 = torch.sum(noise_or_not1[ind1[ind_1_sorted[:num_remember1]]]) / float(num_remember1)# 计算第一个模型在记忆样本中的纯样本比例。
    pure_ratio_2 = torch.sum(noise_or_not2[ind2[ind_2_sorted[:num_remember2]]]) / float(num_remember2)# 计算第二个模型在记忆样本中的纯样本比例。

    #选择数据量小的数据
    min_numremember = min(num_remember1,num_remember2)
    ind_1_update = ind_1_sorted[:min_numremember]  # 获取第一个模型应更新的样本索引。
    ind_2_update = ind_2_sorted[:min_numremember]  # 获取第二个模型应更新的样本索引。

    selected_indices1.extend(torch.tensor(ind1[ind_1_update], device='cuda:0').cpu().tolist())
    selected_indices2.extend(torch.tensor(ind2[ind_2_update], device='cuda:0').cpu().tolist())

    # exchange
    loss_1_update = (torch.abs((y_1[ind_2_update] - t1[ind_2_update]) / (t1[ind_2_update]))) * 100
    loss_2_update = (torch.abs((y_2[ind_1_update] - t2[ind_1_update]) / (t2[ind_1_update]))) * 100

    loss_1_final = torch.sum(loss_1_update) / min_numremember
    loss_2_final = torch.sum(loss_2_update) / min_numremember

    return loss_1_final, loss_2_final, pure_ratio_1, pure_ratio_2,\
           ind_1_update.cpu(),ind_2_update.cpu(),loss_1[ind_1_update],loss_2_sorted[ind_2_update],selected_indices1,selected_indices1



def loss_coteaching2(y_1, y_2, t1,t2, forget_rate, ind1,ind2, noise_or_not1, noise_or_not2, selected_indices1,selected_indices2):

    loss_1 = (torch.abs((y_1 - t1) / t1)) * 100
    loss_1 = loss_1.squeeze().cpu().detach().numpy()
    ind_1_sorted = np.argsort(loss_1)  # 对第一个模型的损失进行排序，并返回排序后的索引。
    ind_1_sorted = torch.tensor(ind_1_sorted, device='cuda:0')
    loss_1_sorted = loss_1[ind_1_sorted]  # 根据排序后的索引，对第一个模型的损失进行排序。

    loss_2 = (torch.abs((y_2 - t2) / t2)) * 100
    loss_2 = loss_2.squeeze().cpu().detach().numpy()
    ind_2_sorted = np.argsort(loss_2)  # 对第二个模型的损失进行排序，并返回排序后的索引。
    ind_2_sorted = torch.tensor(ind_2_sorted, device='cuda:0')
    loss_2_sorted = loss_2[ind_2_sorted]  # 根据排序后的索引，对第二个模型的损失进行排序。

    clean_ind_1 = [i for i in ind_1_sorted if noise_or_not1[ind1[i]]]  # 筛选出干净数据索引
    clean_ind_2 = [i for i in ind_2_sorted if noise_or_not2[ind2[i]]]  # 筛选出干净数据索引

    num_remember1 = len(clean_ind_1)
    num_remember2 = len(clean_ind_2)
    '''
        # 补充干净数据使得两者数量相等:向长的看齐
    if num_remember1 != num_remember2:
        smaller_indices = clean_ind_1 if num_remember1 < num_remember2 else clean_ind_2
        larger_indices = clean_ind_2 if smaller_indices is filtered_ind_1 else clean_ind_1
        additional_indices = [i for i in larger_indices if i not in smaller_indices][:abs(num_remember1 - num_remember2)]
        smaller_indices.extend(additional_indices)

    '''
    # 补充干净数据使得两者数量相等:向短的看齐
    if num_remember1 != num_remember2:
        smaller_indices = clean_ind_1 if num_remember1 < num_remember2 else clean_ind_2
        larger_indices = clean_ind_2 if smaller_indices is clean_ind_1 else clean_ind_1
        additional_indices = larger_indices[:abs(num_remember1 - num_remember2)]
        smaller_indices.extend(additional_indices)


    ind_1_update = torch.tensor(clean_ind_1[:num_remember1], device='cuda:0')
    ind_2_update = torch.tensor(clean_ind_2[:num_remember2], device='cuda:0')

    selected_indices1.extend(torch.tensor(ind1[ind_1_update], device='cuda:0').cpu().tolist())
    selected_indices2.extend(torch.tensor(ind2[ind_2_update], device='cuda:0').cpu().tolist())

    # 输出更新后的索引

    pure_ratio_1 = torch.sum(noise_or_not1[ind1[clean_ind_1[:num_remember1]]]) / float(num_remember1)# 计算第一个模型在记忆样本中的纯样本比例。
    pure_ratio_2 = torch.sum(noise_or_not2[ind2[clean_ind_1[:num_remember2]]]) / float(num_remember2)# 计算第二个模型在记忆样本中的纯样本比例。

    # 计算更新后的损失
    loss_1_update = (torch.abs((y_1[ind_2_update] - t1[ind_2_update]) / (t1[ind_2_update]))) * 100
    loss_2_update = (torch.abs((y_2[ind_1_update] - t2[ind_1_update]) / (t2[ind_1_update]))) * 100

    return torch.sum(loss_1_update) / num_remember2, torch.sum(loss_2_update) / num_remember1, pure_ratio_1, pure_ratio_2,\
           ind_1_update.cpu(), ind_2_update.cpu(), loss_1[ind_1_update], loss_2[ind_2_update],selected_indices1,selected_indices2


# 返回计算得到的损失、纯样本比例等结果。


