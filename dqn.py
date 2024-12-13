import os

import numpy as np
# 导入PyTorch和相关库
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, ByteTensor
from collections import namedtuple
import random


# 定义神经网络模型
class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[128,128]):
        super(DQNNet, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, s):
        s = s.view(s.size(0), 1, self.input_size)  # 改变输入形状，第一个维度是batch，第二个维度是1，第三个维度是35
        s = F.relu(self.linear1(s))  # 第一个全连接层
        s = F.relu(self.linear2(s))  # 第二个全连接层
        s = self.linear3(s)  # 输出
        return s


Tensor = FloatTensor

MODELS_DIR="models"
# 定义DQN智能体类
class DQN(object):
    def __init__(self, opt, agent_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_name = agent_name
        self.opt = opt
        self.net, self.target_net = DQNNet(self.opt.input_size, self.opt.output_size, self.opt.hidden_size), DQNNet(
            self.opt.input_size, self.opt.output_size, self.opt.hidden_size)  # 创建DQN神经网络和目标神经网络
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.learn_step_counter = 0  # 学习步数
        self.memory = []  # 经验回放缓冲区
        self.position = 0  # 当前经验存储位置
        self.capacity = self.opt.memory_capacity  # 经验回放缓冲区容量
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr)  # 优化器
        self.loss_func = nn.MSELoss()  # 损失函数
        # 定义经验存储结构
        self.transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))
        if self.opt.train:
            # 保存模型路径
            self.model_save_path = os.path.join(self.opt.save_dir, MODELS_DIR, self.agent_name)
            os.makedirs(self.model_save_path,exist_ok=True)
    # 选择动作的方法
    def choose_action(self, s, e):
        x = np.expand_dims(s, axis=0)  # 增加维度以匹配网络输入形状
        is_random=False
        if np.random.uniform() < 1 - e:  # 使用贪婪策略或随机策略
            actions_value = self.net.forward(torch.FloatTensor(x).to(self.device))  # 基于当前状态选择动作
            action = torch.max(actions_value, -1)[1].data.numpy()  # 选择值最大的动作
            action = action.max()  # 转为标量
        else:
            is_random=True
            action = np.random.randint(0, self.opt.action_space)  # 随机选择动作
        return action,is_random

    # 存储经验的方法
    def push_memory(self, s, a, r, s_):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 如果经验回放缓冲区未满，添加新条目
        self.memory[self.position] = self.transition(torch.unsqueeze(torch.FloatTensor(s), 0).to(self.device),
                                                     torch.unsqueeze(torch.FloatTensor(s_), 0).to(self.device), \
                                                     torch.from_numpy(np.array([a])),
                                                     torch.from_numpy(np.array([r], dtype='float32')))  #
        self.position = (self.position + 1) % self.capacity  # 更新经验存储位置

    # 从经验中获取样本的方法
    def get_sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)  # 随机采样一批经验样本
        return sample

    # 学习和更新网络参数的方法
    def learn(self):
        if self.learn_step_counter % self.opt.copy_step == 0:
            self.target_net.load_state_dict(self.net.state_dict())  # 定期更新目标网络
        self.learn_step_counter += 1

        transitions = self.get_sample(self.opt.batch_size)  # 从经验回放缓冲区中获取批次样本
        batch = self.transition(*zip(*transitions))  # 解包样本元组

        b_s = Variable(torch.cat(batch.state))  # 转为Tensor并封装为Variable
        b_s_ = Variable(torch.cat(batch.next_state))
        b_a = Variable(torch.cat(batch.action))
        b_r = Variable(torch.cat(batch.reward))

        q_eval = self.net.forward(b_s).squeeze(1).cpu().gather(1, b_a.unsqueeze(1).to(torch.int64))  # 计算Q值
        q_next = self.target_net.forward(b_s_).cpu().detach()  # 目标网络Q值
        q_target = b_r + self.opt.gamma * q_next.squeeze(1).max(1)[0].view(self.opt.batch_size, 1).t()  # 目标Q值
        loss = self.loss_func(q_eval, q_target.t())  # 计算损失
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新网络参数
        return loss

    def save_model(self):
        torch.save(self.net.state_dict(), os.path.join(self.model_save_path, "net.pt"))
        torch.save(self.target_net.state_dict(), os.path.join(self.model_save_path, "target_net.pt"))

    def load_model(self,model_dir):
        self.net.load_state_dict(torch.load(os.path.join(model_dir,self.agent_name,"net.pt")))
        self.target_net.load_state_dict(torch.load(os.path.join(model_dir,self.agent_name,"target_net.pt")))