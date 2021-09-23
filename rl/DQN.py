import numpy as np
import joblib
import pandas as pd
import os
from tqdm import tqdm
import torch
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random
import pickle
import time

import bgp.simglucose.envs.simglucose_gym_env as bgp_env
from bgp.rl import reward_functions
import matplotlib.pyplot as plt

def reward_name_to_function(reward_name):
    if reward_name == 'risk_diff':
        reward_fun = reward_functions.risk_diff
    elif reward_name == 'risk_diff_bg':
        reward_fun = reward_functions.risk_diff_bg
    elif reward_name == 'risk':
        reward_fun = reward_functions.reward_risk
    elif reward_name == 'risk_bg':
        reward_fun = reward_functions.risk_bg
    elif reward_name == 'magni_bg':
        reward_fun = reward_functions.magni_reward
    elif reward_name == 'cameron_bg':
        reward_fun = reward_functions.cameron_reward
    elif reward_name == 'eps_risk':
        reward_fun = reward_functions.epsilon_risk
    elif reward_name == 'target_bg':
        reward_fun = reward_functions.reward_target
    elif reward_name == 'cgm_high':
        reward_fun = reward_functions.reward_cgm_high
    elif reward_name == 'bg_high':
        reward_fun = reward_functions.reward_bg_high
    elif reward_name == 'cgm_low':
        reward_fun = reward_functions.reward_cgm_low
    else:
        raise ValueError('{} not a proper reward_name'.format(reward_name))
    return reward_fun

# 2. Define the network used in both target net and the net for training
class Net(nn.Module):
    def __init__(self,input_size,output_size):
        # Define the network structure, a very simple fully connected network
        super(Net, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(np.prod(input_size), 256)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc1
        self.out = nn.Linear(256, output_size)  # layer 2
        self.out.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc2

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class CNNQ(nn.Module):
    def __init__(self, input_size, output_size, init_w=3e-3):
        super(CNNQ,self).__init__()
        self.channel_size=input_size[0]
        self.signal_length=input_size[1]
        self.convolution=nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(in_channels=self.channel_size, out_channels=32, kernel_size=3)),
            ('bn1_1', nn.BatchNorm1d(num_features=32)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)),
            ('bn1_2', nn.BatchNorm1d(num_features=32)),
            ('relu1_2', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(kernel_size=2)),

            ('conv2_1', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)),
            ('bn2_1', nn.BatchNorm1d(num_features=64)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)),
            ('bn2_2', nn.BatchNorm1d(num_features=64)),
            ('relu2_2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(kernel_size=2))
        ]))

        feature_size = self.determine_feature_size(input_size)

        self.dense = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features=feature_size, out_features=512)),
            # ('bn_d', nn.BatchNorm1d(num_features=512)),
            ('relu_d', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2))
        ]))

        self.last_fc = nn.Linear(512, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.action_size = output_size

    def determine_feature_size(self, input_size):
        with torch.no_grad():
            fake_input = Variable(torch.randn(input_size)[None, :, :])
            fake_out = self.convolution(fake_input)
        return fake_out.view(-1).shape[0]

    def forward(self, input, action_input=None):
        if action_input is not None:
            input = input.reshape((-1, self.channel_size-1, self.signal_length))
            action_stack = tuple(action_input.flatten() for _ in range(self.signal_length))
            action_stack = torch.stack(action_stack).transpose(0, 1)[:, None, :]
            input = torch.cat((action_stack, input), dim=1)
        else:
            input = input.reshape((-1, self.channel_size, self.signal_length))
        feat = self.convolution(input)
        feat = feat.view(input.size(0), -1)
        feat = self.dense(feat)
        return self.last_fc(feat)

class GRUQ(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=1, init_w=3e-3, dilation=False):
        super(GRUQ, self).__init__()
        self.channel_size = input_size[0]
        self.signal_length = input_size[1]

        self.features = nn.GRU(input_size=self.channel_size,
                                   hidden_size=hidden_size, num_layers=num_layers,
                                   batch_first=True)

        self.last_fc = nn.Linear(hidden_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.action_size = output_size

    def forward(self, input, action_input=None):
        if action_input is not None:
            input = input.reshape(-1, self.channel_size-1, self.signal_length).permute(0, 2, 1)
            action_stack = tuple(action_input.flatten() for _ in range(self.signal_length))
            action_stack = torch.stack(action_stack).transpose(0, 1)[:, :, None].float()
            input = torch.cat((action_stack, input), dim=2)
        else:
            input = input.reshape(-1, self.channel_size, self.signal_length).permute(0, 2, 1)
        h, _ = self.features(input)
        feat = h[:, -1, :]
        return self.last_fc(feat)
# 3. Define the DQN network and its corresponding methods
class DQN(object):
    def __init__(self,input_size,output_size,device,net_type,LR=1e-3,MEMORY_CAPACITY=1e6,epsilon=0.9,discount_rate=0.99,BATCH_SIZE=256):

        self.lr=LR
        # -----------Define 2 networks (target and training)------#
        if net_type=='DQN':
            self.eval_net, self.target_net = Net(input_size,output_size).to(device), Net(input_size,output_size).to(device)
        elif net_type=='CNNQ':
            self.eval_net, self.target_net = CNNQ(input_size,output_size).to(device), CNNQ(input_size,output_size).to(device)
            self.lr=1e-3
        elif net_type == 'GRUQ':
            self.eval_net, self.target_net = GRUQ(input_size, output_size).to(device), GRUQ(input_size, output_size).to(
                device)
            self.lr=1e-5

        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # ----Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((MEMORY_CAPACITY, np.prod(input_size) * 2 + 2))

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        # ------Define the loss function-----#
        # self.loss_func = nn.MSELoss()
        self.loss_func = nn.SmoothL1Loss().to(device)
        self.device=device
        self.EPSILON=epsilon
        self.N_ACTIONS=output_size
        self.discount_rate=discount_rate
        self.batch_szie=BATCH_SIZE
        self.TARGET_NETWORK_REPLACE_FREQ=100
        self.N_STATES=np.prod(input_size)
        self.MEMORY_CAPACITY=MEMORY_CAPACITY

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy

        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)  # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < self.EPSILON:  # greedy
            # use epsilon-greedy approach to take action
            self.eval_net.eval()
            with torch.no_grad():
                actions_value = self.eval_net.forward(x).cpu()
            self.eval_net.train()
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]   # return the argmax index
        else:  # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % self.TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.batch_szie)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.N_STATES])).to(self.device)
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1].astype(int))).to(self.device)
        b_r = Variable(torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES + 2])).to(self.device)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.N_STATES:])).to(self.device)

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # print(q_eval.shape)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # print(q_next.shape)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + self.discount_rate * q_next.max(1)[0].view(self.batch_szie, 1)  # (batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

    def load_model(self,path):
        self.eval_net=torch.load(path)
        self.target_net=torch.load(path)


def run_train(variant):
    algo_params=variant['algo_params']

    reward_fun=reward_name_to_function(variant['reward_fun'])
    env = bgp_env.DeepSACT1DEnv(reward_fun=reward_fun,
                                patient_name=variant['patient_name'],
                                seeds={'numpy': variant['base_seed'],
                                       'sensor': variant['base_seed'],
                                       'scenario': variant['base_seed']},
                                reset_lim=variant['reset_lim'], time=variant['include_time'],
                                meal=variant['include_meal'], bw_meals=variant['bw_meals'],
                                load=variant['load'], use_pid_load=variant['use_pid_load'],
                                hist_init=variant['hist_init'],
                                gt=variant['use_ground_truth'], n_hours=variant['n_hours'],
                                norm=variant['norm'], time_std=variant['time_std'],
                                use_old_patient_env=variant['use_old_patient_env'], action_cap=variant['action_cap'],
                                action_bias=variant['action_bias'], action_scale=variant['action_scale'],
                                basal_scaling=variant['basal_scaling'],
                                meal_announce=variant['meal_announce'], residual_basal=variant['residual_basal'],
                                residual_bolus=variant['residual_bolus'], residual_PID=variant['residual_PID'],
                                fake_gt=variant['fake_gt'], fake_real=variant['fake_real'],
                                suppress_carbs=variant['suppress_carbs'], limited_gt=variant['limited_gt'],
                                termination_penalty=variant['termination_penalty'], weekly=variant['weekly'],
                                update_seed_on_reset=variant['update_seed_on_reset'],
                                deterministic_meal_size=variant['deterministic_meal_size'],
                                deterministic_meal_time=variant['deterministic_meal_time'],
                                deterministic_meal_occurrence=variant['deterministic_meal_occurrence'],
                                harrison_benedict=variant['harrison_benedict_sched'],
                                restricted_carb=variant['restricted_sched'], meal_duration=variant['meal_duration'],
                                rolling_insulin_lim=variant['rolling_insulin_lim'], universal=variant['universal'],
                                reward_bias=variant['reward_bias'], carb_error_std=variant['carb_error_std'],
                                carb_miss_prob=variant['carb_miss_prob'], source_dir=variant['source_dir'])
    BAS=env.ideal_basal

    print(BAS)
    ACTION_SPACE=[0,BAS,5*BAS]
    print(ACTION_SPACE)
    N_ACTIONS=len(ACTION_SPACE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_type=variant['net_type']
    dqn=DQN(env.observation_space.shape,N_ACTIONS,device,net_type,LR=1e-3,MEMORY_CAPACITY=algo_params['replay_buffer_size'],
                epsilon=0.95,discount_rate=0.99,BATCH_SIZE=256)
    patient=variant['patient_name']
    seed=variant['base_seed']
    save_path=f'saves/{patient}_{seed}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_last_epoch=None
    model_max_reward=None
    best_ep_r=0

    print('当前病人为{},当前网络为{}'.format(variant['patient_name'],net_type))
    print("---------Begin train------------")
    ep_r_list=[]
    for i_episode in range(algo_params['num_epochs']):

        # dqn.load_model('saves/child#001_0/last_epoch_GRUQ_130.pt')

        start_time=time.time()
        simu_steps = 0
        s=env.reset()
        ep_r=0
        # print("当前为第",i_episode,"个episode")
        print(dqn.memory_counter)
        for i in tqdm(range(algo_params['num_steps_per_epoch'])):
            a=dqn.choose_action(s)
            # print(a)
            a=ACTION_SPACE[1]
            print(len(env.env.BG_hist))
            # if dqn.memory_counter < algo_params['replay_buffer_size']:
            #     # a=ACTION_SPACE[1]
            #     # print(ep_r)
            #     a=random.choice(ACTION_SPACE)
            # if dqn.memory_counter > algo_params['replay_buffer_size']:
            #     print(i)
            #     print(a)
            #     print(ep_r)
            # 做出行为并给出反应
            # print(a)
            s_,r,done,info=env.step(a)
            # print(r)
            # print(f'胰岛素量为{a},血糖值为{env.env.CGM_hist[-1]}')
            # print(s_[284:288])
            # print(s_[-4:])
            # print(r)
            # print(i)
            # 存储
            dqn.store_transition(s,a,r,s_)
            simu_steps += 1
            ep_r+=r
            # print(env.env.insulin_hist)
            # print(ep_r)
            if dqn.memory_counter>=algo_params['replay_buffer_size']:
                dqn.learn()
                if done or simu_steps>=algo_params['num_steps_per_epoch']:
                    simu_steps=0
                    s_ = env.reset()
                    model_last_epoch=dqn.eval_net
                    if i_episode%5==0:
                        torch.save(model_last_epoch, save_path + f'/last_epoch_{net_type}_{i_episode}.pt')
                    if model_max_reward is None:
                        model_max_reward=dqn.eval_net
                        torch.save(model_max_reward, save_path + f'/max_reward_{net_type}.pt')
                        best_ep_r=ep_r
                    elif ep_r>best_ep_r:
                        model_max_reward=dqn.eval_net
                        torch.save(model_max_reward, save_path + f'/max_reward_{net_type}.pt')
                        best_ep_r = ep_r
                    # print('Ep: ', i_episode,
                    #       '| Ep_r: ', round(ep_r, 2))
                    ep_r_list.append(ep_r)
                    # break0
            elif done or simu_steps>=algo_params['num_steps_per_epoch']:
                simu_steps=0
                s_=env.reset()
            s=s_
        print('|epoch:{:3d}/{:3d} | time={:5.2f}s | ep_r={:5.4f}'.format(i_episode,algo_params['num_epochs'],(time.time()-start_time),ep_r))


    # with open(save_path+f'/ep_r_list_{net_type}.pkl','wb') as f:
    #     pickle.dump(ep_r_list,f)

    # plt.plot(ep_r_list)
    # plt.show()

def run_eval(variant,model_path,name):
    algo_params = variant['algo_params']

    reward_fun = reward_name_to_function('risk_bg')
    patient_name=name
    env = bgp_env.DeepSACT1DEnv(reward_fun=reward_fun,
                                patient_name=patient_name,
                                seeds={'numpy': variant['base_seed'],
                                       'sensor': variant['base_seed'],
                                       'scenario': variant['base_seed']},
                                reset_lim=variant['reset_lim'], time=variant['include_time'],
                                meal=variant['include_meal'], bw_meals=variant['bw_meals'],
                                load=variant['load'], use_pid_load=variant['use_pid_load'],
                                hist_init=variant['hist_init'],
                                gt=variant['use_ground_truth'], n_hours=variant['n_hours'],
                                norm=variant['norm'], time_std=variant['time_std'],
                                use_old_patient_env=variant['use_old_patient_env'], action_cap=variant['action_cap'],
                                action_bias=variant['action_bias'], action_scale=variant['action_scale'],
                                basal_scaling=variant['basal_scaling'],
                                meal_announce=variant['meal_announce'], residual_basal=variant['residual_basal'],
                                residual_bolus=variant['residual_bolus'], residual_PID=variant['residual_PID'],
                                fake_gt=variant['fake_gt'], fake_real=variant['fake_real'],
                                suppress_carbs=variant['suppress_carbs'], limited_gt=variant['limited_gt'],
                                termination_penalty=variant['termination_penalty'], weekly=variant['weekly'],
                                update_seed_on_reset=variant['update_seed_on_reset'],
                                deterministic_meal_size=variant['deterministic_meal_size'],
                                deterministic_meal_time=variant['deterministic_meal_time'],
                                deterministic_meal_occurrence=variant['deterministic_meal_occurrence'],
                                harrison_benedict=variant['harrison_benedict_sched'],
                                restricted_carb=variant['restricted_sched'], meal_duration=variant['meal_duration'],
                                rolling_insulin_lim=variant['rolling_insulin_lim'], universal=variant['universal'],
                                reward_bias=variant['reward_bias'], carb_error_std=variant['carb_error_std'],
                                carb_miss_prob=variant['carb_miss_prob'], source_dir=variant['source_dir'])
    s = env.reset()
    BAS = env.ideal_basal

    ACTION_SPACE = [0, BAS, 5 * BAS]
    N_ACTIONS = len(ACTION_SPACE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_type = variant['net_type']
    dqn = DQN(env.observation_space.shape, N_ACTIONS, device, net_type, LR=1e-3,
              MEMORY_CAPACITY=algo_params['replay_buffer_size'],
              epsilon=0.95, discount_rate=0.99, BATCH_SIZE=256)
    dqn.load_model(model_path)      # 读取已经保存的模型


    patient = variant['patient_name']
    seed = variant['base_seed']
    save_path = f'saves/{patient}_{seed}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_last_epoch = None
    model_max_reward = None
    best_ep_r = 0
    n_days=10

    ep_r=0
    for i in tqdm(range(n_days* int(1440/env.sample_time))):
        a = dqn.choose_action(s)
        # print(a)
        a = ACTION_SPACE[a]
        s, r, d, info = env.step(a)
        ep_r += r
    hist = env.env.show_history()[288:]
    print(model_path)
    print('平均风险为',hist['Risk'].mean())
    print('平均风险为',ep_r/(n_days* int(1440/env.sample_time)))


