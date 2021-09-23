# -*- coding: utf-8 -*-
# import the necessary packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
import time
import pandas as pd
import joblib
import os
from collections import OrderedDict
import itertools
from joblib import Parallel, delayed
from bgp.rl import pid
from bgp.rl.reward_functions import risk_diff
import bgp.simglucose.envs.simglucose_gym_env as bgp_env
from bgp.rl import reward_functions
from bgp.rl.DQN import *

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

# 1. Define some Hyper Parameters
BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target netowrk updates
MEMORY_CAPACITY = 2000                  # The capacity of experience replay buffer


# 1. 糖尿病模拟器的参数
t_start = time.time()
base_name = 'tst'
save_path = '/save/path'  # where the outputs will be saved
full_path = '{}/{}'.format(save_path, base_name)
source_path = 'D:/MyWorks/糖尿病预测/强化学习方法/rl4bg'  # the path to the location of the folder 'bgp' which contains the source code+

# General utility parameters
debug = False
device_list = ['cuda:0']  # list of cuda device ids or None for cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # the cuda device to default to for debug runs, can also set to 'cpu'
seed_options = [i for i in range(3)]
validation_seed_offset = 1000000
test_seed_offset = 2000000
# the set of virtual patients to run for, valid options are [child/adolescent/adult]#[001/.../010]
person_options = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                  ['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                  ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
# Transfer
transfer_run = False  # Used to differentiate RL-Scratch from RL-Trans
transfer_init = 'fsp'  # The directory where the original trained models are saved

# Model Selection Strategy
finish = False  # Used to run more test seeds for a fully trained method
finish_mod = ''  # Appendix to rollout save file for a finish=True run
finish_itr = 'best'  # Whether to use the final model or the best model according to some model selection strategy
use_min = 30  # if False, select model with the highest average score

# To enable RL-MA
residual_bolus = False

# To enable Oracle
use_ground_truth = False

# Varying meal timing
time_std = None

# Varying amount of evaluation
num_eval_runs = 100

# Some important training parameters
num_steps_per_epoch = 5760              # 使用20天的数据作为训练集
num_steps_per_eval = 2880               # 使用10天的数据作为验证集
loss_function = nn.SmoothL1Loss
reward_fun = 'risk_diff'
snapshot_gap = 1
discount = 0.99
policy_lr = 3e-4
qf_lr = 3e-4
vf_lr = 3e-4
rnn_size = 128
rnn_layers = 2
ground_truth_network_size = 256

# Universal Action Space
# action_scale = 'basal'
action_scale = 1
basal_scaling = 43.2
action_bias = 0

# Augmented Reward Function
reward_bias = 0
termination_penalty = 1e5

# Realistic Variation in training data
update_seed_on_reset = True

# if not os.path.exists(full_path) and not finish:
#     os.mkdir(full_path)

if transfer_run:
    num_epochs = 50
else:
    num_epochs = 800

# Overwriting training parameters to make short runs for debugging purposes
if debug:
    num_steps_per_epoch = 5760
    num_steps_per_eval = 576
    num_epochs = 10
    num_eval_runs = 1



'''
--------------Procedures of DQN Algorithm------------------
'''
tuples = []
print(person_options)
# person_options=['adolescent#001']
# person_options=['adolescent#001','child#003','adult#001']
person_options=['child#003']
seed_options=[0]
option_dict = OrderedDict([('seed', seed_options),
                           ('person', person_options),
                           ])
net_type='DQN'      # DQN,CNNQ,GRUQ

for setting in itertools.product(*option_dict.values()):
    seed, person= setting
    reset_lim = {'lower_lim': 10, 'upper_lim': 1000}
    name_args = OrderedDict({})
    for i in range(len(setting)):
        name_args[list(option_dict.keys())[i]] = setting[i]

    if transfer_run:
        use_warm_start = True
        warm_start = transfer_init
    else:
        use_warm_start = False
        warm_start = None

    run_name = '{}'.format(base_name)
    for key in name_args:
        run_name += ';{}={}'.format(key, name_args[key])
    run_name += ';'  # allows easily splitting off .pkl
    save_name = '{}/{}/{}'.format(save_path, base_name, run_name)

    variant = dict(
        algo_params=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            batch_size=128,
            max_path_length=num_steps_per_epoch,
            discount=discount,
            reward_scale=1,
            soft_target_tau=.005,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            vf_lr=vf_lr,
            save_environment=True,
            device=device,
            # replay_buffer_size=int(5760),
            replay_buffer_size=int(3e4),
            weight_decay=0,
            gradient_max_value=None,
            save_replay_buffer=False,
            validation_seed_offset=validation_seed_offset,
        ),
        device=device,
        net_type=net_type,
        patient_name=person,
        base_seed=seed,
        run_name=run_name,
        source_dir=source_path,
        log_dir=save_name,
        reward_fun=reward_fun,
        sim_seed_mod=test_seed_offset,
        n_sim_days=10,
        model_type='sac',
        include_time=False,
        include_meal=False,
        use_ground_truth=use_ground_truth,
        net_size=ground_truth_network_size,
        layernorm=False,
        reset_lim=reset_lim,
        bw_meals=True,
        fancy=False,
        rnn=True,
        rnn_size=rnn_size,
        rnn_layers=rnn_layers,
        n_hours=24,
        norm=False,
        loss_function=loss_function,
        time_std=time_std,
        snapshot_gap=snapshot_gap,
        load=False,
        use_pid_load=False,
        hist_init=False,                # 在reset后，是否使用历史数据
        use_old_patient_env=False,
        action_cap=None,
        action_bias=action_bias,
        action_scale=action_scale,
        meal_announce=None,
        residual_basal=False,
        residual_bolus=residual_bolus,
        residual_PID=False,
        fake_gt=False,
        fake_real=False,
        suppress_carbs=False,
        limited_gt=False,
        termination_penalty=termination_penalty,
        dilation=False,
        warm_start=warm_start,
        weekly=False,
        update_seed_on_reset=update_seed_on_reset,
        num_eval_runs=num_eval_runs,
        deterministic_meal_time=False,
        deterministic_meal_size=False,
        deterministic_meal_occurrence=False,
        basal_scaling=basal_scaling,
        deterministic_init=False,
        harrison_benedict_sched=True,
        restricted_sched=False,
        meal_duration=5,
        independent_init=None,
        rolling_insulin_lim=None,
        universal=False,
        finish_mod=finish_mod,
        unrealistic=False,
        reward_bias=reward_bias,
        finish_itr=finish_itr,
        use_min=use_min,
        carb_error_std=0,
        carb_miss_prob=0,
    )
    run_train(variant=variant)
    # for i in range(250,326,5):
    #     run_eval(variant=variant,model_path=f'saves/child#001_0/last_epoch_GRUQ_{i}.pt',name='child#001')
    break
