import train_lca
import train_lca_dqn
import torch
import ale_py 
from ale_py import ALEInterface
ale = ALEInterface()
import gymnasium as gym
import h5py
import numpy as np

torch.set_num_threads(4)

"""
Train LCA:
    1. Create training set
    2. Train half of LCA neurons on paddle data
    3. Fix neurons trained in step 2 and train the other half of neurons on all data


Parameters

target_size: size of the output image
patch_size: size of each patch
frame_analyze: number of concetive frames
trainingset_steps: number of steps we play Pong with random actions, influence the size of training set
x_paddle: the column that paddle locates. Check it before creating training set!!!
          target_size=8, 16, 24, 32, 40
          x_paddle=[0], [0], [0,1], [1], [1]

overcomplete: ratio of overcompleteness
epoch_lca: number of epoch we train LCA
threshold: threshold of LCA neurons' activation
"""
target_size = 40
patch_size = 2
frame_analyze = 2
trainingset_steps = 50000
x_paddle = [1]
device='cuda:0'


overcomplete = 4
epoch_lca = 200
threshold = 1.3




#Create training set and train LCA

#train_lca.create_training_set(trainingset_steps, target_size, frame_analyze, patch_size, x_paddle)
#train_lca.train_LCA(threshold, overcomplete, epoch_lca,device)

#train_lca.check_paddle_location(steps=500, outputsize=target_size)

#test reconstruction performance (reconstruct video)

#train_lca.test_LCA(patch_size,frame_analyze,target_size,threshold, device)

# print('\n=================================')
# print('\nVideo reconstructed by LCA is in the folder\n')
# print('=================================\n')



"""
Train DQN
"""

#train DQN
print('\n=================================')
print('\nBegin to train LCA-DQN\n')
print('=================================\n')

#train_lca_dqn.train_LCA_DQN(threshold,overcomplete,patch_size,target_size,frame_analyze,device)
print('\n=================================')
print('\nFinish training\n')
print('=================================\n')



# #evaluate

print('\n=================================')
print('\nBegin to evaluate LCA-DQN\n')
print('=================================\n')

#train_lca_dqn.evaluate_LCA_DQN(threshold,overcomplete,patch_size,target_size,frame_analyze,device, selected_episodes=range(1920, 2001, 40))
print('\n=================================')
print('\nFinish evaluation\n')
print('=================================\n')

print('\n=================================')
print('\nBegin to evaluate the best LCA-DQN\n')
print('=================================\n')


#train_lca_dqn.evaluate_LCA_DQN_best(threshold,overcomplete,patch_size,target_size,frame_analyze,device)
print('\n=================================')
print('\nFinish evaluation\n')
print('=================================\n')

print('\n=================================')
print('\nRun game with best model from .h5 file\n')
print('=================================\n')

train_lca_dqn.run_game_with_h5("idk/reward_best_40_2.h5", threshold, overcomplete, patch_size, target_size, frame_analyze, device)

print('\n=================================')
print('\nFinished running game with best model\n')
print('=================================\n')

