import train_track_dqn
import train_play_dqn
import numpy as np
import torch

torch.set_num_threads(6)

"""
Train LCA fovea:
    1. Create training set during training
    2. Train half of LCA neurons on paddle data
    3. Fix neurons trained in step 2 and train the other half of neurons on all data


Parameters

periphery_size: size of the periphery image
fovea_size: size of the fovea image
patch_size: size of each patch
frame_analyze: number of concetive frames
trainingset_steps: number of steps we play Pong with random actions, influence the size of training set
x_paddle: the column that paddle locates. Check it before creating training set!!!

overcomplete: ratio of overcompleteness
epoch_lca: number of epoch we train LCA
threshold: threshold of LCA neurons' activation
"""
device = 'cuda:0'

periphery_size = 16
fovea_size = 40
fovea_patch_num = 10

patch_size = np.zeros(2,dtype=int)
patch_size[0] = 2 #fovea
patch_size[1] = 2 #periphery

frame_analyze = 2

overcomplete = 4
epoch_lca = 301
threshold = 1.3 
which_epoch = 1920

"""
Train track DQN
"""

#train LCA of fovea

print('\n=================================')
print('\nBegin to train track DQN\n')
print('=================================\n')


train_track_dqn.train_lca_track(periphery_size, fovea_size, fovea_patch_num, 
                                frame_analyze, patch_size, overcomplete,epoch_lca,
                                threshold, device)




# test reconstruction performance (fovea)

print('\n=================================')
print('\nBegin to evaluate track DQN\n')
print('=================================\n')

train_track_dqn.evaluate_lca_track(periphery_size, fovea_size, fovea_patch_num, 
                                frame_analyze, patch_size, overcomplete,epoch_lca,
                                threshold, device)





"""
Train play DQN
"""

#train DQN
print('\n=================================')
print('\nBegin to train play DQN\n')
print('=================================\n')

train_play_dqn.train_dqn_play(periphery_size, fovea_size, fovea_patch_num, 
                                frame_analyze, patch_size, overcomplete,
                                threshold, device)




# # #evaluate

print('\n=================================')
print('\nBegin to evaluate play DQN\n')
print('=================================\n')

train_play_dqn.evaluate_dqn_play(periphery_size, fovea_size, fovea_patch_num, 
                                frame_analyze, patch_size, overcomplete,
                                threshold, device)



print('\n=================================')
print('\nBegin to evaluate the best play DQN for 1000 games\n')
print('=================================\n')

train_play_dqn.evaluate_dqn_play_best(periphery_size, fovea_size, fovea_patch_num, 
                                frame_analyze, patch_size, overcomplete,
                                threshold, device)



