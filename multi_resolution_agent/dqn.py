import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random, math

from collections import namedtuple
import numpy as np 

MODEL_STORE_PATH = os.getcwd()
print(MODEL_STORE_PATH)
modelname = 'DQN_Pong'
model_path_play = MODEL_STORE_PATH + '/' +'play_model/'+ 'DQN_Pong_guide.pt'

Transition_track = namedtuple('Transion', ('state_f','state_p', 'action_before', 'action', 'next_state_f','next_state_p', 'reward'))

GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
TARGET_UPDATE = 1000

class DQN_play_guide(nn.Module):
    def __init__(self, in_channels=400, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN_play_guide, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(6 * 6 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

  

class DQN_agent_guide():
    def __init__(self, device,in_channels=1, action_space=[]):
        
        self.device=torch.device(device)
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n    
        
        self.stepdone = 0
        self.DQN = DQN_play_guide(self.in_channels, self.action_dim).to(self.device)
        self.target_DQN = DQN_play_guide(self.in_channels, self.action_dim).to(self.device)
        # 加载之前训练好的模型
        self.DQN.load_state_dict(torch.load(model_path_play,map_location=device))
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        
    def select_action(self, state):
        
        state = state.to(self.device)
        action = self.DQN(state).detach().max(1)[1].view(1,1)
            
        return action          
        
        
    

class ReplayMemory_track(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition_track(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN_track(nn.Module):
    def __init__(self, in_channels, fovea_patch_num, periphery_patch_num, device, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            fovea_patch_num (int): number of fovea patch
            device (torch): device to run
            n_actions (int): number of outputs
        """
        super(DQN_track, self).__init__()
        self.conv11 = nn.Conv2d(in_channels[0],16, kernel_size=2, stride=1)
        
        self.conv21 = nn.Conv2d(in_channels[1],16, kernel_size=2, stride=1)
        
        num_f = fovea_patch_num-1
        num_p = periphery_patch_num-1
        
        self.head1 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, n_actions)
        self.head2 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, n_actions)
        self.device = device
        
    def forward(self, x1, x2, x3):#
        x1 = x1.float()
        x1 = F.relu(self.conv11(x1))
        x1=x1.view(x1.size(0), -1)
        
        x2 = x2.float()
        x2 = F.relu(self.conv21(x2))
        x2=x2.view(x2.size(0), -1)
        
        x = torch.cat([x1,x2],dim=1)
        
        x3=x3.float()/4
        
        x4=torch.zeros((x3.size(0),1)).to(self.device)
        x5=torch.zeros((x3.size(0),1)).to(self.device)
        
        x4[:,0]=x3[:,0]
        x5[:,0]=x3[:,1]
        
        
        x4 = torch.cat([x,x4],dim=1).to(self.device)
        x5 = torch.cat([x,x5],dim=1).to(self.device)
        
        
        head1 = self.head1(x4)
        head2 = self.head2(x5)
        
        return head1,head2



class DQN_track_video(nn.Module):
    def __init__(self, in_channels, fovea_patch_num, periphery_patch_num, device, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            fovea_patch_num (int): number of fovea patch
            device (torch): device to run
            n_actions (int): number of outputs
        """
        super(DQN_track_video, self).__init__()
        self.conv11 = nn.Conv2d(in_channels[0],16, kernel_size=2, stride=1)
        
        self.conv21 = nn.Conv2d(in_channels[1],16, kernel_size=2, stride=1)
        
        num_f = fovea_patch_num-1
        num_p = periphery_patch_num-1
        
        self.head1 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, n_actions)
        self.head2 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, n_actions)
        self.device = device
        
    def forward(self, x1, x2, x3):#
        x1 = x1.float()
        x1 = F.relu(self.conv11(x1))
        x1=x1.view(x1.size(0), -1)
        
        data_conv1 = x1.detach()
        
        x2 = x2.float()
        x2 = F.relu(self.conv21(x2))
        x2=x2.view(x2.size(0), -1)
        
        data_conv2 = x2.detach()
        
        x = torch.cat([x1,x2],dim=1)
        
        x3=x3.float()/4
        
        x4=torch.zeros((x3.size(0),1)).to(self.device)
        x5=torch.zeros((x3.size(0),1)).to(self.device)
        
        x4[:,0]=x3[:,0]
        x5[:,0]=x3[:,1]
        
        
        x4 = torch.cat([x,x4],dim=1).to(self.device)
        x5 = torch.cat([x,x5],dim=1).to(self.device)
        
        
        head1 = self.head1(x4)
        
        head2 = self.head2(x5)
        
        return head1,head2,data_conv1,data_conv2


class DQN_agent_track():
    def __init__(self, fovea_patch_num, periphery_patch_num, device,in_channels=1, learning_rate=0.2*1e-3, memory_size=100000, epsilon=0.99):
        
        self.device = torch.device(device)
        self.in_channels = in_channels
        self.action_dim = 5 # fovea action(stop up down left right)
        self.fovea_pos = np.array([9,9])  #position of fovea
        self.memory_buffer = ReplayMemory_track(memory_size)
        self.stepdone =  0
        self.DQN = DQN_track(self.in_channels, fovea_patch_num, periphery_patch_num, self.device, self.action_dim).to(self.device)
        self.target_DQN = DQN_track(self.in_channels, fovea_patch_num, periphery_patch_num, self.device, self.action_dim).to(self.device)
        
        self.batch_size = 32
        self.esp_decay = 200000
        
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=learning_rate, eps=0.001, alpha=0.95)
        
        
        
    def select_action(self, state_f, state_p, action_before):
       
        self.stepdone += 1
        
        
        epsilon = EPS_END + (EPS_START - EPS_END)* \
            math.exp(-1. * self.stepdone / self.esp_decay) 
        
        if random.random()<epsilon:#
             action_fovea1 = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
             action_fovea2 = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
            
        else:
             state_f = state_f.to(self.device)
             state_p = state_p.to(self.device)
             action_before = action_before.to(self.device)
             action_fovea1,action_fovea2 = self.DQN(state_f,state_p,action_before)
             action_fovea1 = action_fovea1.detach().max(1)[1].view(1,1)
             action_fovea2 = action_fovea2.detach().max(1)[1].view(1,1)
         
        return action_fovea1,action_fovea2    
        
        
    def learn(self):
        
        if self.memory_buffer.__len__()<self.batch_size:
            return
        
        transitions = self.memory_buffer.sample(self.batch_size)
        
        batch = Transition_track(*zip(*transitions))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward))) 
    
    
        
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state_f)),
            device=self.device, dtype=torch.uint8).bool()
        
        non_final_next_states_f = torch.cat([s for s in batch.next_state_f
                                           if s is not None]).to(self.device)
        non_final_next_states_p = torch.cat([s for s in batch.next_state_p
                                           if s is not None]).to(self.device)
        non_final_action = torch.cat([a for a in batch.action
                                           if a is not None]).to(self.device)
        
        if non_final_next_states_f.size(0) != 32:
            return
        
        state_f_batch = torch.cat(batch.state_f).to(self.device)
        state_p_batch = torch.cat(batch.state_p).to(self.device)
        action_before_batch = torch.cat(batch.action_before).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(rewards)
        
        action1=torch.unsqueeze(action_batch[:,0], dim=1).to(self.device)
        action2=torch.unsqueeze(action_batch[:,1], dim=1).to(self.device)
        
        state_action_values1, state_action_values2 = self.DQN(state_f_batch,state_p_batch,action_before_batch)#
        
        state_action_values1 = state_action_values1.gather(1, action1)
        state_action_values2 = state_action_values2.gather(1, action2)
        
        
        next_state_values1 = torch.zeros([self.batch_size,self.action_dim], device=self.device)
        next_state_values2 = torch.zeros([self.batch_size,self.action_dim], device=self.device)
        
        next_state_values1[non_final_mask], next_state_values2[non_final_mask] = self.target_DQN(non_final_next_states_f,non_final_next_states_p,non_final_action)
        next_state_values1 = next_state_values1.max(1)[0].detach()
        next_state_values2 = next_state_values2.max(1)[0].detach()
        
        expected_state_action_values1 = (next_state_values1 * GAMMA) + reward_batch
        expected_state_action_values2 = (next_state_values2 * GAMMA) + reward_batch
        
        loss1 = F.smooth_l1_loss(state_action_values1, expected_state_action_values1.unsqueeze(1))
        loss2 = F.smooth_l1_loss(state_action_values2, expected_state_action_values2.unsqueeze(1))
        
        loss = loss1+loss2
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()


class DQN_agent_track_evaluate():
    def __init__(self, fovea_patch_num, periphery_patch_num, device, model_path,in_channels=1, learning_rate=0.2*1e-3, memory_size=100000, epsilon=0.99):
        
        self.device = torch.device(device)
        self.in_channels = in_channels    
        self.action_dim = 5 # fovea action(stop up down left right)
        self.fovea_pos = np.array([9,9])  #position of fovea
        self.stepdone =  0
        self.DQN = DQN_track(self.in_channels, fovea_patch_num, periphery_patch_num, self.device, self.action_dim).to(self.device)
        self.target_DQN = DQN_track(self.in_channels, fovea_patch_num, periphery_patch_num, self.device, self.action_dim).to(self.device)
        
        
        self.DQN.load_state_dict(torch.load(model_path,map_location=device))
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        
        
    def select_action(self, state_f, state_p, action_before):
       
        self.stepdone += 1 
       
        state_f = state_f.to(self.device)
        state_p = state_p.to(self.device)
        action_before = action_before.to(self.device)
        
        action_fovea1,action_fovea2 = self.DQN(state_f,state_p,action_before)
        action_fovea1 = action_fovea1.detach().max(1)[1].view(1,1)
        action_fovea2 = action_fovea2.detach().max(1)[1].view(1,1)
         
        return action_fovea1,action_fovea2    
        


class DQN_agent_track_video():
    def __init__(self, fovea_patch_num, periphery_patch_num, device, model_path,in_channels=1, learning_rate=0.2*1e-3, memory_size=100000, epsilon=0.99):
        
        self.device = torch.device(device)
        self.in_channels = in_channels    
        self.action_dim = 5 # fovea action(stop up down left right)
        self.fovea_pos = np.array([9,9])  #position of fovea
        self.stepdone =  0
        self.DQN = DQN_track_video(self.in_channels, fovea_patch_num, periphery_patch_num, self.device, self.action_dim).to(self.device)
        self.target_DQN = DQN_track_video(self.in_channels, fovea_patch_num, periphery_patch_num, self.device, self.action_dim).to(self.device)
        
        
        self.DQN.load_state_dict(torch.load(model_path,map_location=device))
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        
        
    def select_action(self, state_f, state_p, action_before, self_data_conv1, self_data_conv2):
       
        self.stepdone += 1 
       
        state_f = state_f.to(self.device)
        state_p = state_p.to(self.device)
        action_before = action_before.to(self.device)
        
        action_fovea1,action_fovea2,data_conv1,data_conv2 = self.DQN(state_f,state_p,action_before)
        
        self_data_conv1 = torch.cat([self_data_conv1,data_conv1.cpu()])
        self_data_conv2 = torch.cat([self_data_conv2,data_conv2.cpu()])
        
        action_fovea1 = action_fovea1.detach().max(1)[1].view(1,1)
        action_fovea2 = action_fovea2.detach().max(1)[1].view(1,1)
         
        return action_fovea1,action_fovea2, self_data_conv1, self_data_conv2
        







Transition_play = namedtuple('Transion', ('state_f','state_p','fovea_pos', 'action_fovea_before', 'action_fovea','action_play', 'next_state_f','next_state_p','next_fovea_pos', 'reward'))


class ReplayMemory_play(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition_play(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)



class DQN_play(nn.Module):
    def __init__(self, in_channels, fovea_patch_num, periphery_patch_num, device, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            fovea_patch_num (int): number of fovea patch
            n_actions (int): number of outputs
        """
        super(DQN_play, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[0],16, kernel_size=2, stride=1)
        
        self.conv2 = nn.Conv2d(in_channels[1],16, kernel_size=2, stride=1)
        
        num_f = fovea_patch_num-1
        num_p = periphery_patch_num-1
        
        self.head1 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, 5)
        self.head2 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, 5)
        
        for p in self.parameters():
            p.requires_grad = False
        
        self.fc = nn.Linear((num_p * num_p + num_f * num_f) * 16 + 4, 512)
        self.fc2 = nn.Linear(512,128)
        self.head = nn.Linear(128, n_actions)
        
        self.device = device
        
    def forward(self, x1, x2, x_move, x_pos):
        x1 = x1.float()
        x2 = x2.float()
        
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        x_move = x_move.to(self.device)
        x_pos = x_pos.to(self.device)
        
        
        x1 = F.relu(self.conv1(x1))
        x1=x1.view(x1.size(0), -1)
        
        
        
        x2 = F.relu(self.conv2(x2))
        x2=x2.view(x2.size(0), -1)
        
        x = torch.cat([x1,x2],dim=1)
        
        x_move=x_move.float()/4
        
        x_move_x=torch.zeros((x_move.size(0),1)).to(self.device)
        x_move_y=torch.zeros((x_move.size(0),1)).to(self.device)
        
        x_move_x[:,0]=x_move[:,0]
        x_move_y[:,0]=x_move[:,1]
        
        x_x = torch.cat([x,x_move_x],dim=1).to(self.device)
        x_y = torch.cat([x,x_move_y],dim=1).to(self.device)
        
        head1 = self.head1(x_x)
        head2 = self.head2(x_y)
        
        x_pos = x_pos.float()/18
        
        x=torch.cat([x,x_move,x_pos],dim=1)
        
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x.view(x.size(0), -1)))
        
        head = self.head(x)
        
        return head1,head2,head


class DQN_play_video(nn.Module):
    def __init__(self, in_channels, fovea_patch_num, periphery_patch_num, device, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            fovea_patch_num (int): number of fovea patch
            n_actions (int): number of outputs
        """
        super(DQN_play_video, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[0],16, kernel_size=2, stride=1)
        
        self.conv2 = nn.Conv2d(in_channels[1],16, kernel_size=2, stride=1)
        
        num_f = fovea_patch_num-1
        num_p = periphery_patch_num-1
        
        self.head1 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, 5)
        self.head2 = nn.Linear((num_p*num_p + num_f*num_f) * 16+1, 5)
        
        self.fc = nn.Linear((num_p * num_p + num_f * num_f) * 16 + 4, 512)
        self.fc2 = nn.Linear(512,128)
        self.head = nn.Linear(128, n_actions)
        
        self.device = device
        
    def forward(self, x1, x2, x_move, x_pos):
        x1 = x1.float()
        x1 = F.relu(self.conv1(x1))
        x1=x1.view(x1.size(0), -1)
        
        data_conv1 = x1.detach()
        
        x2 = x2.float()
        x2 = F.relu(self.conv2(x2))
        x2=x2.view(x2.size(0), -1)
        
        data_conv2 = x2.detach()
        
        x = torch.cat([x1,x2],dim=1)
        
        power_conv = float(x.detach().sum())
        
        x_move=x_move.float()/4
        
        x_move_x=torch.zeros((x_move.size(0),1)).to(self.device)
        x_move_y=torch.zeros((x_move.size(0),1)).to(self.device)
        
        x_move_x[:,0]=x_move[:,0]
        x_move_y[:,0]=x_move[:,1]
        
        x_x = torch.cat([x,x_move_x],dim=1).to(self.device)
        x_y = torch.cat([x,x_move_y],dim=1).to(self.device)
        
        head1 = self.head1(x_x)
        head2 = self.head2(x_y)
        
        x_pos = x_pos.float()/18
        
        x=torch.cat([x,x_move,x_pos],dim=1)
        
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        data_all1 = x.detach()
        x = F.relu(self.fc2(x.view(x.size(0), -1)))
        
        data_all2 = x.detach()
        power_net = float(x.detach().sum())
          
        head = self.head(x)
        
        
        return head1, head2, head, power_conv, power_net, data_conv1, data_conv2, data_all1, data_all2


class DQN_agent_play():
    def __init__(self,in_channels, fovea_patch_num, periphery_patch_num, device, action_space=[], learning_rate=2*1e-4, memory_size=100000, epsilon=0.99):
        
        self.device=torch.device(device)
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n 
        self.memory_buffer = ReplayMemory_play(memory_size)        
        self.stepdone = 0
        self.DQN = DQN_play(self.in_channels,fovea_patch_num, periphery_patch_num,self.device,self.action_dim).to(self.device)
        self.target_DQN = DQN_play(self.in_channels, fovea_patch_num, periphery_patch_num,self.device,self.action_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=learning_rate, eps=1e-3, alpha=0.95)
        
        
        self.EPS_DECAY = 1000000
        self.BATCH_SIZE = 128
        
        self.fix_list = ['conv1.weight','conv1.bias','conv2.weight','conv2.bias','head1.weight','head1.bias','head2.weight','head2.bias']
        
        
    def select_action(self, state_f, state_p, action_fovea_before, fovea_pos):
        
        self.stepdone += 1
        
        epsilon = EPS_END + (EPS_START - EPS_END)* \
            math.exp(-1. * self.stepdone / self.EPS_DECAY) 
        
        # print(epsilon)
        if random.random()<epsilon:
            action_fovea1,action_fovea2,action_play = self.DQN(state_f,state_p,action_fovea_before,fovea_pos)
            action_fovea1 = action_fovea1.detach().max(1)[1].view(1,1)
            action_fovea2 = action_fovea2.detach().max(1)[1].view(1,1)
        
            action_play = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
            
        else:
            state_f = state_f.to(self.device)
            state_p = state_p.to(self.device)
            action_fovea_before = action_fovea_before.to(self.device)
            fovea_pos = fovea_pos.to(self.device)
            
            action_fovea1,action_fovea2,action_play = self.DQN(state_f,state_p,action_fovea_before,fovea_pos)
            action_fovea1 = action_fovea1.detach().max(1)[1].view(1,1)
            action_fovea2 = action_fovea2.detach().max(1)[1].view(1,1)
            action_play = action_play.detach().max(1)[1].view(1,1)
        
        return action_fovea1,action_fovea2,action_play         
        
    
    
    def learn(self):
        if self.memory_buffer.__len__()<self.BATCH_SIZE:
            return
        
        transitions = self.memory_buffer.sample(self.BATCH_SIZE)
        
        batch = Transition_play(*zip(*transitions))
        # print(batch)
        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action_play))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward))) 
    
    
        
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state_f)),
            device=self.device, dtype=torch.uint8).bool()
        
        non_final_next_states_f = torch.cat([s for s in batch.next_state_f
                                           if s is not None]).to(self.device)
        non_final_next_states_p = torch.cat([s for s in batch.next_state_p
                                           if s is not None]).to(self.device)
        non_final_action_fovea = torch.cat([a for a in batch.action_fovea
                                           if a is not None]).to(self.device)
        non_final_next_fovea_pos = torch.cat([a for a in batch.next_fovea_pos
                                           if a is not None]).to(self.device)
        
        if non_final_next_states_f.size(0) != self.BATCH_SIZE:
            return
        
        # print(type(batch.state))
        state_f_batch = torch.cat(batch.state_f).to(self.device)
        state_p_batch = torch.cat(batch.state_p).to(self.device)
        fovea_pos_batch = torch.cat(batch.fovea_pos).to(self.device)
        
        action_fovea_before_batch = torch.cat(batch.action_fovea_before).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        
        state_action_values_fovea1, state_action_values_fovea2, state_action_values = self.DQN(state_f_batch,state_p_batch,action_fovea_before_batch,fovea_pos_batch)
        state_action_values = state_action_values.gather(1, action_batch)#
        
        next_state_values_fovea1 = torch.zeros((self.BATCH_SIZE,5), device=self.device)
        next_state_values_fovea2 = torch.zeros((self.BATCH_SIZE,5), device=self.device)
        next_state_values1 = torch.zeros((self.BATCH_SIZE,6), device=self.device)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        
        next_state_values_fovea1[non_final_mask],next_state_values_fovea2[non_final_mask],next_state_values1[non_final_mask] = self.target_DQN(non_final_next_states_f,non_final_next_states_p,non_final_action_fovea,non_final_next_fovea_pos)
        next_state_values = next_state_values1.max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) #smooth_l1_loss
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        for name, param in self.DQN.named_parameters():
            if name in self.fix_list:
                continue
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()




class DQN_agent_play_evaluate():
    def __init__(self,in_channels, fovea_patch_num, periphery_patch_num, device, model_path_play,action_space=[], learning_rate=2*1e-4, memory_size=100000, epsilon=0.99):
        
        self.device=torch.device(device)
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n 
        self.memory_buffer = ReplayMemory_play(memory_size)        
        self.stepdone = 0
        self.DQN = DQN_play(self.in_channels,fovea_patch_num, periphery_patch_num,self.device,self.action_dim).to(self.device)
        self.target_DQN = DQN_play(self.in_channels, fovea_patch_num, periphery_patch_num,self.action_dim).to(self.device)
        
        self.DQN.load_state_dict(torch.load(model_path_play,map_location=device))
        # self.target_DQN.load_state_dict(self.DQN.state_dict())
        
        
        
        
    def select_action(self, state_f, state_p, action_fovea_before, fovea_pos):
        
        self.stepdone += 1
        
        state_f = state_f.to(self.device)
        state_p = state_p.to(self.device)
        action_fovea_before = action_fovea_before.to(self.device)
        fovea_pos = fovea_pos.to(self.device)
            
        action_fovea1,action_fovea2,action_play = self.DQN(state_f,state_p,action_fovea_before,fovea_pos)
        action_fovea1 = action_fovea1.detach().max(1)[1].view(1,1)
        action_fovea2 = action_fovea2.detach().max(1)[1].view(1,1)
        q = action_play.detach().max().cpu()
        action_play = action_play.detach().max(1)[1].view(1,1)
        
        return action_fovea1,action_fovea2,action_play,q         
        
    
class DQN_agent_play_video():
    def __init__(self,in_channels, fovea_patch_num, periphery_patch_num, device, model_path_play,action_space=[], learning_rate=2*1e-4, memory_size=100000, epsilon=0.99):
        
        self.device=torch.device(device)
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n 
        self.memory_buffer = ReplayMemory_play(memory_size)        
        self.stepdone = 0
        self.DQN = DQN_play_video(self.in_channels,fovea_patch_num, periphery_patch_num,self.device,self.action_dim).to(self.device)
        # self.target_DQN = DQN_play_video(self.in_channels, fovea_patch_num, periphery_patch_num,self.action_dim).to(self.device)
        
        self.DQN.load_state_dict(torch.load(model_path_play,map_location=device))
        # self.target_DQN.load_state_dict(self.DQN.state_dict())
        
        
        
        
    def select_action(self, state_f, state_p, action_fovea_before, fovea_pos, self_power_conv,
                      self_power_net, self_data_conv1, self_data_conv2, self_data_all1,self_data_all2):
        
        self.stepdone += 1
        
        state_f = state_f.to(self.device)
        state_p = state_p.to(self.device)
        action_fovea_before = action_fovea_before.to(self.device)
        fovea_pos = fovea_pos.to(self.device)
            
        action_fovea1,action_fovea2, action_play, power_conv, power_net, data_conv1, data_conv2, data_all1, data_all2 = self.DQN(state_f,state_p,action_fovea_before,fovea_pos)
        
        self_power_conv += power_conv
        self_power_net  += power_net
        
        self_data_conv1 = torch.cat([self_data_conv1,data_conv1.cpu()])
        self_data_conv2 = torch.cat([self_data_conv2,data_conv2.cpu()])
        self_data_all1 = torch.cat([self_data_all1,data_all1.cpu()])
        self_data_all2 = torch.cat([self_data_all2,data_all2.cpu()])
        
        action_fovea1 = action_fovea1.detach().max(1)[1].view(1,1)
        action_fovea2 = action_fovea2.detach().max(1)[1].view(1,1)
        action_play = action_play.detach().max(1)[1].view(1,1)
        
        return action_fovea1,action_fovea2,action_play,self_power_conv,self_power_net,self_data_conv1,self_data_conv2,self_data_all1,self_data_all2       