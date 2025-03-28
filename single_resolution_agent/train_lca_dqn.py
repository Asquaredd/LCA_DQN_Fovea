import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import tlca
import gym, random, os.path, math
import h5py
from itertools import count
import numpy as np 
import time
import matplotlib.pyplot as plt
 
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

"""
Parameter

BATCH_SIZE: mini-batch size for training
GAMMA: gamma for q-learning
TARGET_UPDATE: period of update weights of target network
lr: learning rate
INITIAL_MEMORY: replay start size
MEMORY_SIZE: replay memory size

EPS_*: parameters for epsilon-greedy exploration
EPS_START: initial exploration
EPS_END: final exploration
EPS_DECAY: frequence of exploration decay

n_episode: number of training episode
e_episode: number of evaluation episode for every weights

"""

BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 1000
lr = 0.2*1e-3
INITIAL_MEMORY = 10000
MEMORY_SIZE = 50 * INITIAL_MEMORY

EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000


RENDER = False
n_episode = 3001
e_episode = 20
e_episode_final = 1000


MODEL_STORE_PATH = os.getcwd()
modelname = 'DQN_Pong'
madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode0.pt'


class ReplayMemory(object): #
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self, num_image, in_channels=400, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        
        n_neuron = int(num_image/2-1)
        
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,32, kernel_size=4, stride=2)
        self.fc4 = nn.Linear(n_neuron * n_neuron * 32, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DQN_power(nn.Module):
    def __init__(self, num_image, in_channels=400, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        
        n_neuron = int(num_image/2-1)
        
        super(DQN_power, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,32, kernel_size=4, stride=2)
        self.fc4 = nn.Linear(n_neuron * n_neuron * 32, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float()
        
        x = F.relu(self.conv1(x))
        
        x=x.view(x.size(0), -1)
        
        data_conv = x.detach()
        
        x = F.relu(self.fc4(x))
        
        data_all = x.detach()
        
        return self.head(x),data_conv,data_all


class DQN_agent():
    def __init__(self,num_image,devicename,in_channels=1, action_space=[], learning_rate=0.2*1e-3, memory_size=100000, epsilon=0.99):
        
        self.device=torch.device(devicename)
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n     
                
        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone =  0
        self.DQN = DQN(num_image,self.in_channels, self.action_dim).to(self.device)
        self.target_DQN = DQN(num_image,self.in_channels, self.action_dim).to(self.device)
        
        
        # self.DQN.load_state_dict(torch.load(madel_path))
        # self.target_DQN.load_state_dict(self.DQN.state_dict())
        
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=learning_rate, eps=0.001, alpha=0.95)
        
        
        
    def select_action(self, state):
        
        self.stepdone += 1
        state = state.to(self.device)
        epsilon = EPS_END + (EPS_START - EPS_END)* \
            math.exp(-1. * self.stepdone / EPS_DECAY) 
        
        # print(epsilon)
        if random.random()<epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
        else:
            action = self.DQN(state).detach().max(1)[1].view(1,1)
            
        return action        
        
        
    def learn(self):
        
        if self.memory_buffer.__len__()<BATCH_SIZE:
            return
        
        transitions = self.memory_buffer.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward))) 
    
    
        
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.uint8).bool()
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.device)
        
        # print(type(batch.state))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        state_action_values = self.DQN(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
    
           
class Trainer():

    def __init__(self, env, agent, n_episode, state_channel,
                 patch_size, frame_analyze, lca, num_image, dict_num,device):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        # self.losslist = []
        self.rewardlist = []
        self.sc=state_channel
        self.p_s=patch_size
        self.fa=frame_analyze
        self.frame_state=np.zeros((state_channel,patch_size*patch_size*frame_analyze))
        self.lca=lca
        self.ni=num_image
        self.dn=dict_num
        self.device=device
        
        self.target_size = num_image*patch_size
        
        self.start=time.time()
        self.end=time.time()
        
        
    def get_state(self,obs):
        
        t_u=10
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,self.target_size)
        frame=frame[np.newaxis,:,:]
        
        self.frame_state=tlca.video2image_piece(frame,self.p_s, 1,self.frame_state)
        state = torch.from_numpy(self.frame_state).to(self.device).float()
        a_state=tlca.run_lca(state.t(), self.lca, t_u, self.fa, self.p_s,self.device)
        a_state=torch.reshape(a_state,(self.dn,self.ni,self.ni))
        a_state=torch.unsqueeze(a_state,0).cpu()
        
        return a_state    
        
    def train(self):

        for episode in range(0,self.n_episode):
            
            obs = self.env.reset()
            state = self.get_state(obs)
            episode_reward = 0.0
            
            for t in count():  
                           
                action = self.agent.select_action(state)
                if RENDER:
                    self.env.render()
                
                
                obs,reward,done,info = self.env.step(action)
                episode_reward += reward
                
                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None
                    self.frame_state=np.zeros((self.sc,self.p_s*self.p_s*self.fa))
                    self.rewardlist.append(episode_reward)
                    
                reward = torch.tensor([reward], device=self.device)
                
               
                '''
                state: batch_size channel h w    size: batch_size * frame_analyze
                action: size: batch_size * 1
                next_state: batch_size channel h w    size: batch_size * frame_analyze
                reward: size: batch_size * 1                
                '''
                
                self.agent.memory_buffer.push(state, action.to('cpu'), next_state, reward.to('cpu')) # 里面的数据都是Tensor
                state = next_state
                
                if self.agent.stepdone > INITIAL_MEMORY:
                    self.agent.learn()
                    if self.agent.stepdone % TARGET_UPDATE == 0:
                        self.agent.target_DQN.load_state_dict(self.agent.DQN.state_dict())
                
                if done:
                    break
            
            if episode % 20 == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print (localtime)
                self.end=time.time()
                print('total time',self.end-self.start)
                torch.save(self.agent.DQN.state_dict(), MODEL_STORE_PATH + '/' + "model/{}_episode{}.pt".format(modelname, episode))
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.agent.stepdone, episode, t, episode_reward))
                self.start=time.time()
                hdffile=h5py.File('rewardlist.h5','w')
                hdffile.create_dataset('data',data=self.rewardlist)
                hdffile.close()
            
                
            
            self.env.close()
        return
    
    def smooth(self,rewardlist,step):
        y=np.zeros(step)
        num=50

        for i in range(step):
            if i==0:
                y[i]=rewardlist[i]
            if 0<i<num:
                y[i]=np.mean(rewardlist[:i])
            if i>=num:
                y[i]=np.mean(rewardlist[i-num:i])
        return y
    
    
    def plot_rewardlist(self):
        
        dictfile=h5py.File('rewardlist.h5','r')
        rewardlist1=np.array([dictfile['data']])[0,:]
        dictfile.close() 

        step1 = rewardlist1.shape[0]
        x1=np.linspace(0, step1-1, step1)
        y1=self.smooth(rewardlist1,step1)


        plt.figure(figsize=(10,5))
        plt.plot(x1,y1,color='b')
                
        plt.ylim([-21,21])
        plt.xlim([0,int(self.n_episode)])

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel('reward',fontsize=15)
        plt.xlabel('episode',fontsize=15)

        plt.savefig('rewardlist.png',dpi=150)



class DQN_agent_test():
    def __init__(self,num_image,devicename, madel_path, in_channels=1, action_space=[], learning_rate=0.2*1e-3, memory_size=100000, epsilon=0.99):
        
        self.device=torch.device(devicename)
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n     
                
        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.DQN = DQN(num_image,self.in_channels, self.action_dim).to(self.device)
        self.target_DQN = DQN(num_image,self.in_channels, self.action_dim).to(self.device)
        
        self.DQN.load_state_dict(torch.load(madel_path,map_location=devicename))
        
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=learning_rate, eps=0.001, alpha=0.95)
        
        
    def select_action(self, state):
        
        state = state.to(self.device)
        a = self.DQN(state).detach()
        q = a.max().cpu()
        action=a.max(1)[1].view(1,1)
        
            
        return action,q         


class DQN_agent_power():
    def __init__(self,num_image,devicename, madel_path, in_channels=1, action_space=[], learning_rate=0.2*1e-3, memory_size=100000, epsilon=0.99):
        
        self.device=torch.device(devicename)
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n     
                
        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.DQN = DQN_power(num_image,self.in_channels, self.action_dim).to(self.device)
        self.target_DQN = DQN_power(num_image,self.in_channels, self.action_dim).to(self.device)
        
        self.DQN.load_state_dict(torch.load(madel_path,map_location=devicename))
        
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=learning_rate, eps=0.001, alpha=0.95)
        
        
    def select_action(self, state, self_data_conv, self_data_all):
        
        state = state.to(self.device)
        action, data_conv, data_all = self.DQN(state)
        action = action.detach().max(1)[1].view(1,1)
        
        self_data_conv = torch.cat([self_data_conv,data_conv.cpu()])
        self_data_all = torch.cat([self_data_all,data_all.cpu()])
            
        return action, self_data_conv, self_data_all  



class Evaluator():

    def __init__(self, env, agent, n_episode, state_channel,
                 patch_size, frame_analyze, lca, num_image, dict_num, device):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        # self.losslist = []
        self.rewardlist = []
        self.sc=state_channel
        self.p_s=patch_size
        self.fa=frame_analyze
        self.frame_state=np.zeros((state_channel,patch_size*patch_size*frame_analyze))
        self.lca=lca
        self.ni=num_image
        self.dn=dict_num
        
        self.q=np.zeros((1))
        self.score=np.zeros((1))
        self.target_size = num_image*patch_size
        
        self.start=time.time()
        self.end=time.time()
        self.device=device
        
        
    def get_state(self,obs):
        
        t_u=10
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,self.target_size)
        frame=frame[np.newaxis,:,:]
        
        self.frame_state=tlca.video2image_piece(frame,self.p_s, 1,self.frame_state)
        state = torch.from_numpy(self.frame_state).to(self.device).float()
        a_state=tlca.run_lca(state.t(), self.lca, t_u, self.fa, self.p_s,self.device)
        a_state=torch.reshape(a_state,(self.dn,self.ni,self.ni))
        a_state=torch.unsqueeze(a_state,0).cpu()
        
        return a_state    
        
    
    def evaluate(self):
        r_reward=np.zeros(self.n_episode)
        q_value = np.zeros(self.n_episode)

        for episode in range(0,self.n_episode):
            
            obs = self.env.reset()
            state = self.get_state(obs)
            episode_reward = 0.0
            
            for t in count():  
                           
                action,q = self.agent.select_action(state)
                self.q=np.append(self.q,q)
                if RENDER:
                    self.env.render()
                
                
                obs,reward,done,info = self.env.step(action)
                episode_reward += reward
                if reward!=0:
                    self.score=np.append(self.score,reward)
                
                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None
                    self.frame_state=np.zeros((self.sc,self.p_s*self.p_s*self.fa))
                    
                    if episode%50==0:
                        print('epoch='+str(episode))
                        print('reward='+str(episode_reward))
                        
                
                state = next_state
                
                if done:
                    break
            
            r_reward[episode]=episode_reward
            q_value[episode]=self.q[1:].mean()
            self.q=np.zeros((1))
            self.rewardlist.append(episode_reward)
            
            
            self.env.close()
        return r_reward,q_value,self.score[1:101]
    
    def plot_average_reward(self):
        
        hdffile=h5py.File('DQN_lca_Pong_reward_'+str(self.n_episode)+'.h5','r')
        reward=np.array([hdffile['reward']])[0,:,:]
        hdffile.close()
        
        x=np.linspace(0,self.n_episode,int(self.n_episode/40+1))

        reward_mean=np.mean(reward,axis=1)
        reward_std=np.std(reward,axis=1)
        
        plt.figure(figsize=(6,4),dpi=200)
        
        plt.plot(x,reward_mean,color='b')
        plt.fill_between(x, reward_mean-reward_std,reward_mean+reward_std, color='b',alpha=0.2)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel('Performance',fontsize=15)
        plt.xlabel('Episode',fontsize=15)

        plt.savefig('evaluation_result.png',dpi=200)





def train_LCA_DQN(th,overcomplete,patch_size,target_size,frame_analyze,devicename):
    
    device=torch.device(devicename)
    # create environment
    env = gym.make("ALE/Pong-v5",obs_type='grayscale') #,render_mode='human',frameskip=4
    action_space = env.action_space
    
    #load lca dictionary
    dictfile=h5py.File('dict/dict'+str(th)+'.h5','r')
    dictionary=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    
    
    num_image=int(target_size/patch_size)
    input_size=dictionary.shape[0]
    dictionary_size=dictionary.shape[1]
    num_patch = int(num_image**2)
    
    #initialize
    dictionary=torch.from_numpy(dictionary).to(device)
    agent = DQN_agent(num_image,devicename,in_channels = dictionary_size, action_space= action_space, memory_size=MEMORY_SIZE)
    lca=tlca.TLCA(input_size,dictionary_size,th,dictionary,num_patch,device)
    
    #train
    trainer = Trainer(env, agent, n_episode,num_patch,patch_size,frame_analyze,lca,num_image,dictionary_size,device)
    trainer.train()
    
    ### comment when don't need visualization
    # trainer.plot_rewardlist()
    

      
def evaluate_LCA_DQN(th,overcomplete,patch_size,target_size,frame_analyze,devicename):
    device=torch.device(devicename)
    # create environment
    env = gym.make("ALE/Pong-v5",obs_type='grayscale') #,render_mode='human',frameskip=4
    action_space = env.action_space
    
    #get lca dictionary
    dictfile=h5py.File('dict/dict'+str(th)+'.h5','r')
    dictionary=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    
    
    num_image=int(target_size/patch_size)
    input_size=dictionary.shape[0]
    dictionary_size=dictionary.shape[1]
    num_patch = int(num_image**2)
    
    dictionary=torch.from_numpy(dictionary).to(device)
    start=time.time()
    end=time.time()
    
    """run every weights for 10 times"""
    num_e=76  #test weights saved every 40 episode. 3000/40+1=76
    reward_record=np.zeros((num_e,e_episode))
    q_record=np.zeros((num_e,e_episode))
    score_record=np.zeros((num_e,100))
    
    for i in range(0,num_e):
        p_str=str(i*40)
    
        madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode'+p_str+'.pt'
        agent = DQN_agent_test(num_image,devicename, madel_path,in_channels = dictionary_size, action_space= action_space, memory_size=MEMORY_SIZE)
        lca=tlca.TLCA(input_size,dictionary_size,th,dictionary,num_patch,device)
        
        evaluator = Evaluator(env, agent, e_episode, num_patch, patch_size, frame_analyze,lca,num_image,dictionary_size,device)
        reward_record[i,:],q_record[i,:],score_record[i,:] = evaluator.evaluate()
        
        localtime = time.asctime( time.localtime(time.time()) )
        print (localtime)
        end=time.time()
        print('total time',end-start)
        print('episode='+p_str+' is done')
        print('average reward='+str(reward_record[i,:].mean()))
        print('average q='+str(q_record[i,:].mean()))
        start=time.time()
        
        hdffile=h5py.File('reward_play_'+str(target_size)+'_'+str(frame_analyze)+'.h5','w')
        hdffile.create_dataset('reward',data=reward_record)
        hdffile.create_dataset('q',data=q_record)
        hdffile.create_dataset('score',data=score_record)
        hdffile.close()
        
        ### comment when don't need visualization
        
        # if i == (num_e-1):
        #     evaluator.plot_average_reward()


    
def evaluate_LCA_DQN_best(th,overcomplete,patch_size,target_size,frame_analyze,devicename):
    device=torch.device(devicename)
    dictfile=h5py.File('reward_play_'+str(target_size)+'_'+str(frame_analyze)+'.h5','r')
    reward=np.array([dictfile['reward']])[0,:,:]
    dictfile.close()
    
    reward_mean=np.mean(reward,axis=1)
    p_str=str(40*int(np.where(reward_mean==reward_mean.max())[0]))
    
    # create environment
    env = gym.make("ALE/Pong-v5",obs_type='grayscale') #,render_mode='human',frameskip=4
    action_space = env.action_space
    
    #get lca dictionary
    dictfile=h5py.File('dict/dict'+str(th)+'.h5','r')
    dictionary=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    
    
    num_image=int(target_size/patch_size)
    input_size=dictionary.shape[0]
    dictionary_size=dictionary.shape[1]
    num_patch = int(num_image**2)
    
    dictionary=torch.from_numpy(dictionary).to(device)
    start=time.time()
    end=time.time()
    
    # num_e=76  #test weights saved every 40 episode. 3000/40+1=76
    reward_record=np.zeros(e_episode_final)
    

    madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode'+p_str+'.pt'
    agent = DQN_agent_test(num_image,devicename, madel_path,in_channels = dictionary_size, action_space= action_space, memory_size=MEMORY_SIZE)
    lca=tlca.TLCA(input_size,dictionary_size,th,dictionary,num_patch,device)
    
    evaluator = Evaluator(env, agent, e_episode_final, num_patch, patch_size, frame_analyze,lca,num_image,dictionary_size,device)
    reward_record,q,score = evaluator.evaluate()
    
    hdffile=h5py.File('reward_best_'+str(target_size)+'_'+str(frame_analyze)+'.h5','w')
    hdffile.create_dataset('reward',data=reward_record)
    hdffile.close()
    
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    end=time.time()
    print('total time',end-start)
    print('episode='+p_str+' is done')
    print('average reward='+str(reward_record.mean()))
    start=time.time()
    
      
    
  
