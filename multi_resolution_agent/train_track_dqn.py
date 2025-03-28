import tlca
import dqn
import h5py, gym, time, torch
from itertools import count
import numpy as np
import os
# import skvideo.io

INITIAL_MEMORY=10000
TARGET_UPDATE = 1000
memorysize = 100000
n_episode = 10

MODEL_STORE_PATH = os.getcwd()
modelname = 'DQN_Pong'

class Trainer():
    def __init__(self, env,device, agent, agent_play, epoch_lca,state_channel,
                 patch_size,fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,
                 dict_num,fovea_size, periphery_size):
        self.env = env
        self.device = device
        self.epoch_lca = epoch_lca
        self.agent = agent
        self.agent_play = agent_play
        
        self.rewardlist = []
        self.sc=state_channel
        self.i_s=patch_size
        self.fa=fa
        self.new_size = int((18+fovea_patch_num)*patch_size[0])
        
        self.lca_traindata=torch.zeros((1,patch_size[0]*patch_size[0]*fa[0])).to(device)
        self.input_size=patch_size[0]*patch_size[0]*fa[0]
       
        self.frame_state_f=np.zeros((state_channel[0],patch_size[0]*patch_size[0]*fa[0]))
        
        self.frame_state_p=np.zeros((state_channel[1],patch_size[1]*patch_size[1]*fa[1]))
        self.frame_state=torch.zeros((4,80,80)).float()
         
        self.lca_f=lca_f
        self.lca_p=lca_p
        
        self.ps=periphery_patch_num
        self.dn=dict_num
        self.fs=fovea_patch_num
        self.rdd = int((self.new_size-fovea_size)/4*patch_size[0]) #扩大的范围
        
        self.fovea_size = fovea_size
        self.periphery_size = periphery_size
        
        self.bug_solve=0
        self.bug = 0
        self.on_target = 0
        
        self.start=time.time()
        self.end=time.time()
        
    
        
        
    def get_state(self,obs,fovea_pos):
        
        t_u=10
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,80)
        frame = torch.from_numpy(frame)
        state = torch.zeros((4,80,80)).float()
        state[:3,:,:]=self.frame_state[1:,:,:]
        state[3,:,:]=frame
        self.frame_state=state
        state=torch.unsqueeze(state,0)
        
        
        
        t_u=10
        
        frame = tlca.resize_frame(obs,self.fovea_size)
        frame = frame[np.newaxis,:,:]
         
        frame_f = np.zeros((self.new_size,self.new_size))
        frame_f[self.rdd:self.rdd+self.fovea_size,self.rdd:self.rdd+self.fovea_size] = frame
        fovea_frame = frame_f[y0*self.i_s[0]:(y0+self.fs)*self.i_s[0],x0*self.i_s[0]:(x0+self.fs)*self.i_s[0]]
        
        
        frame_f=fovea_frame[np.newaxis,:,:]
        
        self.frame_state_f=tlca.video2image_piece(frame_f,self.i_s[0], 1,self.frame_state_f)
        frame_state_f = torch.from_numpy(self.frame_state_f).to(self.device).float()
        a_state_f = tlca.run_lca(frame_state_f.t(), self.lca_f, t_u, self.fa[0],self.device)#, self.i_s[0],self.sc[0]
        a_state_f = torch.reshape(a_state_f,(self.dn[0],self.fs,self.fs))
        
        state_lca = torch.unique(frame_state_f,dim=0)
        self.lca_traindata=torch.cat((self.lca_traindata,state_lca),dim=0).to(self.device)
        a_state_f=torch.unsqueeze(a_state_f,0).cpu()
        
        
        
        frame_p = tlca.resize_frame(obs,self.periphery_size)
        
        frame_p=frame_p[np.newaxis,:,:]
        
        self.frame_state_p=tlca.video2image_piece(frame_p,self.i_s[1], 1,self.frame_state_p)
        frame_state_p = torch.from_numpy(self.frame_state_p).to(self.device).float()
        a_state_p = tlca.run_lca(frame_state_p.t(), self.lca_p, t_u, self.fa[1],self.device)#, self.i_s[1],self.sc[1]
        a_state_p = torch.reshape(a_state_p,(self.dn[1],self.ps,self.ps))

        a_state_p=torch.unsqueeze(a_state_p,0).cpu()
        
        return a_state_f,a_state_p, state   
    
        


    def move_fovea(self,action_fovea,reward):
        
        
        action1 = int(action_fovea[0,0]-2)
        action2 = int(action_fovea[0,1]-2)
        
        
        if action1 >= 0:
            self.agent.fovea_pos[0]=min(18,self.agent.fovea_pos[0]+action1)
        else:
            self.agent.fovea_pos[0]=max(0,self.agent.fovea_pos[0]+action1)
                
        if action2 >= 0:
            self.agent.fovea_pos[1]=min(18,self.agent.fovea_pos[1]+action2)
        else:
            self.agent.fovea_pos[1]=max(0,self.agent.fovea_pos[1]+action2)
            
        
        
        
        
        
    def get_reward(self,obs):
        
        fovea_pos=self.agent.fovea_pos
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,self.fovea_size)
        
        frame[frame<3]=0
        
        if  frame.sum()<1:
            
            if self.bug_solve >1:
                self.agent.fovea_pos=np.array([9,9])
            
            self.bug_solve += 1
            self.bug+=1
            return 0
        
        self.bug_solve = 0
        
        frame_f = np.zeros((self.new_size,self.new_size))
        frame_f[self.rdd:self.rdd+self.fovea_size,self.rdd:self.rdd+self.fovea_size] = frame
        
        dis_nc = int((self.fs-2)/2)
        
        center = frame_f[(y0+dis_nc)*self.i_s[0]:(y0+self.fs-dis_nc)*self.i_s[0],(x0+dis_nc)*self.i_s[0]:(x0+self.fs-dis_nc)*self.i_s[0]]
        
        
        if center.sum()>=1:
            self.on_target+=1
            return 0.01
        
        fovea_area = frame_f[y0*self.i_s[0]:(y0+self.fs)*self.i_s[0],x0*self.i_s[0]:(x0+self.fs)*self.i_s[0]]
        
        if fovea_area.sum()<1:
            return -0.1
        
        return -0.05
        
        
    
    def train(self):

        for episode in range(0,self.epoch_lca):
            
            obs = self.env.reset()
            state_f,state_p,state_play = self.get_state(obs,self.agent.fovea_pos)
            episode_reward = 0.0
            action_before=torch.tensor([[2,2]],dtype=int).to(self.device)
            accurate_rate=0
            for t in count():                             
                action_paddle = self.agent_play.select_action(state_play)
                
                action1, action2 = self.agent.select_action(state_f,state_p,action_before)
                action = torch.cat([action1,action2],dim=1)
                
                
                obs,reward_,done,info = self.env.step(action_paddle)
                self.move_fovea(action,reward_)
                
                reward = self.get_reward(obs)
                episode_reward += reward
                
                
                if not done:
                    next_state_f, next_state_p, next_state_play = self.get_state(obs,self.agent.fovea_pos)
                    
                else:
                    next_state_f = None
                    next_state_p = None
                    self.frame_state = torch.zeros((4,80,80)).float()
                    self.frame_state_f=np.zeros((self.sc[0],self.i_s[0]*self.i_s[0]*self.fa[0]))
                    self.frame_state_p=np.zeros((self.sc[1],self.i_s[1]*self.i_s[1]*self.fa[1]))
                reward = torch.tensor([reward], device=self.device)
                
                # 将四元组存到memory中
                '''
                state: batch_size channel h w    size: batch_size * 2
                action_before
                action: size: batch_size * 1
                next_state: batch_size channel h w    size: batch_size * 4
                reward: size: batch_size * 1                
                '''
                self.agent.memory_buffer.push(state_f,state_p, action_before.to('cpu'), action.to('cpu'), next_state_f,next_state_p, reward.to('cpu')) # 里面的数据都是Tensor
                state_f = next_state_f
                state_p = next_state_p
                state_play = next_state_play
                action_before=action
                # 经验池满了之后开始学习
                if self.agent.stepdone > INITIAL_MEMORY:
                    self.agent.learn()
                    if self.agent.stepdone % TARGET_UPDATE == 0:
                        self.agent.target_DQN.load_state_dict(self.agent.DQN.state_dict())
                
                if done:
                    
                    accurate_rate=round(self.on_target/(t-self.bug),3)
                    self.rewardlist.append(accurate_rate)
                    
                    self.lca_traindata = torch.unique(self.lca_traindata,dim=0)
                    if episode<201:
                        self.lca_f,error = tlca.train_model(self.lca_f,self.input_size,self.lca_traindata,self.device)
                    self.lca_f.u=torch.zeros(self.lca_f.dict_size,self.sc[0]).to(self.device)
                    self.lca_f.a=torch.zeros(self.lca_f.dict_size,self.sc[0]).to(self.device)
                    
                    if episode%10==0:
                        hdffile=h5py.File('dict/dict_tracking'+str(episode)+'.h5','w') 
                        hdffile.create_dataset('dictionary',data=self.lca_f.dictionary.cpu().numpy())
                        hdffile.close()
                        if episode<201:
                            print('cost=',error)
                        
                    
                    self.lca_traindata=torch.zeros((1,self.i_s[0]*self.i_s[0]*self.fa[0])).to(self.device)
                    
                    break
            # print(episode_reward)
            if episode % 10 == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print (localtime)
                self.end=time.time()
                print('total time',self.end-self.start)
                
                torch.save(self.agent.DQN.state_dict(), MODEL_STORE_PATH + '/' + "track_model/{}_episode{}.pt".format(modelname, episode))
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.agent.stepdone, episode, t, accurate_rate))
                self.start=time.time()
                
                hdffile=h5py.File('rewardlist.h5','w')
                hdffile.create_dataset('data',data=self.rewardlist)
                hdffile.close()
                
            
            self.bug=0 
            self.on_target=0
            self.env.close()
        return


class Evaluator():
    def __init__(self, env,device, agent, agent_play,state_channel,patch_size,
                 fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,
                 dict_num,fovea_size, periphery_size):
        self.env = env
        self.device = device
        self.n_episode = n_episode
        self.agent = agent
        self.agent_play = agent_play
        
        self.rewardlist = []
        self.sc=state_channel
        self.i_s=patch_size
        self.fa=fa
        self.new_size = int((18+fovea_patch_num)*patch_size[0])
        
        self.input_size=patch_size[0]*patch_size[0]*fa[0]
       
        self.frame_state_f=np.zeros((state_channel[0],patch_size[0]*patch_size[0]*fa[0]))
        
        self.frame_state_p=np.zeros((state_channel[1],patch_size[1]*patch_size[1]*fa[1]))
        self.frame_state=torch.zeros((4,80,80)).float()
         
        self.lca_f=lca_f
        self.lca_p=lca_p
        
        self.ps=periphery_patch_num
        self.dn=dict_num
        self.fs=fovea_patch_num
        self.rdd = int((self.new_size-fovea_size)/4*patch_size[0])#扩大的范围
        
        self.fovea_size = fovea_size
        self.periphery_size = periphery_size
        
        self.bug_solve=0
        self.bug = 0
        self.on_target2=0
        self.on_target4=0
        self.on_target=0
        self.off_target=0
        self.error=0
        
        self.start=time.time()
        self.end=time.time()
        
        
        
    def get_state(self,obs,fovea_pos):
        
        t_u=10
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,80)
        frame = torch.from_numpy(frame)
        state = torch.zeros((4,80,80)).float()
        state[0:3,:,:]=self.frame_state[1:4,:,:]
        state[3,:,:]=frame
        self.frame_state=state
        state=torch.unsqueeze(state,0)
        
        
        
        t_u=10
        
        frame = tlca.resize_frame(obs,self.fovea_size)
        frame=frame[np.newaxis,:,:]
         
        frame_f = np.zeros((self.new_size,self.new_size))
        frame_f[self.rdd:self.rdd+self.fovea_size,self.rdd:self.rdd+self.fovea_size] = frame
        fovea_frame = frame_f[y0*self.i_s[0]:(y0+self.fs)*self.i_s[0],x0*self.i_s[0]:(x0+self.fs)*self.i_s[0]]
        
        
        frame_f=fovea_frame[np.newaxis,:,:]
        
        self.frame_state_f=tlca.video2image_piece(frame_f,self.i_s[0], 1,self.frame_state_f)
        frame_state_f = torch.from_numpy(self.frame_state_f).to(self.device).float()
        a_state_f,error = tlca.run_lca_test(frame_state_f.t(), self.lca_f, t_u, self.fa[0],self.device)
        a_state_f = torch.reshape(a_state_f,(self.dn[0],self.fs,self.fs))
        
        a_state_f=torch.unsqueeze(a_state_f,0).cpu()
        self.error+=error.cpu()
        
        
        frame_p = tlca.resize_frame(obs,self.periphery_size)
        
        frame_p=frame_p[np.newaxis,:,:]
        
        self.frame_state_p=tlca.video2image_piece(frame_p,self.i_s[1], 1,self.frame_state_p)
        frame_state_p = torch.from_numpy(self.frame_state_p).to(self.device).float()
        a_state_p = tlca.run_lca(frame_state_p.t(), self.lca_p, t_u, self.fa[1],self.device) 
        a_state_p = torch.reshape(a_state_p,(self.dn[1],self.ps,self.ps))

        a_state_p=torch.unsqueeze(a_state_p,0).cpu()
        
        return a_state_f,a_state_p, state   
    
        


    def move_fovea(self,action_fovea,reward):
        
        
        action1 = int(action_fovea[0,0]-2)
        action2 = int(action_fovea[0,1]-2)
        
        
        if action1 >= 0:
            self.agent.fovea_pos[0]=min(18,self.agent.fovea_pos[0]+action1)
        else:
            self.agent.fovea_pos[0]=max(0,self.agent.fovea_pos[0]+action1)
                
        if action2 >= 0:
            self.agent.fovea_pos[1]=min(18,self.agent.fovea_pos[1]+action2)
        else:
            self.agent.fovea_pos[1]=max(0,self.agent.fovea_pos[1]+action2)
            
        
        
        
        
        
    def get_reward(self,obs):
        
        fovea_pos=self.agent.fovea_pos
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,self.fovea_size)
        
        frame[frame<3]=0
        
        if  frame.sum()<1:
            
            if self.bug_solve >1:
                self.agent.fovea_pos=np.array([9,9])
            
            self.bug_solve += 1
            self.bug+=1
            return 0
        
        self.bug_solve = 0
        
        frame_f = np.zeros((self.new_size,self.new_size))
        frame_f[self.rdd:self.rdd+self.fovea_size,self.rdd:self.rdd+self.fovea_size] = frame
        
        
        dis_nc = int((self.fs-2)/2)
        
        center = frame_f[(y0+dis_nc)*self.i_s[0]:(y0+self.fs-dis_nc)*self.i_s[0],(x0+dis_nc)*self.i_s[0]:(x0+self.fs-dis_nc)*self.i_s[0]]
        
        
        if center.sum()>=1:
            self.on_target2+=1
        
        center2 = frame_f[(y0+dis_nc-1)*self.i_s[0]:(y0+self.fs+1-dis_nc)*self.i_s[0],(x0+dis_nc-1)*self.i_s[0]:(x0+self.fs+1-dis_nc)*self.i_s[0]]
        
        if center2.sum()>=1:
            self.on_target4+=1
        
        fovea_area = frame_f[y0*self.i_s[0]:(y0+self.fs)*self.i_s[0],x0*self.i_s[0]:(x0+self.fs)*self.i_s[0]]
        
        if fovea_area.sum()>=1:
            self.on_target+=1
        
        return -1
        
        
    
    def evaluate(self):
        e_epoch=np.zeros((self.n_episode,))
        r_reward=np.zeros((self.n_episode,3))

        for episode in range(0,self.n_episode):
            
            obs = self.env.reset()
            state_f,state_p,state_play = self.get_state(obs,self.agent.fovea_pos)
            episode_reward = 0.0
            action_before=torch.tensor([[2,2]],dtype=int).to(self.device)
            
            for t in count():                             
                action_paddle = self.agent_play.select_action(state_play)
                
                action1, action2 = self.agent.select_action(state_f,state_p,action_before)
                action = torch.cat([action1,action2],dim=1)
                
                
                obs,reward_,done,info = self.env.step(action_paddle)
                self.move_fovea(action,reward_)
                
                reward = self.get_reward(obs)
                episode_reward += reward
                
                
                if not done:
                    next_state_f, next_state_p, next_state_play = self.get_state(obs,self.agent.fovea_pos)
                    
                else:
                    next_state_f = None
                    next_state_p = None
                    self.frame_state = torch.zeros((4,80,80)).float()
                    self.frame_state_f=np.zeros((self.sc[0],self.i_s[0]*self.i_s[0]*self.fa[0]))
                    self.frame_state_p=np.zeros((self.sc[1],self.i_s[1]*self.i_s[1]*self.fa[1]))
                reward = torch.tensor([reward], device=self.device)
                
                
                state_f = next_state_f
                state_p = next_state_p
                state_play = next_state_play
                action_before=action
                
                if done:
                    r_reward[episode,0] = round(self.on_target2/(t-self.bug),3)
                    r_reward[episode,1] = round(self.on_target4/(t-self.bug),3)
                    r_reward[episode,2] = round(self.on_target/(t-self.bug),3)
                    e_epoch[episode] = self.error/t
                    
                    
                    break
            
            self.bug=0
            self.rewardlist.append(episode_reward)
            self.on_target=0   
            self.on_target2=0
            self.on_target4=0
            self.error=0
            self.env.close()
        return r_reward, e_epoch


class Evaluator_video():
    def __init__(self, env,device, agent, agent_play,state_channel,patch_size,
                 fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,
                 dict_num,fovea_size, periphery_size):
        self.env = env
        self.device = device
        self.agent = agent
        self.agent_play = agent_play
        
        self.rewardlist = []
        self.sc=state_channel
        self.i_s=patch_size
        self.fa=fa
        self.new_size = int((18+fovea_patch_num)*patch_size[0])
        
        self.input_size=patch_size[0]*patch_size[0]*fa[0]
       
        self.frame_state_f=np.zeros((state_channel[0],patch_size[0]*patch_size[0]*fa[0]))
        
        self.frame_state_p=np.zeros((state_channel[1],patch_size[1]*patch_size[1]*fa[1]))
        self.frame_state=torch.zeros((4,80,80)).float()
         
        self.lca_f=lca_f
        self.lca_p=lca_p
        
        self.ps=periphery_patch_num
        self.dn=dict_num
        self.fs=fovea_patch_num
        self.rdd = int((self.new_size-fovea_size)/4*patch_size[0])#扩大的范围
        
        self.fovea_size = fovea_size
        self.periphery_size = periphery_size
        
        self.bug_solve=0
        self.bug = 0
        self.on_target2=0
        self.on_target4=0
        self.on_target=0
        self.off_target=0
        
        self.start=time.time()
        self.end=time.time()
        
        
        
    def get_state(self,obs,fovea_pos):
        
        t_u=10
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,80)
        frame = torch.from_numpy(frame)
        state = torch.zeros((4,80,80)).float()
        state[0:3,:,:]=self.frame_state[1:4,:,:]
        state[3,:,:]=frame
        self.frame_state=state
        state=torch.unsqueeze(state,0)
        
        
        
        t_u=10
        
        frame = tlca.resize_frame(obs,self.fovea_size)
        frame=frame[np.newaxis,:,:]
         
        frame_f = np.zeros((self.new_size,self.new_size))
        frame_f[self.rdd:self.rdd+self.fovea_size,self.rdd:self.rdd+self.fovea_size] = frame
        fovea_frame = frame_f[y0*self.i_s[0]:(y0+self.fs)*self.i_s[0],x0*self.i_s[0]:(x0+self.fs)*self.i_s[0]]
        
        
        frame_f=fovea_frame[np.newaxis,:,:]
        
        self.frame_state_f=tlca.video2image_piece(frame_f,self.i_s[0], 1,self.frame_state_f)
        frame_state_f = torch.from_numpy(self.frame_state_f).to(self.device).float()
        a_state_f = tlca.run_lca(frame_state_f.t(), self.lca_f, t_u, self.fa[0], self.device)
        a_state_f = torch.reshape(a_state_f,(self.dn[0],self.fs,self.fs))
        
        a_state_f=torch.unsqueeze(a_state_f,0).cpu()
        
        
        
        frame_p = tlca.resize_frame(obs,self.periphery_size)
        
        frame_p=frame_p[np.newaxis,:,:]
        
        self.frame_state_p=tlca.video2image_piece(frame_p,self.i_s[1], 1,self.frame_state_p)
        frame_state_p = torch.from_numpy(self.frame_state_p).to(self.device).float()
        a_state_p = tlca.run_lca(frame_state_p.t(), self.lca_p, t_u, self.fa[1],self.device)
        a_state_p = torch.reshape(a_state_p,(self.dn[1],self.ps,self.ps))

        a_state_p=torch.unsqueeze(a_state_p,0).cpu()
        
        return a_state_f,a_state_p, state   
    
        


    def move_fovea(self,action_fovea,reward):
        
        
        action1 = int(action_fovea[0,0]-2)
        action2 = int(action_fovea[0,1]-2)
        
        
        if action1 >= 0:
            self.agent.fovea_pos[0]=min(18,self.agent.fovea_pos[0]+action1)
        else:
            self.agent.fovea_pos[0]=max(0,self.agent.fovea_pos[0]+action1)
                
        if action2 >= 0:
            self.agent.fovea_pos[1]=min(18,self.agent.fovea_pos[1]+action2)
        else:
            self.agent.fovea_pos[1]=max(0,self.agent.fovea_pos[1]+action2)
            
        
        
        
        
        
    def get_reward(self,obs):
        
        fovea_pos=self.agent.fovea_pos
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,self.fovea_size)
        
        frame[frame<3]=0
        
        if  frame.sum()<1:
            
            if self.bug_solve >1:
                self.agent.fovea_pos=np.array([9,9])
            
            self.bug_solve += 1
            self.bug+=1
            return 0
        
        self.bug_solve = 0
        
        frame_f = np.zeros((self.new_size,self.new_size))
        frame_f[self.rdd:self.rdd+self.fovea_size,self.rdd:self.rdd+self.fovea_size] = frame
        
        dis_nc = int((self.fs-2)/2)
        
        center = frame_f[(y0+dis_nc)*self.i_s[0]:(y0+self.fs-dis_nc)*self.i_s[0],(x0+dis_nc)*self.i_s[0]:(x0+self.fs-dis_nc)*self.i_s[0]]
        
        
        if center.sum()>=1:
            self.on_target2+=1
        
        center2 = frame_f[(y0+dis_nc-1)*self.i_s[0]:(y0+self.fs+1-dis_nc)*self.i_s[0],(x0+dis_nc-1)*self.i_s[0]:(x0+self.fs+1-dis_nc)*self.i_s[0]]
        
        if center2.sum()>=1:
            self.on_target4+=1
        
        fovea_area = frame_f[y0*self.i_s[0]:(y0+self.fs)*self.i_s[0],x0*self.i_s[0]:(x0+self.fs)*self.i_s[0]]
        
        if fovea_area.sum()>=1:
            self.on_target+=1
        
        return -1
        
        
    
    def evaluate(self):
        
        r_reward=np.zeros((1,3))

        for episode in range(0,1):
            
            obs = self.env.reset()
            state_f,state_p,state_play = self.get_state(obs,self.agent.fovea_pos)
            episode_reward = 0.0
            action_before=torch.tensor([[2,2]],dtype=int).to(self.device)
            
            """video range"""
            steps=7000
            e_video=np.zeros((steps,self.new_size,self.new_size))
            
            for t in count():                             
                action_paddle = self.agent_play.select_action(state_play)
                
                action1, action2 = self.agent.select_action(state_f,state_p,action_before)
                action = torch.cat([action1,action2],dim=1)
                
                
                obs,reward_,done,info = self.env.step(action_paddle)
                self.move_fovea(action,reward_)
                
                reward = self.get_reward(obs)
                episode_reward += reward
                
                observation = obs.astype(float)
                frame = tlca.resize_frame(observation,self.fovea_size)
                e_video[t,self.rdd:self.rdd+self.fovea_size,self.rdd:self.rdd+self.fovea_size] = frame*30
                
                y0=self.agent.fovea_pos[0]
                x0=self.agent.fovea_pos[1]
        
                e_video[t,y0*self.i_s[0]:(y0+self.fs)*self.i_s[0],x0*self.i_s[0]:(x0+self.fs)*self.i_s[0]] += 100
                
                if not done:
                    next_state_f, next_state_p, next_state_play = self.get_state(obs,self.agent.fovea_pos)
                    
                else:
                    next_state_f = None
                    next_state_p = None
                    self.frame_state = torch.zeros((4,80,80)).float()
                    self.frame_state_f=np.zeros((self.sc[0],self.i_s[0]*self.i_s[0]*self.fa[0]))
                    self.frame_state_p=np.zeros((self.sc[1],self.i_s[1]*self.i_s[1]*self.fa[1]))
                reward = torch.tensor([reward], device=self.device)
                
                
                state_f = next_state_f
                state_p = next_state_p
                state_play = next_state_play
                action_before=action
                
                if done:
                    r_reward[episode,0] = round(self.on_target2/(t-self.bug),3)
                    r_reward[episode,1] = round(self.on_target4/(t-self.bug),3)
                    r_reward[episode,2] = round(self.on_target/(t-self.bug),3)
                    
                    e_video=e_video[:t,:,:]
                    
                    # vid_out=skvideo.io.FFmpegWriter("tracking.avi",inputdict={'-r': '5'})
                    # for i in range(e_video.shape[0]):
                    #     new_v = e_video[i,:,:]
                    #     vid_out.writeFrame(new_v)
                    # vid_out.close()
                    # print("hahhaha")
                    break
            
            self.bug=0
            self.rewardlist.append(episode_reward)
            self.on_target=0   
            self.on_target2=0
            self.on_target4=0
            self.env.close()
        return r_reward



def train_lca_track(periphery_size, fovea_size, fovea_patch_num, 
                    frame_analyze, patch_size, overcomplete,epoch_lca,
                    threshold, device_name):
    
    
    env = gym.make("ALE/Pong-v5",obs_type='grayscale')#,render_mode='human'
    action_space = env.action_space
    
    device = torch.device(device_name)
    
    dictfile=h5py.File('dict/dict'+str(periphery_size)+'.h5','r')
    dict_p=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_p=torch.from_numpy(dict_p).to(device)
    
    fa=np.zeros(3,dtype=int)
    fa[0]=frame_analyze
    fa[1]=frame_analyze
    fa[2]=4
    
    input_pixel=fa[0]*patch_size[0]**2
    dictionary=torch.rand(input_pixel,input_pixel*overcomplete).to(device)-0.5
    dict_f = dictionary/torch.linalg.norm(dictionary,axis=0).to(device)
    
    input_size = np.zeros(2,dtype=int)
    input_size[0]=dict_f.shape[0]
    input_size[1]=dict_p.shape[0]
    
    
    dict_num = np.zeros(2,dtype=int)
    dict_num[0] = dict_f.shape[1]
    dict_num[1] = dict_p.shape[1]
    
    periphery_patch_num = int(periphery_size/patch_size[1])
    state_channel = np.zeros(2,dtype=int)
    state_channel[0] = int(fovea_patch_num**2)
    state_channel[1] = int(periphery_patch_num**2)
    
    
    
    agent = dqn.DQN_agent_track(fovea_patch_num, periphery_patch_num, device_name, dict_num, memory_size=memorysize)
    agent_play = dqn.DQN_agent_guide(device_name, in_channels = 4, action_space= action_space)
    
    lca_f=tlca.TLCA(input_size[0],dict_num[0],threshold,dict_f,state_channel[0],device)
    lca_p=tlca.TLCA(input_size[1],dict_num[1],threshold,dict_p,state_channel[1],device)
    
    
    trainer = Trainer(env,device, agent, agent_play, epoch_lca,state_channel,
                      patch_size,fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,
                      dict_num, fovea_size, periphery_size)
    trainer.train()




def evaluate_lca_track(periphery_size, fovea_size, fovea_patch_num, 
                    frame_analyze, patch_size, overcomplete,epoch_lca,
                    threshold, device_name):
    
    # create environment
   
    
    
    device = torch.device(device_name)
    
    dictfile=h5py.File('dict/dict'+str(periphery_size)+'.h5','r')
    dict_p=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_p=torch.from_numpy(dict_p).to(device)
    
    fa=np.zeros(3,dtype=int)
    fa[0]=frame_analyze
    fa[1]=frame_analyze
    fa[2]=4
    
    input_pixel=fa[0]*patch_size[0]**2
    
    input_size = np.zeros(2,dtype=int)
    input_size[0]=input_pixel
    input_size[1]=dict_p.shape[0]
    
    
    dict_num = np.zeros(2,dtype=int)
    dict_num[0] = int(input_pixel*overcomplete)
    dict_num[1] = dict_p.shape[1]
    
    periphery_patch_num = int(periphery_size/patch_size[1])
    state_channel = np.zeros(2,dtype=int)
    state_channel[0] = int(fovea_patch_num**2)
    state_channel[1] = int(periphery_patch_num**2)
    
    
    lca_p=tlca.TLCA(input_size[1],dict_num[1],threshold,dict_p,state_channel[1],device)
    
    start=time.time()
    end=time.time()
    
    """run every weights for 10 times"""
    num_e=int(epoch_lca/10+1)  #test weights saved every 40 episode. 3000/40+1=76
    reward_record=np.zeros((num_e,n_episode,3))
    error_record=np.zeros((num_e,n_episode))
    
    for i in range(0,num_e):
        i_str=str(i*10)
        
        env = gym.make("ALE/Pong-v5",obs_type='grayscale') 
        action_space = env.action_space
        agent_play = dqn.DQN_agent_guide(device_name, in_channels = 4, action_space= action_space)
        
        dictfile=h5py.File('dict/dict_tracking'+i_str+'.h5','r')
        dict_f=np.array([dictfile['dictionary']])[0,:,:]
        dictfile.close()
        dict_f=torch.from_numpy(dict_f).to(device)
        lca_f=tlca.TLCA(input_size[0],dict_num[0],threshold,dict_f,state_channel[0],device)
    
    
        model_path = MODEL_STORE_PATH + '/' + 'track_model/' + 'DQN_Pong_episode'+i_str+'.pt'
        agent = dqn.DQN_agent_track_evaluate(fovea_patch_num, periphery_patch_num, device_name, model_path, dict_num)
        
        evaluator = Evaluator(env,device, agent, agent_play, state_channel,
                      patch_size,fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,
                      dict_num, fovea_size, periphery_size)
        reward_record[i,:,:],error_record[i,:] = evaluator.evaluate()
        
        localtime = time.asctime( time.localtime(time.time()) )
        print (localtime)
        end=time.time()
        print('total time',end-start)
        print('episode='+i_str+' is done')
        print('on_target2 reward='+str(round(reward_record[i,:,0].mean(),3)))
        print('on_target4 reward='+str(round(reward_record[i,:,1].mean(),3)))
        print('on_target reward='+str(round(reward_record[i,:,2].mean(),3)))
        print('error='+str(round(error_record[i,:].mean(),3)))
        start=time.time()
        
        
        hdffile=h5py.File('reward_track.h5','w')
        hdffile.create_dataset('reward',data=reward_record)
        hdffile.close()
        hdffile=h5py.File('error_track.h5','w')
        hdffile.create_dataset('reward',data=error_record)
        hdffile.close()




def video_lca_track(periphery_size, fovea_size, fovea_patch_num, 
                    frame_analyze, patch_size, overcomplete,which_epoch,
                    threshold, device_name):
    
    # create environment
   
    
    
    device = torch.device(device_name)
    
    dictfile=h5py.File('dict/dict'+str(periphery_size)+'.h5','r')
    dict_p=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_p=torch.from_numpy(dict_p).to(device)
    
    fa=np.zeros(3,dtype=int)
    fa[0]=frame_analyze
    fa[1]=frame_analyze
    fa[2]=4
    
    input_pixel=fa[0]*patch_size[0]**2
    
    input_size = np.zeros(2,dtype=int)
    input_size[0]=input_pixel
    input_size[1]=dict_p.shape[0]
    
    
    dict_num = np.zeros(2,dtype=int)
    dict_num[0] = int(input_pixel*overcomplete)
    dict_num[1] = dict_p.shape[1]
    
    periphery_patch_num = int(periphery_size/patch_size[1])
    state_channel = np.zeros(2,dtype=int)
    state_channel[0] = int(fovea_patch_num**2)
    state_channel[1] = int(periphery_patch_num**2)
    
    reward_record=np.zeros((1,1,3))
    
    lca_p=tlca.TLCA(input_size[1],dict_num[1],threshold,dict_p,state_channel[1],device)
    
    start=time.time()
    end=time.time()
    
    env = gym.make("ALE/Pong-v5",obs_type='grayscale') 
    action_space = env.action_space
    agent_play = dqn.DQN_agent_guide(device_name, in_channels = 4, action_space= action_space)
        
    
    i_str=str(which_epoch)
        
        
    dictfile=h5py.File('dict/dict_tracking'+i_str+'.h5','r')
    dict_f=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_f=torch.from_numpy(dict_f).to(device)
    lca_f=tlca.TLCA(input_size[0],dict_num[0],threshold,dict_f,state_channel[0],device)
    
    
    model_path = MODEL_STORE_PATH + '/' + 'track_model/' + 'DQN_Pong_episode'+i_str+'.pt'
    agent = dqn.DQN_agent_track_evaluate(fovea_patch_num, periphery_patch_num, device_name, model_path, dict_num)
        
    evaluator_v = Evaluator_video(env,device, agent, agent_play,state_channel,
                          patch_size,fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,
                          dict_num, fovea_size, periphery_size)
    reward_record=evaluator_v.evaluate()
        
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    end=time.time()
    print('total time',end-start)
    print('episode='+i_str+' is done')
    print('on_target2 reward='+str(round(reward_record[0,0].mean(),3)))
    print('on_target4 reward='+str(round(reward_record[0,1].mean(),3)))
    print('on_target reward='+str(round(reward_record[0,2].mean(),3)))

