import numpy as np
import torch
import tlca
import dqn
import gym, os.path
from itertools import count
import h5py,time
# import skvideo.io

MODEL_STORE_PATH = os.getcwd()
n_episode = 20
n_episode_final = 1000
INITIAL_MEMORY = 10000
TARGET_UPDATE = 1000
modelname = 'DQN_Pong'
memorysize = 500000
epoch = 3001

class Trainer():
    def __init__(self, env,device, agent_play,state_channel,patch_size,
                 fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,dict_num, 
                 fovea_size, periphery_size):
        
        self.device = device
        self.env = env
        self.epoch = epoch
        self.agent_play = agent_play
        
        # self.losslist = []
        self.rewardlist = []
        self.sc=state_channel
        self.i_s=patch_size
        self.fa=fa
        
        
        self.input_size=patch_size[0]*patch_size[0]*fa[0]
       
        self.frame_state_f=np.zeros((state_channel[0],patch_size[0]*patch_size[0]*fa[0]))
        self.frame_state_p=np.zeros((state_channel[1],patch_size[1]*patch_size[1]*fa[1]))
         
        # self.lca=lca 
        self.lca_f=lca_f
        self.lca_p=lca_p
        
        self.ps=periphery_patch_num
        self.dn=dict_num
        self.fs=fovea_patch_num
        
        self.fovea_size = fovea_size
        self.periphery_size = periphery_size
        self.fovea_pos = np.array([9,9])
        
        self.bug_solve=0
        self.bug = 0
        
        self.new_size = int((18+fovea_patch_num)*patch_size[0])
        self.rdd = int((self.new_size-fovea_size)/4*patch_size[0]) #扩大的范围
        
        self.start=time.time()
        self.end=time.time()
        
        self.on_target2=0
        self.on_target4=0
        self.on_target=0
        
        
    def get_state(self,obs,fovea_pos):
        
        
        t_u=10
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        
        
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
        a_state_p = tlca.run_lca(frame_state_p.t(), self.lca_p, t_u, self.fa[1], self.device)
        a_state_p = torch.reshape(a_state_p,(self.dn[1],self.ps,self.ps))

        a_state_p=torch.unsqueeze(a_state_p,0).cpu()
        
        return a_state_f,a_state_p  
    


    def move_fovea(self,action_fovea,obs):
        
        
        action1 = int(action_fovea[0,0]-2)
        action2 = int(action_fovea[0,1]-2)
        
        
        if action1 >= 0:
            self.fovea_pos[0]=min(18,self.fovea_pos[0]+action1)
        else:
            self.fovea_pos[0]=max(0,self.fovea_pos[0]+action1)
                
        if action2 >= 0:
            self.fovea_pos[1]=min(18,self.fovea_pos[1]+action2)
        else:
            self.fovea_pos[1]=max(0,self.fovea_pos[1]+action2)
        
        
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,self.fovea_size)
        
        frame[frame<3]=0
        
        if  frame.sum()<1:
            if self.bug_solve >1:
                self.fovea_pos=np.array([9,9])
            
            self.bug_solve += 1
            self.bug+=1
            return 
        
        self.bug_solve = 0
        
        
        return 
        
    
    def train(self):

        for episode in range(0,self.epoch):
            
            obs = self.env.reset()
            state_f,state_p = self.get_state(obs,self.fovea_pos)
            episode_reward = 0.0
            # episode_reward_ = 0.0
            action_fovea_before=torch.tensor([[2,2]],dtype=int)
            fovea_pos=torch.tensor([self.fovea_pos],dtype=int)
            
            for t in count():  
                                             
                action_fovea1, action_fovea2, action_play = self.agent_play.select_action(state_f,state_p,action_fovea_before,fovea_pos)
                
                action_fovea = torch.cat([action_fovea1,action_fovea2],dim=1).cpu()
                
                obs,reward,done,info = self.env.step(action_play)
                self.move_fovea(action_fovea,obs)
                next_fovea_pos=torch.tensor([self.fovea_pos],dtype=int)
                
                episode_reward += reward
                
                if not done:
                    next_state_f, next_state_p = self.get_state(obs,self.fovea_pos)
                else:
                    next_state_f = None
                    next_state_p = None
                    next_fovea_pos = None
                    self.frame_state_f=np.zeros((self.sc[0],self.i_s[0]*self.i_s[0]*self.fa[0]))
                    self.frame_state_p=np.zeros((self.sc[1],self.i_s[1]*self.i_s[1]*self.fa[1]))
                    self.rewardlist.append(episode_reward)
                
                # 将四元组存到memory中
                '''
                state: batch_size channel h w    size: batch_size * 2
                action_before
                action: size: batch_size * 1
                next_state: batch_size channel h w    size: batch_size * 4
                reward: size: batch_size * 1                
                '''
                self.agent_play.memory_buffer.push(state_f, state_p, fovea_pos,action_fovea_before,action_fovea, action_play.cpu(), next_state_f, next_state_p,next_fovea_pos, reward) # 里面的数据都是Tensor
                
                state_f = next_state_f
                state_p = next_state_p
                action_fovea_before=action_fovea
                fovea_pos=next_fovea_pos
                # 经验池满了之后开始学习
                if self.agent_play.stepdone > INITIAL_MEMORY:
                    self.agent_play.learn()
                    if self.agent_play.stepdone % TARGET_UPDATE == 0:
                        self.agent_play.target_DQN.load_state_dict(self.agent_play.DQN.state_dict())
                
                if done:
                    break
                
            if episode % 20 == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print (localtime)
                self.end=time.time()
                print('total time',self.end-self.start)
                torch.save(self.agent_play.DQN.state_dict(), MODEL_STORE_PATH + '/' + "play_model/{}_episode{}.pt".format(modelname, episode))
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {} '.format(self.agent_play.stepdone, episode, t, round(episode_reward,1)))#, round(accurate_rate,2)
                self.start=time.time()
                hdffile=h5py.File('rewardlist.h5','w')
                hdffile.create_dataset('data',data=self.rewardlist)
                hdffile.close()
                
            
            self.bug=0    
            
            self.env.close()
        return




class Evaluator():
    def __init__(self, env,device, agent_play,state_channel,patch_size,
                 fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,dict_num, 
                 fovea_size, periphery_size, nepisode):
        
        self.device = device
        self.env = env
        self.n_episode = nepisode
        self.agent_play = agent_play
        
        # self.losslist = []
        self.rewardlist = []
        self.sc=state_channel
        self.i_s=patch_size
        self.fa=fa
        
        self.q=np.zeros((1))
        self.score=np.zeros((1))
        self.input_size=patch_size[0]*patch_size[0]*fa[0]
       
        self.frame_state_f=np.zeros((state_channel[0],patch_size[0]*patch_size[0]*fa[0]))
        self.frame_state_p=np.zeros((state_channel[1],patch_size[1]*patch_size[1]*fa[1]))
         
        self.lca_f=lca_f
        self.lca_p=lca_p
        
        self.ps=periphery_patch_num
        self.dn=dict_num
        self.fs=fovea_patch_num
        
        self.fovea_size = fovea_size
        self.periphery_size = periphery_size
        
        self.bug_solve=0
        self.bug = 0
        
        self.new_size = int((18+fovea_patch_num)*patch_size[0])
        self.rdd = int((self.new_size-fovea_size)/4*patch_size[0]) #扩大的范围
        
        self.start=time.time()
        self.end=time.time()
        self.fovea_pos = np.array([9,9])
        
        
    def get_state(self,obs,fovea_pos):
        
        
        t_u=10
        
        y0=fovea_pos[0]
        x0=fovea_pos[1]
        
        obs=obs.astype(float)
        
        
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
        a_state_p = tlca.run_lca(frame_state_p.t(), self.lca_p, t_u, self.fa[1], self.device)
        a_state_p = torch.reshape(a_state_p,(self.dn[1],self.ps,self.ps))

        a_state_p=torch.unsqueeze(a_state_p,0).cpu()
        
        return a_state_f,a_state_p  
    
        


    def move_fovea(self,action_fovea,obs):
        
        
        action1 = int(action_fovea[0,0]-2)
        action2 = int(action_fovea[0,1]-2)
        
        
        if action1 >= 0:
            self.fovea_pos[0]=min(18,self.fovea_pos[0]+action1)
        else:
            self.fovea_pos[0]=max(0,self.fovea_pos[0]+action1)
                
        if action2 >= 0:
            self.fovea_pos[1]=min(18,self.fovea_pos[1]+action2)
        else:
            self.fovea_pos[1]=max(0,self.fovea_pos[1]+action2)
        
        
        
        obs=obs.astype(float)
        frame = tlca.resize_frame(obs,self.fovea_size)
        
        frame[frame<3]=0
        
        if  frame.sum()<1:
            if self.bug_solve >1:
                self.fovea_pos=np.array([9,9])
            
            self.bug_solve += 1
            self.bug+=1
            return 
        
        self.bug_solve = 0
        
        return 
        



        
    def evaluate(self):
        
        r_reward = np.zeros(self.n_episode)
        q_value = np.zeros(self.n_episode)

        for episode in range(0,self.n_episode):
            
            obs = self.env.reset()
            state_f,state_p = self.get_state(obs,self.fovea_pos)
            episode_reward = 0.0
            action_fovea_before=torch.tensor([[2,2]],dtype=int).to(self.device)
            fovea_pos=torch.tensor([self.fovea_pos],dtype=int)
            
            for t in count():  
                                             
                action_fovea1, action_fovea2, action_play,q = self.agent_play.select_action(state_f,state_p,action_fovea_before,fovea_pos)
                self.q=np.append(self.q,q)
                action_fovea = torch.cat([action_fovea1,action_fovea2],dim=1)
                
                
                obs,reward,done,info = self.env.step(action_play)
                self.move_fovea(action_fovea,obs)
                next_fovea_pos=torch.tensor([self.fovea_pos],dtype=int)
                if reward!=0:
                    self.score=np.append(self.score,reward)
                
                episode_reward += reward
                
                if not done:
                    next_state_f, next_state_p = self.get_state(obs,self.fovea_pos)
                else:
                    next_state_f = None
                    next_state_p = None
                    next_fovea_pos = None
                    self.frame_state_f=np.zeros((self.sc[0],self.i_s[0]*self.i_s[0]*self.fa[0]))
                    self.frame_state_p=np.zeros((self.sc[1],self.i_s[1]*self.i_s[1]*self.fa[1]))
                    if episode%50==0:
                        print('epoch='+str(episode))
                        print('reward='+str(episode_reward))
                
                
                state_f = next_state_f
                state_p = next_state_p
                action_fovea_before=action_fovea
                fovea_pos=next_fovea_pos
                
                
                if done:
                    break
                
            r_reward[episode]=episode_reward
            q_value[episode]=self.q[1:].mean()
            self.q=np.zeros((1))
            self.rewardlist.append(episode_reward)
            self.bug=0    
            
            self.env.close()
        return r_reward,q_value,self.score[1:101]





def train_dqn_play(periphery_size, fovea_size, fovea_patch_num, 
                    frame_analyze, patch_size, overcomplete,
                    threshold, device_name):
    
    
    env = gym.make("ALE/Pong-v5",obs_type='grayscale')
    action_space = env.action_space
    
    device = torch.device(device_name)
    
    dictfile=h5py.File('dict/dict'+str(periphery_size)+'.h5','r')
    dict_p=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_p=torch.from_numpy(dict_p).to(device)
    
    dictfile=h5py.File('dict/dict_tracking300.h5','r')
    dict_f=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_f=torch.from_numpy(dict_f).to(device)
    
    fa=np.zeros(2,dtype=int)
    fa[0]=frame_analyze
    fa[1]=frame_analyze
    
    
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
    
    
    
    track_model_path = MODEL_STORE_PATH + '/' + 'track_model/' + 'DQN_Pong_episode300.pt'
    track_model = torch.load(track_model_path)
    
    agent_play = dqn.DQN_agent_play( dict_num,fovea_patch_num, periphery_patch_num, device_name,action_space, memory_size=memorysize)
    agent_play.DQN.conv1.weight.data=track_model['conv11.weight'].to(device)
    agent_play.DQN.conv1.bias.data=track_model['conv11.bias'].to(device)
    agent_play.DQN.conv2.weight.data=track_model['conv21.weight'].to(device)
    agent_play.DQN.conv2.bias.data=track_model['conv21.bias'].to(device)
    agent_play.DQN.head1.weight.data=track_model['head1.weight'].to(device)
    agent_play.DQN.head1.bias.data=track_model['head1.bias'].to(device)
    agent_play.DQN.head2.weight.data=track_model['head2.weight'].to(device)
    agent_play.DQN.head2.bias.data=track_model['head2.bias'].to(device)
    
    
    
    lca_f=tlca.TLCA(input_size[0],dict_num[0],threshold,dict_f,state_channel[0],device)
    lca_p=tlca.TLCA(input_size[1],dict_num[1],threshold,dict_p,state_channel[1],device)
    
    trainer = Trainer(env,device, agent_play,state_channel,patch_size,
                      fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,
                      dict_num, fovea_size, periphery_size)
    trainer.train()





def evaluate_dqn_play(periphery_size, fovea_size, fovea_patch_num, 
                    frame_analyze, patch_size, overcomplete,
                    threshold, device_name):
    
    
    env = gym.make("ALE/Pong-v5",obs_type='grayscale')
    action_space = env.action_space
    
    device = torch.device(device_name)
    
    dictfile=h5py.File('dict/dict'+str(periphery_size)+'.h5','r')
    dict_p=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_p=torch.from_numpy(dict_p).to(device)
    
    dictfile=h5py.File('dict/dict_tracking300.h5','r')
    dict_f=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_f=torch.from_numpy(dict_f).to(device)
    
    fa=np.zeros(2,dtype=int)
    fa[0]=frame_analyze
    fa[1]=frame_analyze
    
    
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
    
    
    lca_f=tlca.TLCA(input_size[0],dict_num[0],threshold,dict_f,state_channel[0],device)
    lca_p=tlca.TLCA(input_size[1],dict_num[1],threshold,dict_p,state_channel[1],device)
    
    num_e = int((epoch-1)/40+1)
    reward_record=np.zeros((num_e,n_episode))
    q_record=np.zeros((num_e,n_episode))
    score_record=np.zeros((num_e,100))
    
    start=time.time()
    end=time.time()
    
    for i in range(0,num_e):
        
        i_str=str(i*40) #
        
        model_path_play = MODEL_STORE_PATH + '/' + 'play_model/' + 'DQN_Pong_episode'+i_str+'.pt'
        agent_play = dqn.DQN_agent_play_evaluate(dict_num,fovea_patch_num, periphery_patch_num, device_name, model_path_play,action_space, memory_size=memorysize)
    
        evaluator = Evaluator(env,device, agent_play,state_channel,patch_size,
                              fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,dict_num, 
                              fovea_size, periphery_size,n_episode)
        
        reward_record[i,:],q_record[i,:],score_record[i,:] =evaluator.evaluate()
        
        
        localtime = time.asctime( time.localtime(time.time()) )
        print (localtime)
        end=time.time()
        print('total time',end-start)
        print('episode='+i_str+' is done')
        print('average reward='+str(reward_record[i,:].mean()))
        print('average q='+str(q_record[i,:].mean()))
        start=time.time()
    
    
        hdffile=h5py.File('reward_play_'+str(periphery_size)+'_'+str(fovea_patch_num)+'_'+str(frame_analyze)+'.h5','w')
        hdffile.create_dataset('reward',data=reward_record)
        hdffile.create_dataset('q',data=q_record)
        hdffile.create_dataset('score',data=score_record)
        hdffile.close()






def evaluate_dqn_play_best(periphery_size, fovea_size, fovea_patch_num, 
                    frame_analyze, patch_size, overcomplete,
                    threshold, device_name):
    
    dictfile=h5py.File('reward_play_'+str(periphery_size)+'_'+str(fovea_patch_num)+'_'+str(frame_analyze)+'.h5','r')
    reward=np.array([dictfile['reward']])[0,:,:]
    dictfile.close()
    
    reward_mean=np.mean(reward,axis=1)
    i_str=str(40*int(np.where(reward_mean==reward_mean.max())[0]))
    
    env = gym.make("ALE/Pong-v5",obs_type='grayscale')
    action_space = env.action_space
    
    device = torch.device(device_name)
    
    dictfile=h5py.File('dict/dict'+str(periphery_size)+'.h5','r')
    dict_p=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_p=torch.from_numpy(dict_p).to(device)
    
    dictfile=h5py.File('dict/dict_tracking300.h5','r')
    dict_f=np.array([dictfile['dictionary']])[0,:,:]
    dictfile.close()
    dict_f=torch.from_numpy(dict_f).to(device)
    
    fa=np.zeros(2,dtype=int)
    fa[0]=frame_analyze
    fa[1]=frame_analyze
    
    
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
    
    
    lca_f=tlca.TLCA(input_size[0],dict_num[0],threshold,dict_f,state_channel[0],device)
    lca_p=tlca.TLCA(input_size[1],dict_num[1],threshold,dict_p,state_channel[1],device)
    
    reward_record=np.zeros(n_episode_final)
    
    start=time.time()
    end=time.time()
    
    
    model_path_play = MODEL_STORE_PATH + '/' + 'play_model/' + 'DQN_Pong_episode'+i_str+'.pt'
    agent_play = dqn.DQN_agent_play_evaluate(dict_num,fovea_patch_num, periphery_patch_num, device_name, model_path_play,action_space, memory_size=memorysize)

    evaluator = Evaluator(env,device, agent_play,state_channel,patch_size,
                          fa,lca_f,lca_p,periphery_patch_num,fovea_patch_num,dict_num, 
                          fovea_size, periphery_size, n_episode_final)
    
    reward_record,q,score =evaluator.evaluate()
    
    
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    end=time.time()
    print('total time',end-start)
    print('episode='+i_str+' is done')
    print('average reward='+str(reward_record.mean()))
    start=time.time()


    hdffile=h5py.File('reward_best_'+str(periphery_size)+'_'+str(fovea_patch_num)+'_'+str(frame_analyze)+'.h5','w')
    hdffile.create_dataset('reward',data=reward_record)
    hdffile.close()

