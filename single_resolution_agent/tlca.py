import torch
import numpy as np
import time
# import skvideo.io
from skimage import transform



def resize_frame(frame,new_size):
    obs = frame[34:194,5:-5]
    frame=np.zeros((new_size,new_size))
    
    amplitude=0.5
        
    factorx=160/new_size
    factory=150/new_size
    
    ball_sizex=np.max((1,round(4/factorx)))
    ball_sizey=np.max((1,round(2/factory)))
    paddle_sizey=np.max((1,round(4/factory)))
    
    paddle_ai=np.where((obs[:,135]>140) & (obs[:,135]<150))
    
    if len(paddle_ai[0])>0:
        
        ai_x=np.min((round(paddle_ai[0].min()/factorx),new_size-1))
        ai_y=np.min((round(135/factory),new_size-1))
        ai_sizex=np.max((1,round((paddle_ai[0].max()-paddle_ai[0].min()+1)/factorx)))
        if ai_sizex+ai_x>new_size:
            ai_sizex=new_size-ai_x
            
        frame[ai_x:(ai_x+ai_sizex),ai_y:(ai_y+paddle_sizey)]=np.ones((ai_sizex,paddle_sizey))*amplitude
    
    
    
    paddle_atari=np.where((obs[:,11]>140) & (obs[:,11]<150))
    if len(paddle_atari[0])>0:
        
        atari_x=np.min((round(paddle_atari[0].min()/factorx),new_size-1))
        atari_y=np.min((round(11/factory),new_size-1))
       
        atari_sizex=np.max((1,round((paddle_atari[0].max()-paddle_atari[0].min()+1)/factorx)))
        if atari_sizex+atari_x>new_size:
                atari_sizex=new_size-atari_x
            
        
        frame[atari_x:(atari_x+atari_sizex),atari_y:(atari_y+paddle_sizey)]=np.ones((atari_sizex,paddle_sizey))*amplitude
    
    ball=np.where(obs>200)
    if len(ball[0])>0:
        ball_x=np.min([round(ball[0].min()/factorx),new_size-1])
        ball_y=np.min([round(ball[1].min()/factory),new_size-1])
        if ball_sizex+ball_x>new_size:
            ball_sizex=new_size-ball_x
        if ball_sizey+ball_y>new_size:
            ball_sizey=new_size-ball_y
        frame[ball_x:(ball_x+ball_sizex),ball_y:(ball_y+ball_sizey)]=np.ones((ball_sizex,ball_sizey))
    
    return frame*4

class TLCA:
    
    """
    input_size: size of input image series, shape of input=(input_size,1)
    dict_size: size of dictionary, shape of dictionary=(input_size,dictionary_size)
    dt: time interval between two timestep
    threshold: threshold of activation function
    lr: learning rate of dictionary
    tau: time constant of TLCA
    
    dictionary: dictionary of TLCA, shape=(input_size,dictionary_size), tensor
    lateral_weight: lateral inhibitory weights of TLCA, shape=(dictionary_size,dictionary_size), tensor
    input_weight: dot product of input and dictionary, shape=(dictionary_size,1)
    """
    
    def __init__(self,patch_size,dictionary_size,
                 threshold,dictionary,num_patch,device):
        
        self.input_size=patch_size
        self.dict_size=dictionary_size
        self.dt=torch.tensor([0.01]).to(device)
        self.threshold=torch.tensor([threshold]).to(device)
        self.lr=0.002 
        self.tau=0.1
        self.mode=0
        
        self.dictionary=dictionary
        self.lateral_weight=torch.matmul(dictionary.t(),dictionary)-torch.eye(dictionary_size).to(device)
        self.input_weight=None 
        self.num_patch = num_patch
        
        self.u=torch.zeros(self.dict_size,self.num_patch).to(device) 
        self.a=torch.zeros(self.dict_size,self.num_patch).to(device)
        self.device=device
        
        
    def initialize(self, inputdata):
        
        
        self.input_weight=torch.matmul(self.dictionary.t(),inputdata)



    #update u, a, dictionary
    def update_au(self,inputdata):
        
        #update u
        self.u += (self.input_weight-1*torch.matmul(self.lateral_weight,self.a) #0.001
                           -self.u)/self.tau*self.dt
        self.u = torch.maximum(self.u, torch.tensor([0],dtype=self.u.dtype).to(self.device)) 
        #update a 
        self.a = torch.where(self.u<self.threshold,torch.tensor(0,dtype=self.u.dtype).to(self.device),self.u) #-self.threshold
   
        
    def update_dict(self,inputdata):
       
        #update & normalize dictionary
        delta_s=inputdata-torch.matmul(self.dictionary,self.a)
        
        """only update ball dictionary"""
        if self.mode==0:
            self.dictionary += self.lr*delta_s*self.a.t()
        else:
            a_ball=torch.zeros(self.a.shape).to(self.device)
            a_ball[:int(self.a.shape[0]/2),:]=self.a[:int(self.a.shape[0]/2),:]
            self.dictionary += self.lr*delta_s*a_ball.t()
        
        
        """update paddle dictionary"""
        
        self.dictionary = self.dictionary / torch.linalg.norm(self.dictionary,axis=0)
        
        
        #update lateral weight
        self.lateral_weight=torch.matmul(self.dictionary.t(),self.dictionary)-torch.eye(self.dict_size).to(self.device)
        
        #make all lateral weights positive
        self.lateral_weight=torch.maximum(self.lateral_weight, torch.tensor(0,dtype=self.u.dtype).to(self.device)) #-self.lateral_weight
        
        #update input_weight
        self.input_weight=torch.matmul(self.dictionary.t(),inputdata)
    
    def error(self,inputdata):
        
        delta = inputdata - torch.matmul(self.dictionary,self.a)
        delta_norm=torch.linalg.norm(delta,axis=0)
        
        return delta_norm


        

def run_lca(video_p,model,t_dict_update,frame_analyze, patch_size,device):
    a_last=np.zeros((model.dict_size,1))
    
    if np.max(video_p.cpu().numpy())!=0:
        model.initialize(video_p)
        for i in range(t_dict_update):
            model.update_au(video_p)
    
    a_last=model.a
    
    model.u=torch.zeros(model.dict_size,model.num_patch).to(device)
    model.a=torch.zeros(model.dict_size,model.num_patch).to(device)
    
    return  a_last





def video2image_piece(videodata,patch_size, frame_analyze,memory=None):

    sizex=int(videodata.shape[1]/patch_size)
    sizey=int(videodata.shape[2]/patch_size)
    
    traindata = np.zeros((int(sizex*sizey*(videodata.shape[0]-frame_analyze+1)),patch_size*patch_size*frame_analyze))
    
    for i in range(videodata.shape[0]-frame_analyze+1):
        for j in range(sizex):
            for k in range(sizey):
                id_image=i*sizex*sizey+j*sizey+k
                traindata[id_image,:] = videodata[i:i+frame_analyze,j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size].reshape((1,patch_size*patch_size*frame_analyze))
    
    
    memory=np.hstack((memory[:,patch_size**2:],traindata)) 
    return memory






def video_save(v, output_name):
    vid_out=skvideo.io.FFmpegWriter(output_name,inputdict={'-r': '20'})
    for i in range(v.shape[0]):
        new_v = v[i,:,:]*255
        vid_out.writeFrame(new_v)
    vid_out.close()






def train_model(model,input_size,epoch,
                num_sample,input_v,device,t_dict_update=10):
    order_sample=np.linspace(0,num_sample-1,num_sample,dtype=int)
    
    for t in range(epoch):
        #start time
        time_start=time.time()
        print('epoch='+str(t+1))
        e_epoch=0
        np.random.shuffle(order_sample)
        for j in order_sample:
            
            video_piece=input_v[j]
            if np.max(video_piece)==0:
                continue
            inputdata=(torch.from_numpy(video_piece).reshape((input_size,1)).float())   
            inputdata=inputdata.to(device)
        
            model.initialize(inputdata)
     
            for i in range(t_dict_update):
                model.update_au(inputdata)
                if (i+1) % t_dict_update == 0 :
                    e_epoch+=model.error(inputdata)
                    model.update_dict(inputdata)
                    
                    model.u=torch.zeros(model.dict_size,1).to(device)
                    model.a=torch.zeros(model.dict_size,1).to(device)
            
        r_e=e_epoch/(num_sample)
        
        time_end=time.time()
        localtime = time.asctime( time.localtime(time.time()) )
        print (localtime)
        print('total time',time_end-time_start)
        print('cost =', r_e.cpu().numpy()[0])
        
    return model



def create_dictionary(input_size,dictionary_size,device):
    dictionary=torch.rand(input_size,dictionary_size).to(device)-0.5
    return dictionary/torch.linalg.norm(dictionary,axis=0).to(device)


def test(video_p,model,frame_analyze,device,t_dict_update=10):
    a_last=np.zeros((model.dict_size,1))
    
    if np.max(video_p.cpu().numpy())!=0:
        model.initialize(video_p)
        for i in range(t_dict_update):
            model.update_au(video_p)
    a_last=model.a.cpu().numpy()
    
    model.u=torch.zeros(model.dict_size,1).to(device)
    model.a=torch.zeros(model.dict_size,1).to(device)
    
    return a_last
