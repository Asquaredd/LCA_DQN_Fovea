import torch
import numpy as np
import tlca
import gym
import h5py

def check_paddle_location(steps, outputsize):
    env = gym.make("ALE/Pong-v5", obs_type='grayscale')
    observation= env.reset()
    video=np.zeros((steps,outputsize,outputsize))
    
    for i in range(steps):
        action = env.action_space.sample()
        observation, rewards, done, info = env.step(action)
        observation=observation.astype(float)
        
        video[i]=tlca.resize_frame(observation, outputsize)
        
        if done:
            observation, info = env.reset(return_info=True)
    env.close()
    
    tlca.video_save(video, 'video_check.avi')
    
    # hdffile=h5py.File('video_check.h5','w') #_paddle
    # hdffile.create_dataset('data',data=video) #
    # hdffile.close()
    



def video2patch(videodata, patch_size, frame_analyze, x_paddle):

    sizex=int(videodata.shape[1]/patch_size)
    sizey=int(videodata.shape[2]/patch_size)
    
    traindata = np.zeros((int(sizex*sizey*(videodata.shape[0]-frame_analyze+1)),patch_size*patch_size*frame_analyze))
    traindata_paddle = np.zeros((int(sizex*sizey*(videodata.shape[0]-frame_analyze+1)),patch_size*patch_size*frame_analyze))
    
    for i in range(videodata.shape[0]-frame_analyze+1):
        for j in range(sizex):
            for k in range(sizey):
                id_image=i*sizex*sizey+j*sizey+k
                traindata[id_image,:] = videodata[i:i+frame_analyze,j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size].reshape((1,patch_size*patch_size*frame_analyze))
                
                if k == x_paddle[0]:
                    id_image2=i*sizey*2+j*2
                    traindata_paddle[id_image2,:] = videodata[i:i+frame_analyze,j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size].reshape((1,patch_size*patch_size*frame_analyze))
                if k == (sizey-1-x_paddle[0]):
                    id_image2=i*sizey*2+j*2
                    traindata_paddle[id_image2,:] = videodata[i:i+frame_analyze,j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size].reshape((1,patch_size*patch_size*frame_analyze))
     
                if len(x_paddle)==2:
                    if k == x_paddle[1]:
                        id_image2=i*sizey*2+j*2
                        traindata_paddle[id_image2,:] = videodata[i:i+frame_analyze,j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size].reshape((1,patch_size*patch_size*frame_analyze))
                    if k == (sizey-1-x_paddle[1]):
                        id_image2=i*sizey*2+j*2
                        traindata_paddle[id_image2,:] = videodata[i:i+frame_analyze,j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size].reshape((1,patch_size*patch_size*frame_analyze))
     
                    
                
    return traindata,traindata_paddle



def create_training_set(steps, outputsize, frame_analyze, patch_size,x_paddle):
    env = gym.make("ALE/Pong-v5", obs_type='grayscale')
    observation= env.reset()
    train_video=np.zeros((steps,outputsize,outputsize))
    
    for i in range(steps):
        action = env.action_space.sample()
        observation, rewards, done, info = env.step(action)
        observation=observation.astype(float)
        
        train_video[i]=tlca.resize_frame(observation, outputsize)
        
        
        if done:
            observation, info = env.reset(return_info=True)
    env.close()
    
    traindata,traindata_paddle = video2patch(train_video, patch_size, frame_analyze, x_paddle)
    
    traindata_paddle_unique=np.unique(traindata_paddle,axis=0)

    hdffile=h5py.File('dict/data_paddle.h5','w')
    hdffile.create_dataset('data',data=traindata_paddle_unique)
    hdffile.close()

    traindata_unique=np.unique(traindata,axis=0)

    hdffile=h5py.File('dict/data.h5','w')
    hdffile.create_dataset('data',data=traindata_unique)
    hdffile.close()
    
    print('\n=================================')
    print('\nTraining set is ready\n')
    print('=================================\n')





def train_LCA(th,overcomplete,epoch,devicename):
    device=torch.device(devicename)
    #train paddle dictionary first
    hdffile=h5py.File('dict/data_paddle.h5','r')#
    traindata_unique=np.array([hdffile['data']])[0,:,:]
    hdffile.close()
    
    
    input_size=traindata_unique.shape[1]
    dictionary_size=int(input_size*overcomplete/2)
    dictionary=torch.rand(input_size,dictionary_size).to(device)-0.5
    
    dictionary1 = dictionary/torch.linalg.norm(dictionary,axis=0).to(device)
    num_sample=traindata_unique.shape[0]
    
    model = tlca.TLCA(input_size,dictionary_size,th,dictionary1,1,device)
    model = tlca.train_model(model,input_size, epoch, num_sample, traindata_unique,device)
    
    
    hdffile=h5py.File('dict/dict_paddle'+str(th)+'.h5','w') #
    hdffile.create_dataset('dictionary',data=model.dictionary.cpu().numpy())
    hdffile.close()
    
    print('\n=================================')
    print('\nLCA has been trained on paddle data\n')
    print('=================================\n')
    
    
    #train whole dictionary
    hdffile=h5py.File('dict/data.h5','r')#
    traindata_unique=np.array([hdffile['data']])[0,:,:]
    hdffile.close()
    
    
    input_size=traindata_unique.shape[1]
    dictionary_size1=int(input_size*overcomplete) 
    dictionary=torch.rand(input_size,dictionary_size1).to(device)-0.5
    
    hdffile=h5py.File('dict/dict_paddle'+str(th)+'.h5','r') #_paddle
    dictionary_paddle=np.array([hdffile['dictionary']])[0,:,:]
    dictionary_paddle=torch.from_numpy(dictionary_paddle).to(device) #
    hdffile.close()
    
    dictionary1 = dictionary/torch.linalg.norm(dictionary,axis=0).to(device)
    dictionary1[:,int(dictionary_size1/2):]=dictionary_paddle 
    num_sample=traindata_unique.shape[0]
    
    model1 = tlca.TLCA(input_size,dictionary_size1,th,dictionary1,1,device)
    model1.mode=1 #change dictionary update mode
    model1 = tlca.train_model(model1,input_size, epoch, num_sample, traindata_unique,device)
    
    
    hdffile=h5py.File('dict/dict'+str(th)+'.h5','w') #
    hdffile.create_dataset('dictionary',data=model1.dictionary.cpu().numpy())
    hdffile.close()
    
    print('\n=================================')
    print('\nLCA has been trained on all data\n')
    print('=================================\n')



def test_LCA(patch_size, frame_analyze, outputsize, th,devicename):
    device=torch.device(devicename)
    hdffile=h5py.File('video_check.h5','r')
    test_video=np.array([hdffile['data']])[0,:,:]
    hdffile.close()
    testdata,testdata_paddle=video2patch(test_video, patch_size, frame_analyze, 0)
    
    a_last=None    
    all_sample=testdata.shape[0]
    output_video=np.zeros((test_video.shape[0]-frame_analyze+1,outputsize*frame_analyze,outputsize))
    test_output=np.zeros((outputsize*frame_analyze,outputsize))
    
    hdffile=h5py.File('dict/dict'+str(th)+'.h5','r')
    dictionary = np.array([hdffile['dictionary']])[0,:,:]
    dictionary1=torch.from_numpy(dictionary).to(device) #
    hdffile.close()
    
    input_size = dictionary.shape[0]
    dictionary_size = dictionary.shape[1]
    model = tlca.TLCA(input_size,dictionary_size,th,dictionary1,1,device)
    
    patch_per_frame=int(outputsize*outputsize/patch_size/patch_size)

    for i in range(all_sample):#
        
        video_piece=testdata[i]
        
        video_piece=(torch.from_numpy(video_piece).reshape((input_size,1)).float())   
        video_piece=video_piece.to(device)
        
        a_last=tlca.test(video_piece, model, frame_analyze)
        
        c=dictionary.dot(a_last)
        d=c.reshape((patch_size*frame_analyze,patch_size))
       
        x=int((i%patch_per_frame)/outputsize*patch_size)
        y=int(i%(outputsize/patch_size))
        
        
        for p in range(frame_analyze):
            test_output[p*outputsize+x*patch_size:p*outputsize+(x+1)*patch_size,
                        y*patch_size:(y+1)*patch_size]=d[p*patch_size:(p+1)*patch_size,:]
        
        
        if (i+1)%patch_per_frame==0:
              frame_num=int(i/patch_per_frame)
              output_video[frame_num,:,:]=test_output
              test_output=np.zeros((outputsize*frame_analyze,outputsize))


    tlca.video_save(output_video, 'reconstruct_video_'+str(th)+'.avi')

    
