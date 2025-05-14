import torch
import numpy as np
import tlca
import gymnasium as gym # gymnasium is the new version of gym
import h5py
import ale_py
from ale_py import ALEInterface
import matplotlib.pyplot as plt

ale = ALEInterface()

def check_paddle_location(steps, outputsize):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", obs_type='grayscale')
    observation, info = env.reset(options={"return_info": True})
    video=np.zeros((steps,outputsize,outputsize))
    
    for i in range(steps):
        action = env.action_space.sample()
        observation, rewards, done, truncated, info = env.step(action)
        done = done or truncated
        observation=observation.astype(float)
        
        video[i]=tlca.resize_frame(observation, outputsize)
        
        if done:
            observation, info = env.reset(options={"return_info": True})
    env.close()
    
    tlca.video_save(video, 'video_check.avi')
    
    hdffile=h5py.File('video_check.h5','w') #_paddle
    hdffile.create_dataset('data',data=video) #
    hdffile.close()
    



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
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", obs_type='grayscale')
    observation, info = env.reset(options={"return_info": True})
    train_video=np.zeros((steps,outputsize,outputsize))
    
    for i in range(steps):
        action = env.action_space.sample()
        observation, rewards, done, truncated, info = env.step(action)
        done = done or truncated
        observation=observation.astype(float)
        
        train_video[i]=tlca.resize_frame(observation, outputsize)
        
        
        if done:
            observation, info = env.reset(options={"return_info": True})
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


def plot_loss_curve(losses, filename):
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LCA Training Loss Curve')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def visualize_dictionary(dictionary, patch_size, filename):
    num_elements = dictionary.shape[1]
    grid_size = int(np.ceil(np.sqrt(num_elements)))
    plt.figure(figsize=(10, 10))
    for i in range(num_elements):
        element_size = dictionary[:, i].size
        if element_size != patch_size * patch_size:
            print(f"Skipping element {i}: cannot reshape array of size {element_size} into shape ({patch_size}, {patch_size})")
            continue
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(dictionary[:, i].reshape(patch_size, patch_size), cmap='gray')
        plt.axis('off')
    plt.suptitle('Learned Dictionary Elements')
    plt.savefig(filename)
    plt.close()

def calculate_reconstruction_error(data, dictionary):
    reconstruction = dictionary @ np.linalg.pinv(dictionary) @ data.T
    error = np.mean((data.T - reconstruction) ** 2)
    return error


def preprocess_to_square(data):
    """
    Preprocess data to make it square by padding or truncating.
    If the data is already 2D, return it as is.
    """
    if len(data.shape) == 2:  # Already 2D
        return data

    size = data.size
    side_length = int(np.ceil(np.sqrt(size)))
    padded_data = np.zeros((side_length * side_length,))
    padded_data[:size] = data[:size]  # Truncate or pad with zeros
    return padded_data.reshape((side_length, side_length))


def visualize_basis_vectors(dictionary, patch_size, filename_prefix):
    """
    Visualize all dictionary basis vectors in a single grid.
    """
    num_elements = dictionary.shape[1]
    grid_size = int(np.ceil(np.sqrt(num_elements)))
    plt.figure(figsize=(10, 10))
    for i in range(num_elements):
        reshaped_element = preprocess_to_square(dictionary[:, i])
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(reshaped_element, cmap='gray')
        plt.axis('off')
    plt.suptitle('Dictionary Basis Vectors')
    plt.savefig(f'{filename_prefix}_basis_vectors.png')
    plt.close()


def visualize_feature_maps(input_data, dictionary, filename_prefix):
    """
    Generate and visualize feature maps by applying the dictionary to the input data.
    """
    feature_maps = dictionary.T @ input_data.T  # Compute feature maps
    num_features = feature_maps.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_features)))
    plt.figure(figsize=(10, 10))
    for i in range(num_features):
        reshaped_feature = preprocess_to_square(feature_maps[i, :])
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(reshaped_feature, cmap='hot')
        plt.axis('off')
    plt.suptitle('Feature Maps')
    plt.savefig(f'{filename_prefix}_feature_maps.png')
    plt.close()


def visualize_specific_activations(activations, feature_name, filename_prefix):
    """
    Visualize activation maps for specific features like the ball or paddle.
    """
    plt.figure(figsize=(10, 10))
    for idx, activation in enumerate(activations[:min(10, len(activations))]):  # Visualize up to 10 activations
        reshaped_activation = preprocess_to_square(activation)
        plt.subplot(5, 2, idx + 1)
        plt.imshow(reshaped_activation, aspect='auto', cmap='hot')
        plt.colorbar()
        plt.title(f'{feature_name} Activation Map {idx + 1}')
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_{feature_name}_activation_maps.png')
    plt.close()


def visualize_all(dictionary, activations, losses, synapse_values, patch_size, filename_prefix, input_data=None):
    """
    Visualize all relevant metrics including dictionary, activation maps, loss, synapse values, weights, and feature maps.
    """
    # Visualize dictionary basis vectors
    visualize_basis_vectors(dictionary, patch_size, filename_prefix)

    # Visualize feature maps if input data is provided
    if input_data is not None:
        visualize_feature_maps(input_data, dictionary, filename_prefix)

    # Visualize activation maps for specific features (e.g., ball)
    visualize_specific_activations(activations, 'ball', filename_prefix)

    # Visualize loss curve
    plot_loss_curve(losses, f'{filename_prefix}_loss_curve.png')

    # Visualize synapse values
    plt.figure()
    plt.plot(synapse_values, label='Synapse Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Synapse Values Growth')
    plt.legend()
    plt.savefig(f'{filename_prefix}_synapse_values.png')
    plt.close()

    # Visualize number of relevant patches
    num_relevant_patches = [np.count_nonzero(act) for act in activations]
    plt.figure()
    plt.plot(num_relevant_patches, label='Relevant Patches')
    plt.xlabel('Sample Index')
    plt.ylabel('Number of Relevant Patches')
    plt.title('Number of Relevant Patches')
    plt.legend()
    plt.savefig(f'{filename_prefix}_relevant_patches.png')
    plt.close()


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
    plot_loss_curve(losses, 'lca_loss_curve_paddle.png')
    visualize_dictionary(model.dictionary.cpu().numpy(), int(np.sqrt(input_size)), 'lca_dictionary_paddle.png')
    reconstruction_error = calculate_reconstruction_error(traindata_unique, model.dictionary.cpu().numpy())
    print(f'Reconstruction Error (Paddle): {reconstruction_error}')
    
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
    losses = []
    for ep in range(epoch):
        loss = tlca.train_model(model1, input_size, 1, num_sample, traindata_unique, device)
        losses.append(loss)
    
    plot_loss_curve(losses, 'lca_loss_curve_all.png')
    visualize_dictionary(model1.dictionary.cpu().numpy(), int(np.sqrt(input_size)), 'lca_dictionary_all.png')
    reconstruction_error = calculate_reconstruction_error(traindata_unique, model1.dictionary.cpu().numpy())
    print(f'Reconstruction Error (All): {reconstruction_error}')
    
    hdffile=h5py.File('dict/dict'+str(th)+'.h5','w') #
    hdffile.create_dataset('dictionary',data=model1.dictionary.cpu().numpy())
    hdffile.close()
    
    print('\n=================================')
    print('\nLCA has been trained on all data\n')
    print('=================================\n')



def test_LCA(patch_size, frame_analyze, outputsize, th, devicename):
    device = torch.device(devicename)
    hdffile = h5py.File('video_check.h5', 'r')
    test_video = np.array([hdffile['data']])[0, :, :]
    hdffile.close()
    testdata, testdata_paddle = video2patch(test_video, patch_size, frame_analyze, [0])

    a_last = None
    all_sample = testdata.shape[0]
    output_video = np.zeros((test_video.shape[0] - frame_analyze + 1, outputsize * frame_analyze, outputsize))
    test_output = np.zeros((outputsize * frame_analyze, outputsize))

    hdffile = h5py.File('dict/dict' + str(th) + '.h5', 'r')
    dictionary = np.array([hdffile['dictionary']])[0, :, :]
    dictionary1 = torch.from_numpy(dictionary).to(device)
    hdffile.close()

    input_size = dictionary.shape[0]
    dictionary_size = dictionary.shape[1]
    model = tlca.TLCA(input_size, dictionary_size, th, dictionary1, 1, device)

    patch_per_frame = int(outputsize * outputsize / patch_size / patch_size)

    activations = []
    for i in range(all_sample):
        video_piece = testdata[i]
        video_piece = (torch.from_numpy(video_piece).reshape((input_size, 1)).float())
        video_piece = video_piece.to(device)

        a_last = tlca.test(video_piece, model, frame_analyze, device)
        if isinstance(a_last, torch.Tensor):
            activations.append(a_last.cpu().numpy())
        else:
            activations.append(a_last)

        c = dictionary.dot(a_last)
        d = c.reshape((patch_size * frame_analyze, patch_size))

        x = int((i % patch_per_frame) / outputsize * patch_size)
        y = int(i % (outputsize / patch_size))

        for p in range(frame_analyze):
            test_output[p * outputsize + x * patch_size:p * outputsize + (x + 1) * patch_size,
                        y * patch_size:(y + 1) * patch_size] = d[p * patch_size:(p + 1) * patch_size, :]

        if (i + 1) % patch_per_frame == 0:
            frame_num = int(i / patch_per_frame)
            output_video[frame_num, :, :] = test_output
            test_output = np.zeros((outputsize * frame_analyze, outputsize))

    tlca.video_save(output_video, 'reconstruct_video_' + str(th) + '.avi')

    # Visualize activation maps
    plt.figure(figsize=(10, 10))
    for idx, activation in enumerate(activations[:min(10, len(activations))]):  # Visualize up to 10 activations
        plt.subplot(5, 2, idx + 1)
        plt.imshow(activation, aspect='auto', cmap='hot')
        plt.colorbar()
        plt.title(f'Activation Map {idx + 1}')
    plt.tight_layout()
    plt.savefig('activation_maps.png')
    plt.close()

    # Visualize sparse activations
    sparse_activations = [np.count_nonzero(act) / act.size for act in activations]
    plt.figure()
    plt.plot(sparse_activations, label='Sparsity')
    plt.xlabel('Sample Index')
    plt.ylabel('Sparsity Ratio')
    plt.title('Sparse Activations')
    plt.legend()
    plt.savefig('sparse_activations.png')
    plt.close()

    # Visualize dictionary weights
    plt.figure(figsize=(10, 10))
    for i in range(dictionary_size):
        element_size = dictionary[:, i].size
        if element_size != patch_size * patch_size:
            print(f"Skipping dictionary element {i}: cannot reshape array of size {element_size} into shape ({patch_size}, {patch_size})")
            continue
        plt.subplot(int(np.sqrt(dictionary_size)), int(np.sqrt(dictionary_size)), i + 1)
        plt.imshow(dictionary[:, i].reshape(patch_size, patch_size), cmap='gray')
        plt.axis('off')
    plt.suptitle('Dictionary Weights')
    plt.savefig('dictionary_weights.png')
    plt.close()

    # Visualize atoms
    plt.figure()
    plt.plot(dictionary.flatten(), label='Atoms')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Dictionary Atoms')
    plt.legend()
    plt.savefig('dictionary_atoms.png')
    plt.close()


