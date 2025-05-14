# ECE4990: Deep Learning
Aman Adhikari & Josiah Wiggins 
California State Polytechnic University, Pomona
Dr. Sedat Ozer
## This is a Fork and expansion the work of rpchen0128. 



# How to run 

1. Create two new folders named __dict__ and __model__ (for single-resolution) or __track_model__ (for multi-resolution) to save LCA dictionay and DQN weights

2. Run main.py to train LCA and LCA-DQN. 

3. Results are saved in dict.h5 (LCA dictionary),  DQN_Pong_episode.pt (LCA-DQN weights), reward.h5 (evaluation result)



# Reference
1. https://github.com/libenfan/DQN_pong
2. Rozell, C.J., et al., *Sparse coding via thresholding and local competition in neural circuits*. Neural Computation, 2008. 20(10)
3. Yogeswaran, A., P. Payeur, and Aaai. *Leveraging Saccades to Learn Smooth Pursuit: A Self-Organizing Motion Tracking Model Using Restricted Boltzmann Machines.* in 31st AAAI Conference on Artificial Intelligence. 2017
4. Chen R, Kunde GJ, Tao L and Sornborger AT. *Foveal vision reduces neural resources in agent-based game learning.* Frontiers in Neuroscience, 2025, 19:1547264. 
