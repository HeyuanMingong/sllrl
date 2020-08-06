# Scalable Lifelong Reinforcement Learning

This repo contains code accompaning the manuscript: [Zhi Wang, Chunlin Chen, and Daoyi Dong, "A Dirichlet Mixture of Robust Task Models for Scalable Lifelong Reinforcement Learning", submitted.]()
It contains code for running the lifelong learning tasks, including 2D navigation, Reacher, and Hopper domains.

### Dependencies
This code requires the following:
* python 3.5+
* pytorch 0.4+
* gym
* MuJoCo license

### Data
* For the 2D navigation domains, data is generated from `envs/navigation.py`
* For the Hopper/HalfCheetah/Ant Mujoco domains, the modified Mujoco enviornments are in `envs/mujoco/*`

### Usage 
* For example, to run the code in the 2D navigation domain, just run the bash script `navigation.sh`, also see the usage instructions in the python scripts `main.py`.
* When getting the results in `output/*/*.npy` files, plot the results using `data_process.py`. For example, the results for `navigation.sh` and `hopper.sh` are as follow:

navigation | hopper
------------ | -------------
![experimental results for navigation_v3 domain](https://github.com/HeyuanMingong/sllrl/blob/master/exp/navi_rews.png) | ![experimental results for ant domain](https://github.com/HeyuanMingong/sllrl/blob/master/exp/hopper_rews.png)

Also, the results for other demo scripts are shown in `exp/*`

### Contact 
For safety reasons, the source code is coming soon.

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/sllrl/issues), or email to zhiwang@nju.edu.cn.
 
