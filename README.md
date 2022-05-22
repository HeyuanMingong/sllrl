# Scalable Lifelong Reinforcement Learning

This repo contains code accompaning the manuscript: [Zhi Wang, Chunlin Chen, and Daoyi Dong, "A Dirichlet Mixture of Robust Task Models for Scalable Lifelong Reinforcement Learning", IEEE Transactions on Cybernetics, DOI: 10.1109/TCYB.2022.3170485, 2022.](https://ieeexplore.ieee.org/document/9777250)
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
* For example, to run the code in the 2D navigation domain, just run the bash script `navi_v1.sh`, also see the usage instructions in the python scripts `main_sllrl.py`.
* When getting the results in `output/*/*.npy` files, plot the results using `plot_results.py`. For example, the result for `navi_v1.sh` is:

<p align='center'>navigation</p>
![experimental results for navigation_v3 domain](https://github.com/HeyuanMingong/sllrl/blob/master/exp/comparison_navi1.png) 


### Contact 
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/sllrl/issues), or email to zhiwang@nju.edu.cn.
 
