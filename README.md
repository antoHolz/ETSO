# Event-Triggered Safe Bayesian Optimization
This repository contains code for the paper ["Event-Triggered Safe Bayesian Optimization on Quadcopters"](https://arxiv.org/abs/2312.08058) which has been accepted to the [6th Annual Learning for Dynamics and Control Conference 2024 (L4DC)](https://l4dc.web.ox.ac.uk/) in Oxford, UK.

![ETSO intuition](https://github.com/antoHolz/ETSO/blob/main/ETSO_intuition_method.png)

If you find our code or paper useful, please consider citing
```
@article{holzapfel2023event,
  title={Event-Triggered Safe {Bayesian} Optimization on Quadcopters},
  author={Holzapfel, Antonia and Brunzema, Paul and Trimpe, Sebastian},
  booktitle={Proceedings of the 6nd Conference on Learning for Dynamics and Control},
  year={2024},
  publisher={PMLR},
}
```
---

We propose a new algorithm, **Event-Triggered SafeOpt (ETSO)**, which adapts to changes online solely relying on the observed costs. At its core, ETSO uses an *event trigger* to detect significant deviations between observations and the current surrogate of the objective function (similar to [ET-GP-UCB](https://arxiv.org/abs/2208.10790)). ETSO starts by optimizing a the performance function with *SafeOpt*. When a significant change is detected, the algorithm reverts to a safe backup controller, calculates a new threshold J<sub>min,t</sub>, and exploration is restarted. In this way, safety is recovered and maintained across changes. 

We evaluated ETSO on quadcopter controller tuning, both in simulation and hardware experiments. ETSO outperforms state-of-the-art safe BO, achieving superior control performance over time while maintaining safety. For the hardware results, we also have a video available in https://youtu.be/nLmeO-fMIvg. 

![Header](https://github.com/antoHolz/ETSO/blob/main/ETSO_header_figure.png)

Our implementation uses [GPyTorch](https://gpytorch.ai), [SafeOpt](https://github.com/befelix/SafeOpt) and the [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) gym simulation environment. 

## Results and parameters

The code in this repository  is intended as a means to reproduce the results obtained in our paper. For the simulation, the parameters provided in the python files should produce the same results as we obtained. For the hardware experiments, this will depend on the exact hardware setup. In our setup we used a Bitcraze Crazyflie 2.1, a Vicon camera system consisting of 12 cameras, and a ground computer.


The folders "ETSO_simulation" and "ETSO_hardware" contain each the code for the experiments performed and its results.
To recreate our results, first install the necessary packages and then run the corresponding python files (with the necessary adjustments in the hardware case, if a different system is being used). 

Below you will find a list of the used hyperparameters (as are also found in the code):

![Hyperparam](https://github.com/antoHolz/ETSO/blob/main/Hyperparameters%20ETSO.PNG)
