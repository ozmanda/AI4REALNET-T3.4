# AI4REALNET-T3.4
Repository for experimental development of fully autonomous AI for the AI4REALNET project. The work focuses on the development of solutions for multi-agent systems, using the flatland simulation environment. 

## Installation
These packages were developed with python 3.8.10. For package versions see the ``requirements.txt``. In the .vscode folder, a ``launch.json`` and ``settings.json`` are available to run the different models and perform unittesting.

## Communication Baselines
Currently the repository contains the following baseline models upon which further development should be done: 

### IC3NET 
The [original code](https://github.com/IC3Net/IC3Net) from the [paper](https://arxiv.org/abs/1812.09755) *Learning When to Communicate at Scale in Multiagent Cooperative and Competitive Tasks* by Singh et al., published at the ICLR in 2019, has been adapted to the flatland environment and integrated into the T3.4 codebase. 

### JBR_HSE
The JBR_HSE group developed the best RL solution to the [Flatland challenge at NeurIPS 2020](https://arxiv.org/abs/2103.16511). The [solution](https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution/tree/master) was already developed specifically for flatland and hence only needed to be slightly adapted for integration into the T3.4 codebase. 


# Interpretable Communication
The interpretable communication protocols and model architecture are currently WIPs and not functional. 


