# Observation Utility Functions


## Tree Observation
The flatland built-in tree observation creates a tree of depth $n$ which encodes 12 features per explored branch: 
1. Distance to own target if it lies on the explored branch
2. Distance to other agent's target if detected
3. Distance to other agents if detected
4. Distance to predicted conflict (based on a predictor)
5. Distance to an unusable switch
6. Distance to next node
7. Distance remaining from node to the agent's target if this path is chosen
8. Number of agents going in the same direction on the path to the node
9. Number of agents going in the opposite direction on the path to the node
10. Time steps that a malfunctioning / blocking agent will remain blocked
11. Slowest observed speed of an agent in the same direction
12. Number of agents ready to depart but not yet active

The ``obs_utils.py`` script contains functions to unwrap the observations delivered by the flatland environment, unpacking nested nodes into a fixed-size tensor. The number of nodes are calculated as a geometric progression, with four possible branchings per node. The total number of nodes is dependent on the maximum depth of the observation and calculated as:
$$
\frac{(4^{\text{max\_depth}}-1)}{3}
$$


## Normalisation
To support learning, a ``Normalisation`` class is implemented which maintains a mean $\mu$ and standard deviation $\sigma$ of either a single value or a series of values (i.e., features) over all observed samples. It normalises observations (1) and allows for clipping (2):
$$
\begin{align}
\tilde{x} &= \frac{x-\mu}{\sigma+\epsilon} \\ 
\tilde{x} &\leftarrow \text{clip}(\tilde{x},-c, c) \qquad\qquad (\text{e.g., } c=5)
\end{align}
$$

Where $\epsilon$ is a small constant that prevents division by zero. To maintain $\mu$ and $\sigma$ over all gathered samples without saving them all, the ``RunningMeanStd`` class (or ``FeatureRunningMeanStd for tensors with individual features) uses Welford's algorithm to calculate the new $\mu$ and $\sigma$ with either single added samples or entire batches of samples. The formulae are relatively simple for single samples:

$$
\begin{align}
\mu_N &= \frac{1}{N}\sum_{i=1}^Nx_i \\
\sigma^2_N&=\frac{1}{N}\sum_{i=1}^N(x_i-\mu_N)^2
\end{align}
$$

For batches of samples, a more complicated process is followed: 

$$
\begin{align}
n_{AB} &= n_A + n_B \\
\delta &= \bar{x}_B - \bar{x}_A \\
\bar{x}_{AB} &= \bar{x}_A+\delta\cdot\frac{n_B}{n_{AB}} \\
M_{2,AB} &= M_{2,A} + M_{2,B} +\delta^2\cdot\frac{n_An_B}{n_{AB}} \\
\sigma_{AB} &= \frac{M_{2,AB}}{(n_AB + \epsilon)}   
\end{align}
$$


### Flatland Normalisation
