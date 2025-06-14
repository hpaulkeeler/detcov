# detschedule
Determinantal scheduling in wireless networks (in MATLAB or Python).

We propose a new class of algorithms for randomly scheduling wireless network. The idea is to use (discrete) determinantal point processes (subsets) to randomly schedule medium access for repulsive subsets of potential transmitters and receivers. This approach can be seen as a  natural  extension of (spatial) Aloha, which schedules transmissions independently. 

# Starting point
For a starting point, run the (self-contained) files DemoDetPoisson.m or DemoDetPoisson.py to simulate or sample a single determinantally-thinned Poisson point process. The determinantal simulation is also performed by the file funSimSimpleDPP.m/py, which requires the eigendecomposition of a L kernel matrix.

# Papers
Some of the code  was used to generate results for published papers.

## Adaptive determinantal scheduling with fairness in wireless networks

The paper:

* 2025 – Błaszczyszyn, and Keeler, _Adaptive determinantal scheduling with fairness in wireless networks_.

We have implemented all our mathematical results in MATLAB and Python code, which is located in  the respective repositories:

https://github.com/hpaulkeeler/detschedule/tree/master/detfairness_matlab

https://github.com/hpaulkeeler/detschedule/tree/master/detfairness_python

We formulate the determinantal scheduling problem with an utility function representing fairness. We then recast this formulation as a convex optimization problem over a certain class of determinantal point processes called $L$-ensembles, which are particularly suited for statistical and numerical treatments. These determinantal processes, which have already proven valuable in subset learning, offer an attractive approach to network resource scheduling and allocating. We demonstrate the suitability of determinantal processes for network models based on the signal-to-interference-plus-noise ratio (SINR). Our results highlight the potential of determinantal scheduling coupled with fairness, especially compared to Aloha scheduling. This work bridges recent advances in machine learning with wireless communications, providing a mathematically elegant and computationally tractable approach to network scheduling.

If you use this code in published research, please cite the above paper.

## Coverage probability in wireless networks with determinantal scheduling

The paper:

* 2020 – Błaszczyszyn, Brochard, and Keeler, _Coverage probability in wireless networks with determinantal scheduling_.

A copy of the above paper can be found here:

https://arxiv.org/abs/2006.05038

We have implemented all our mathematical results in MATLAB and Python code, which is located in  the respective repositories:

https://github.com/hpaulkeeler/detschedule/tree/master/detcov_matlab

https://github.com/hpaulkeeler/detschedule/tree/master/detcov_python

Under a general path loss model and Rayleigh fading, we show that, similarly to Aloha, the network model is also subject to elegant analysis of the coverage probabilities and transmission attempts (also known as local delay). This is mainly due to the explicit, determinantal form of the conditional (Palm) distribution and closed-form  expressions for the Laplace functional of determinantal processes. Interestingly, the derived performance characteristics of the network are amenable to various optimizations of the scheduling parameters, which are determinantal kernels, allowing the use of techniques developed for statistical  learning with determinantal processes. Well-established sampling algorithms for determinantal processes can be used to cope with implementation issues, which is is beyond the scope of this paper, but it creates paths for further research.

The mathematical results for transmitter-and-receiver pair network are implemented in the file ProbCovPairsDet.m/py; also see funProbCovPairsDet.m/py. The  mathematical results for transmitter-or-receiver network are implemented in the file ProbCovTXRXDet.m/py; also see funProbCovTXRXDet.m/py. These files typically require other files located in the repositories. 

If you use this code in published research, please cite the above paper.

# Simulation results
We have also written the corresponding network simulations. The mathematical results agree excellently with simulations, which reminds us that determinantal point processes do not suffer from edge effects (induced by finite simulation windows). All mathematical and simulation results were obtained on a standard desktop machine, taking typically seconds to be executed. 


