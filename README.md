# detschedule
Determinantal scheduling in wireless networks (in MATLAB or Python).

We propose a new class of algorithms for randomly scheduling wireless network transmissions. The idea is to use (discrete) determinantal point processes (subsets) to randomly assign medium access to  various repulsive subsets of potential transmitters. This approach can be seen as a  natural  extension of (spatial) Aloha, which schedules transmissions independently. Under a general path loss model and Rayleigh fading, we show that, similarly to Aloha, they are also subject to elegant analysis of the coverage probabilities and transmission attempts (also known as local delay). This is mainly due to the explicit, determinantal form of the conditional (Palm) distribution and closed-form  expressions for the Laplace functional of determinantal processes. Interestingly, the derived performance characteristics of the network are amenable to various optimizations of the scheduling parameters, which are determinantal kernels, allowing the use of techniques developed for statistical  learning with determinantal processes. Well-established sampling algorithms for determinantal processes can be used to cope with implementation issues, which is is beyond the scope of this paper, but it creates paths for further research.

# Starting point
For a starting point, run the (self-contained) files DemoDetPoisson.m or DemoDetPoisson.py to simulate or sample a single determinantally-thinned Poisson point process. The determinantal simulation is also performed by the file funSimSimpleDPP.m/py, which requires the eigendecomposition of a L kernel matrix.

# Papers
Some of the code  was used to generate results for published papers.

## Coverage probability in wireless networks with determinantal scheduling

The paper:

* 2020 – Błaszczyszyn, Brochard, and Keeler, `Coverage probability in wireless networks with determinantal scheduling'.

We have implemented all our mathematical results in MATLAB and Python code, which is located in  the respective repositories:

https://github.com/hpaulkeeler/detschedule/detcov_matlab

https://github.com/hpaulkeeler/detschedule/detcov_python

The mathematical results for transmitter-and-receiver pair network are implemented in the file ProbCovPairsDet.m/py; also see funProbCovPairsDet.m/py. The  mathematical results for transmitter-or-receiver network are implemented in the file ProbCovTXRXDet.m/py; also see funProbCovTXRXDet.m/py. These files typically require other files located in the repositories. 

## Adaptive determinantal scheduling with fairness in wireless networks

The paper:

* 2025 – Błaszczyszyn, and Keeler, `Adaptive determinantal scheduling with fairness in wireless networks'.

We have implemented all our mathematical results in MATLAB and Python code, which is located in  the respective repositories:

https://github.com/hpaulkeeler/detschedule/detcov_matlab

https://github.com/hpaulkeeler/detschedule/detcov_python


# Simulation results
We have also written the corresponding network simulations. The mathematical results agree excellently with simulations, which reminds us that determinantal point processes do not suffer from edge effects (induced by finite simulation windows). All mathematical and simulation results were obtained on a standard desktop machine, taking typically seconds to be executed. 


