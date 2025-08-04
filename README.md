# Hyperkriging for Surrogate Modeling
**Author**: Atticus Rex, Ph.D. student (Georgia institute of Technology)
**Supervisors:** David Peterson, Ph.D. (Wright-Patterson AFRL), Elizabeth Qian, Ph.D. (Georgia Institute of Technology)


---

This repository contains a condensed version of the codebase developed during my summer as a graduate research intern at Wright-Patterson Air Force Research Lab in 2025. Hyperkriging is a multi-fidelity regression method I developed working with Dr. Qian as a part of my graduate research. There are three important files in this repository: 
- `util.py` - this contains all the necessary utility functions to perform Hyperkriging on a multi-fidelity dataset. This includes a simple Gaussian Process regression implementation using `jax`, the Hyperkriging implementation as well as a Radial Basis Function kernel covariance function, and the ADAM optimization routine. 
- `DataAcquisition.py` - this file contains the code used to collect Laminar Flame Speed (LFS) data using the `cantera` python package. NOTE: This requires that the chemical mechanisms are contained within a directory called `mechanisms` which is not currently the case because of distribution restrictions. The main script collects equivalence ratio sweeps at a user-defined set of initial temperatures and saves each in its own `pickle` file under the data directory. Each file is labeled `FlameSpeedData[temperature].pkl`. 
- `FlameSpeedNotebook.ipynb` - this is a Jupyter Notebook in which the data collected by the acquisition file is processed. One should be able to run the notebook without having access to any of the chemical mechanisms. You may want to run the optimize cells more than once to allow the kernel hyperparameter training to converge. If you encounter too many `NaN` values, try lowering the learning rate or momentum terms.
- `VelocityNotebook.ipynb` - this is a Jupyter Notebook in which a velocity flow-field of a center-body combustion test problem is analyzed at multiple resolution levels and solution schemes. 
