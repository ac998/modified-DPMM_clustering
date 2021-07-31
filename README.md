# Modified Dirichlet Process Mixture Models for Video Analysis
This is the reference implementation for the non-parametric clustering algorithm developed in the paper [Likelihood Learning in Modified Dirichlet Process Mixture Model for Video Analysis](https://www.sciencedirect.com/science/article/abs/pii/S0167865519302557).  
This work is **still in progress** and the final reference version is coming soon. 

## Dependecies
- Python (>= 3.4)
- Cython (0.29.2)
- OpenCV (3.3.1)

#### Installation guide
Cython code requires a C compiler to be present in the system (gcc by default). See [here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) for more details.  
For Linux users, simply run `pip install cython` or `conda install -c anaconda cython` if you prefer using the Anaconda environment.  
For Windows users, please follow [this](https://github.com/cython/cython/wiki/InstallingOnWindows) or [this](https://stackoverflow.com/questions/52864588/how-to-install-cython-an-anaconda-64-bits-with-windows-10) for a step by step guide to installation.


## Citation
@article{KUMARAN2019211,
title = {Likelihood learning in modified Dirichlet Process Mixture Model for video analysis},  
journal = {Pattern Recognition Letters},  
author = {Santhosh, K. K. and Chakravarty, A. and Dogra, D. P. and Roy, P. P.}  
volume = {128},  
pages = {211-219},  
year = {2019},  
issn = {0167-8655},  
doi = [https://doi.org/10.1016/j.patrec.2019.09.005](https://doi.org/10.1016/j.patrec.2019.09.005),  
}
