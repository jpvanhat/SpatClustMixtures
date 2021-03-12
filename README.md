# SpatClustMixtures: Spatiotemporal Clustering using Gaussian Processes Embedded in a Mixture Model

Maintainer: Jarno Vanhatalo (jarno.vanhatalo@helsinki.fi)

## Reference

If you use SpatClustMixtures or otherwise refer to it, please use the following reference:

Jarno Vanhatalo, Scott D. Foster and Geoffrey R. Hosack (manuscript). Spatiotemporal Clustering using Gaussian Processes Embedded in a Mixture Model

## Introduction 

SpatClustMixtures is a Matlab code package to do Gaussian process (GP) spatiotemporal smoothing for cluster components in mixture modeling of multidimensional data. Many applications of clustering, including the majority of tasks in ecology, use data that is inherently spatial and is often also temporal. However, spatiotemporal dependence is typically ignored when clustering multivariate data. We present a finite mixture model for spatial and spatiotemporal clustering that incorporates spatial and spatiotemporal autocorrelation by including appropriate Gaussian processes into a model for the mixing proportions. We also allow for flexible and semi-parametric dependence on environmental covariates, once again using Gaussian processes. The package employs Bayesian inference through three tiers of approximate methods: a Laplace approximation that allows efficient analysis of large data sets, and both partial and full Markov chain Monte Carlo approaches that improve accuracy at the cost of increased computational time. 

The clustering model and methods are described in detail in the above reference.


## Installing the toolbox 

1) Install the GPstuff toolbox by cloning the develop branch from <https://github.com/gpstuff-dev/gpstuff> and following the installation instructions
   
2) Clone this “SpatClustMixtures” repository into your own computer.  

## User quide (very short)

1) Open your Matlab
2) Change the Matlab working directory to the root-directory of the SpatClustMixtures package (the same folder from where you find the file demo_simulated_data.m
3) Open demo_simulated_data.m to see an example on how to use the package

Note! It is essential to set the working directory of Matlab to the root-directory of the SpatClustMixtures package since the package has overloaded functions with GPstuff (such as gpla_e). Using the GPstuff version of these files results in an error. This package will be merged to GPstuff at a later stage.

## License 
This software is distributed under the GNU General Public Licence (version 3 or later); please refer to the file Licence.txt, included with the software, for details.
