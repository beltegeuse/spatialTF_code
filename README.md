# A Spatial Target Function for Metropolis Photon Tracing #

This code extends Mitsuba and implements the algorithm presented in the paper "A Spatial Target Function for Metropolis Photon Tracing" by Gruson et al.

The algorithm proposed in this paper is based on Stochastic Progressive Photon Mapping (SPPM) with a specific Target Function (TF). This TF is used in Metropolis sampling to distribute the light samples over the image plane to minimize relative error.

Project home page:
https://beltegeuse.github.io/research/publication/2016_spatialIF/

In case of problems/questions/comments don't hesitate to contact us directly:
adrien.gruson@gmail.com

## Features ##

This code is a modified version of Mitsuba rendering original code. Compare to it, the following algorithm have been added:
 
 "Arbitrary Importance Functions for Metropolis Light Transport"
 J Hoberock and J. C. Hart
 Multi-pass TF implemented with PSSMLT and MLT.
 
 "Robust Adaptive Photon Tracing using Photon Path Visibility"
 T. Hachisuka and H. W.  
 Visibility TF with SPPM.
  
 "Improved Stochastic Progressive Photon Mapping with. Metropolis Sampling"
 J. Chen et al.
 Custoum TF with SPPM.
 
 "Photon Shooting with Programmable Scalar Contribution Function"
 Q. Zeng and C. Zeng
 Another TF with SPPM. Note that the implementation is not fully finished.
 
## Code organisation ##

The main integrator used in this project is "msppm". It is a general metropolis version of SPPM where the TF can be changed easily. This current technique 


## Change log ##
