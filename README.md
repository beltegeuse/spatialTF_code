# A Spatial Target Function for Metropolis Photon Tracing #
This code extends Mitsuba and implements the algorithm presented in the paper "A Spatial Target Function for Metropolis Photon Tracing" by Gruson et al.

The algorithm proposed in this article is based on Stochastic Progressive Photon Mapping (SPPM) with a particular Target Function (TF). This TF is used in Metropolis sampling to distribute the light samples over the image plane to minimize relative error.

Project home page:
https://profs.etsmtl.ca/agruson/publication/2016_spatialif/

In the case of problems/questions/comments don't hesitate to contact us directly:
adrien.gruson@gmail.com

## Features ##
This code is a modified version of Mitsuba rendering original code. Compare to it; the following algorithm has been added:

 "Arbitrary Importance Functions for Metropolis Light Transport" by J Hoberock and J. C. Hart. Only Multi-pass TF has been implemented in this framework. Our implementation is compatible with PSSMLT and MLT.
 
 "Robust Adaptive Photon Tracing using Photon Path Visibility." by T. Hachisuka and H. W. Jensen. It is the first work to apply MCMC and photon tracing. In this technique, the photons are guided using the visibility information (= if they contribute or not).
  
 "Improved Stochastic Progressive Photon Mapping with Metropolis Sampling" by J. Chen et al. It is a Custom TF for photon tracing that modulate the visibility by a precomputed inverse density.
 
 "Visual importance-based adaptive photon tracing" by Q. Zeng and C. Zeng. It is another TF for photon tracing. Note that the implementation is not entirely functional.
 
## Code organisation ##

The primary integrator used in this project is "msppm." This integrator is a "flexible" Metropolis version of SPPM where the TF can be changed easily.

Note that the number of photon shot per iteration (when gather points are regenerated) has to be big enough in case of dynamic TF to ensure a correct normalization factor (only in a really difficult scene). Please see scenes example at the following address:
https://data.adrien-gruson.com/research/2016_SpatialTF/comparison/index.html

The code of "msppm" is a little bit complex due to our aim to add VCM implementation on top of it. However, this incorporation is not straightforward at all. Please check this new paper for more details:
"Robust Light Transport Simulation via Metropolized Bidirectional Estimators" by M. Sik et al.

Finally, some internal code inside Mitsuba has been changed (like how photons are shot). So, if you plan to copy the code in your own Mitsuba version, you need to be carefull.
