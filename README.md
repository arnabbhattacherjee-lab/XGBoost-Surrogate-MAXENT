# Learning Chromatin Interaction Landscapes from Micro-C via Maximum Entropy Surrogates

## Overview
This project develops a machine-learning surrogate for maximum-entropy (MaxEnt) inference of chromatin interaction landscapes from Hi-C and Micro-C contact maps. The approach uses a two-stage XGBoost hurdle model trained on local contact-map patches to directly predict the interaction matrix λᵢⱼ, which describes the underlying chromatin organization.

The surrogate substantially reduces the computational cost of MaxEnt inference, from hours per genomic locus to seconds, while maintaining high agreement with independently inferred MaxEnt solutions. The predicted λ-matrices can also be used in forward polymer simulations to reproduce experimental chromatin contact maps, demonstrating that the learned interaction landscape retains physically meaningful information.

Overall, the project provides a scalable and physically interpretable framework for genome-wide chromatin ensemble inference, enabling the study of chromatin organization beyond the polymer-connectivity effects directly visible in experimental contact maps.

## Surrogate framework
The overall workflow consists of two major stages. We have experimentally measured Micro-C contact maps and their corresponding MaxEnt-inferred λ-maps for 12 genomic loci. These data are used to train a two-stage XGBoost hurdle model that learns the mapping from local contact-map features to the underlying interaction matrix λij. The trained surrogate is then used to predict λ-maps directly from Hi-C contact maps. Finally, the predicted λ-maps are used as input to forward polymer simulations to generate ensembles of 3D chromatin structures and corresponding simulated contact maps, which are compared against the experimental Hi-C maps.
<p align="center">
  <img src="Figures/Framework.png" width="1000">
</p>
