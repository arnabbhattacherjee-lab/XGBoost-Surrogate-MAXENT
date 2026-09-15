# Learning Chromatin Interaction Landscapes from Micro-C via Maximum Entropy Surrogates

## Overview
This project develops a machine-learning surrogate for maximum-entropy (MaxEnt) inference of chromatin interaction landscapes from Hi-C and Micro-C contact maps. The approach uses a two-stage XGBoost hurdle model trained on local contact-map patches to directly predict the interaction matrix λᵢⱼ, which describes the underlying chromatin organization.

The surrogate substantially reduces the computational cost of MaxEnt inference, from hours per genomic locus to seconds, while maintaining high agreement with independently inferred MaxEnt solutions. The predicted λ-matrices can also be used in forward polymer simulations to reproduce experimental chromatin contact maps, demonstrating that the learned interaction landscape retains physically meaningful information.

Overall, the project provides a scalable and physically interpretable framework for genome-wide chromatin ensemble inference, enabling the study of chromatin organization beyond the polymer-connectivity effects directly visible in experimental contact maps.
