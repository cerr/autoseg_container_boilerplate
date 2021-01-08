# autoseg_container_boilerplate

The boilerplate example code in this repository can be used for adapting home-brewed deep learning models for use with the CERR Auto-Segmentation Pipeline.

Please note: for optimal compatibility with the wrappers enclosed here, we suggest providing Python requirements compatible with miniconda3.

The main components of a pipeline which are required by CERR:
  1. Python wrapper: The script which processes the input imaging data (as prepared by CERR pipeline in H5 format) using the custom network model. 
  2. Singularity container recipe: Creates the virtualized Singularity container for running the process. The container bundles the model Python wrapper and additional dependencies. The wrapper is invoked by the container as defined by the command in the %%app section of the recipe.
  
### CERR wiki links
* [Auto-Segmentation models](https://github.com/cerr/CERR/wiki/Auto-Segmentation-models)
  
### References: 
* Aditya P. Apte, Aditi Iyer, Maria Thor, Rutu Pandya, Rabia Haq, Jue Jiang, Eve LoCastro, Amita Shukla-Dave, Nishanth Sasankan, Ying Xiao, Yu-Chi Hu, Sharif Elguindi, Harini Veeraraghavan, Jung Hun Oh, Andrew Jackson, Joseph O. Deasy, Library of deep-learning image segmentation and outcomes model-implementations, Physica Medica, Volume 73, 2020, Pages 190-196, ISSN 1120-1797, https://doi.org/10.1016/j.ejmp.2020.04.011.
