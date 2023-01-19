# BISON

Accurate differentiation of brain tissue types from MR images  is necessary in many neuroscience and clinical applications. Accurate automated tissue segmentation is challenging due to the variability in the tissue intensity profiles caused by differences in scanner models, acquisition protocols, as well as the age of the subjects and presence of pathology. We have developed BISON (Brain tIsue SegmentatiOn pipeliNe), a new pipeline for tissue segmentation using a random forests classifier and a set of intensity and location priors obtained based on T1w images.


## Execution Example:
python BISON.py -c RF -m Trained_Classifiers/ -o Outputs/ -t Temp_Files/ -e PT -n List.csv -p Trained_Classifiers/ -l 3

## Input csv format:
Subjects,T1s,Masks,XFMs 
S1,t1.mnc,mask.mnc,xfm.xfm

This version of the pipeline is also available [here](https://nist.mni.mcgill.ca/tissue-classification/)

## New Version with 8 Labels 
python ./BISON.py -c RF0 -m Pretrained_Library/ -o  Outputs/ -t Temp_Files/ -e PT -n List.csv  -p  Pretrained_Library/ -l 8


## Reference:
- Dadar, M., & Collins, D. L. (2021). BISON: Brain tissue segmentation pipeline using T1‐weighted magnetic resonance images and a random forest classifier. Magnetic Resonance in Medicine, 85(4), 1881-1894.


