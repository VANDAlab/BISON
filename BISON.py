# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:43:32 2019
######################################################################################################################################
Copyright (c)  2022 Mahsa Dadar

Script for 9 Class Tissue Segmentation from T1-w, T2-w, PD, and FLAIR images
This version of the pipeline has the option to perform its own preprocessing and works with .nii images.
The pipeline provides the option of using other classifiers, but is has been tested and validated with random forests 
Input arguments:
BISON.py -c <Classifier (Default: RF)> -i <Input CSV File> 
 -m <Template Mask File>  -f <Number of Folds in K-fold Cross Validation (Default=10)>
 -o <Output Path> -t <Temp Files Path> -e <Classification Mode> -n <New Data CSV File> 
 -p <Pre-trained Classifiers Path> -d  <Do Preprocessing> -l < The Number of Classes>
 
CSV File Column Headers: Subjects, XFMs, T1s, T2s, PDs, FLAIRs, Labels, Masks
Subjects:   Subject ID
T1s:        Path to preprocessed T1 image, coregistered with primary modality (mandatory)
T2s:        Path to preprocessed T2 image, coregistered with primary modality (if exists)
PD:         Path to preprocessed PD image, coregistered with primary modality (if exists)
FLAIR:      Path to preprocessed FLAIR image, coregistered with primary modality (if exists)
XFMs:       Nonlinear transformation from template to primary modality image 
Masks:      Brain mask or mask of region of interest
Labels:     Labels (For Training, not necessary if using pre-trained classifiers)
 
Preprocessing Options: 
 Y:   Perform Preprocessing 

Classification Mode Options: 
 CV:   Cross Validation (On The Same Dataset) 
 TT:   Train-Test Model (Training on Input CSV Data, Segment New Data, Needs an extra CSV file)
 PT:   Using Pre-trained Classifiers  
 
Classifier Options:
 NB:   Naive Bayes
 LDA:  Linear Discriminant Analysis
 QDA:  Quadratic Discriminant Analysis
 LR:   Logistic Regression
 KNN:  K Nearest Neighbors 
 RF:   Random Forest 
 SVM:  Support Vector Machines 
 Tree: Decision Tree
 Bagging
 AdaBoost
#####################################################################################################################################
@author: mdadar
"""
import os
import numpy as np
import sys
import getopt
import SimpleITK as sitk

try:
    import joblib
except ModuleNotFoundError:
    # for old scikit-learn
    from sklearn.externals import joblib
import tempfile

def run_command(cmd_line):
    """
    Execute command and check the return status
    throw an exception if command failed
    """
    r=os.system(cmd_line)
    if r!=0:
        raise OSError(r,cmd_line)

#DEBUG
def draw_histograms(hist,out,modality='',dpi=100 ):
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
   
    x=np.arange(hist.shape[0])
    for c in range(hist.shape[1]):
        plt.plot(x, hist[:,c], label=f'{c+1}')  # Plot some data on the (implicit) axes.

    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.legend()
    if modality is not None:
        plt.title(modality)

    plt.savefig(out, bbox_inches='tight', dpi=dpi)
    plt.close()
    plt.close('all')
#DEGUB

def doPreprocessing(path_nlin_mask,path_Temp, ID_Test, Label_Files_Test , Label, T1_Files_Test , t1 , T2_Files_Test , t2 , PD_Files_Test , pd , FLAIR_Files_Test , flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair):
    nlmf = 'Y'
    nuf = 'Y'
    volpolf = 'Y'
    if '.nii' in T1_Files_Test[0]: 
        fileFormat = 'nii'
    else:
        fileFormat = 'mnc'
    preprocessed_list = {}
    str_t1_proc = ''
    str_t2_proc = ''
    str_pd_proc = ''
    str_flair_proc = ''
    preprocessed_list_address=path_Temp+'Preprocessed.csv'
    print('Preprocessing The Images ...')
    for i in range(0 , len(T1_Files_Test)):
        if (t1 != ''):
            str_File_t1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):
                new_command = 'nii2mnc ' + str_File_t1 + ' ' + path_Temp + str(ID_Test[i]) + '_T1.mnc'      
            else:
                new_command = 'cp ' + str_File_t1 + ' ' + path_Temp + str(ID_Test[i]) + '_T1.mnc'
            os.system(new_command)
            new_command = 'bestlinreg_s2 ' +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' +  path_av_t1 + ' ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  path_nlin_mask + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc -invert_transform -like ' + path_Temp + str(ID_Test[i]) + '_T1.mnc -nearest -clobber'
            os.system(new_command)
            str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1.mnc'
            str_main_modality = str_t1_proc
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_T1.mnc ' + path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc -beta 0.7 -clobber'
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc'
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_T1_N3.mnc -mask '+ path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc  -iter 200 -distance 50 -clobber'
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_N3.mnc'
            if (volpolf == 'Y'):
                #new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_T1_N3.mnc '  + path_av_t1 + ' --order 2 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_T1_norm --clobber --source_mask '+ path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc --target_mask '+path_nlin_mask 
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_T1_N3.mnc '  + path_av_t1 + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_T1_norm --clobber ' + path_Temp + str(ID_Test[i]) + '_T1_VP.mnc '
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_VP.mnc'
                
            new_command = 'bestlinreg_s2 ' +  str_t1_proc + ' ' +  path_av_t1 + ' ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_lin.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  str_t1_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_lin.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T1_lin.mnc -like ' + path_av_t1 + ' -clobber'
            os.system(new_command)
            new_command = 'nlfit_s ' + path_Temp + str(ID_Test[i]) + '_T1_lin.mnc ' +  path_av_t1 + ' ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_nlin.xfm -level 2 -clobber'
            os.system(new_command)
            new_command = 'xfmconcat ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_lin.xfm ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_nlin.xfm '+ path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_both.xfm'
            os.system(new_command)

        if (t2 != ''):
            str_File_t2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):                
                new_command = 'nii2mnc ' + str_File_t2 + ' ' + path_Temp + str(ID_Test[i]) + '_T2.mnc'      
            else:
                new_command = 'cp ' + str_File_t2 + ' ' + path_Temp + str(ID_Test[i]) + '_T2.mnc'
            os.system(new_command)
            new_command = 'bestlinreg_s2 -lsq6 ' +  path_Temp + str(ID_Test[i]) + '_T2.mnc '  +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' + ' ' +  path_Temp + str(ID_Test[i]) + '_T2toT1.xfm -mi -close'
            os.system(new_command)
            str_t2_proc = path_Temp + str(ID_Test[i]) + '_T2.mnc'
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_T2.mnc ' + path_Temp + str(ID_Test[i]) + '_T2_NLM.mnc -beta 0.7 -clobber'
                os.system(new_command)
                str_t2_proc = path_Temp + str(ID_Test[i]) + '_T2_NLM.mnc'
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_T2_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_T2_N3.mnc -mask '+ path_Temp + str(ID_Test[i]) + '_T2_Mask.mnc  -iter 200 -distance 50 -clobber'
                os.system(new_command)
                str_t2_proc = path_Temp + str(ID_Test[i]) + '_N3.mnc'
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_T2_N3.mnc '  + path_av_t2 + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_T2_norm --clobber ' + path_Temp + str(ID_Test[i]) + '_T2_VP.mnc ' 
                os.system(new_command)
                str_t2_proc = path_Temp + str(ID_Test[i]) + '_T2_VP.mnc'                         
            
        if (pd != ''):
            str_File_pd = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):                
                new_command = 'nii2mnc ' + str_File_pd + ' ' + path_Temp + str(ID_Test[i]) + '_PD.mnc'    
            else:
                new_command = 'cp ' + str_File_pd + ' ' + path_Temp + str(ID_Test[i]) + '_PD.mnc'    
            os.system(new_command)
            new_command = 'bestlinreg_s2 -lsq6 ' +  path_Temp + str(ID_Test[i]) + '_PD.mnc '  +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' + ' ' +  path_Temp + str(ID_Test[i]) + '_PDtoT1.xfm -mi -close'
            os.system(new_command)
            str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD.mnc'
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_PD.mnc ' + path_Temp + str(ID_Test[i]) + '_PD_NLM.mnc -beta 0.7 -clobber'
                os.system(new_command)
                str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD_NLM.mnc'
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_PD_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_PD_N3.mnc -mask '+ path_Temp + str(ID_Test[i]) + '_PD_Mask.mnc  -iter 200 -distance 50 -clobber'
                os.system(new_command)
                str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD_N3.mnc'
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_PD_N3.mnc '  + path_av_pd + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_PD_norm --clobber ' + path_Temp + str(ID_Test[i]) + '_PD_VP.mnc ' 
                os.system(new_command)
                str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD_VP.mnc'
                
        if (flair != ''):
            str_File_flair = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):                
                new_command = 'nii2mnc ' + str_File_flair + ' ' + path_Temp + str(ID_Test[i]) + '_FLAIR.mnc'   
            else:
                new_command = 'cp ' + str_File_flair + ' ' + path_Temp + str(ID_Test[i]) + '_FLAIR.mnc' 
            os.system(new_command)
            new_command = 'bestlinreg_s2 -lsq6 ' +  path_Temp + str(ID_Test[i]) + '_FLAIR.mnc '  +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' + ' ' +  path_Temp + str(ID_Test[i]) + '_FLAIRtoT1.xfm -mi -close'
            os.system(new_command)
            str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR.mnc'
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_FLAIR.mnc ' + path_Temp + str(ID_Test[i]) + '_FLAIR_NLM.mnc -beta 0.7 -clobber'
                os.system(new_command)
                str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR_NLM.mnc'
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_FLAIR_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_FLAIR_N3.mnc -mask '+ path_Temp + str(ID_Test[i]) + '_FLAIR_Mask.mnc  -iter 200 -distance 50 -clobber'
                os.system(new_command)
                str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR_N3.mnc'
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_FLAIR_N3.mnc '  + path_av_flair + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_FLAIR_norm --clobber '+ path_Temp + str(ID_Test[i]) + '_FLAIR_VP.mnc ' 
                os.system(new_command)
                str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR_VP.mnc'
        if (Label != ''):
            str_File_Label = str(Label_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):   
                new_command = 'nii2mnc ' + str_File_Label + ' ' + path_Temp + str(ID_Test[i]) + '_Label.mnc' 
            else:
                new_command = 'cp ' + str_File_Label + ' ' + path_Temp + str(ID_Test[i]) + '_Label.mnc'
            os.system(new_command)
            str_File_Label = path_Temp + str(ID_Test[i]) + '_Label.mnc' 
        
        if (flair != ''):
            new_command = 'bestlinreg_s2 -lsq6 ' +  str_flair_proc + ' '  +  str_main_modality + ' ' +  path_Temp + str(ID_Test[i]) + '_FLAIRtoMain.xfm  -mi -close -clobber'
            os.system(new_command)
            new_command = 'mincresample ' +  str_flair_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_FLAIRtoMain.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_FLAIR_CR.mnc -like ' + str_main_modality + ' -clobber'
            os.system(new_command)
            str_t1_proc =  path_Temp + str(ID_Test[i]) + '_T1_CR.mnc'        
                                   
        if (t2 != ''):
            new_command = 'bestlinreg_s2 -lsq6 ' +  str_t2_proc + ' '  +  str_main_modality + ' ' +  path_Temp + str(ID_Test[i]) + '_T2toMain.xfm -mi -close'
            os.system(new_command)
            new_command = 'mincresample ' +  str_t2_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_T2toMain.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T2_CR.mnc -like ' + str_main_modality + ' -clobber'
            os.system(new_command)
            str_t1_proc =  path_Temp + str(ID_Test[i]) + '_T2_CR.mnc'
        
        if (pd != ''):    
            new_command = 'bestlinreg_s2 -lsq6 ' +  str_pd_proc + ' '  +  str_main_modality + ' ' +  path_Temp + str(ID_Test[i]) + '_PDtoMain.xfm -mi -close'
            os.system(new_command)
            new_command = 'mincresample ' +  str_pd_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_PDtoMain.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_PD_CR.mnc -like ' + str_main_modality + ' -clobber'
            os.system(new_command)
            str_pd_proc =  path_Temp + str(ID_Test[i]) + '_PD_CR.mnc'

        new_command = 'mincresample ' +  path_nlin_mask + ' -transform ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_both.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_Mask_nl.mnc -like ' + str_main_modality + ' -invert_transformation -nearest -clobber'
        os.system(new_command) 
        str_Mask = path_Temp + str(ID_Test[i]) + '_Mask_nl.mnc'
        nl_xfm = path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_both.xfm'
        print('.')
        preprocessed_list[0,0]= 'Subjects,T1s,Masks,XFMs'
        preprocessed_list[i+1,0]= str(ID_Test[i]) + ',' + str_t1_proc + ',' + str_Mask + ',' + nl_xfm
        if (t2 != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',T2s'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_t2_proc
        if (pd != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',PDs'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_pd_proc
        if (flair != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',FLAIRs'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_flair_proc
        if (Label != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',Labels'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_File_Label
            
    outfile = open( preprocessed_list_address, 'w' )
    for key, value in sorted( preprocessed_list.items() ):
        outfile.write(  str(value) + '\n' )
    outfile = open( preprocessed_list_address, 'w' )
    for key, value in sorted( preprocessed_list.items() ):
        outfile.write(  str(value) + '\n' )
    return [preprocessed_list_address]
###########################################################################################################################################################################
def Calculate_Tissue_Histogram(Files_Train , Masks_Train , Label_Files_Train , image_range , n_labels):
    PDF_Label = np.zeros(shape = (image_range , n_labels),dtype=np.float64)
    print(('Calculating Histograms of Tissues: .'), end=' ',flush=True)
    for i in range(0 , len(Files_Train)):
        print(('.'), end='',flush=True)
        str_File = str(Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        str_Mask = str(Masks_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        str_Label = str(Label_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')

        manual_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(str_Label))
        image_vol = sitk.GetArrayFromImage(sitk.ReadImage(str_File))      
        brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))> 0
        
        image_vol  = np.round(image_vol).astype(np.int32)

        for nl in range(0 , n_labels):
            masked_vol = image_vol[ (manual_segmentation==(nl+1)) * brain_mask]
            for j in range(1 , image_range):
                PDF_Label[j,nl] = PDF_Label[j,nl] + np.sum( masked_vol == j, dtype=np.float64)

    # VF: normalize after all files are processed
    for nl in range(0 , n_labels):
        PDF_Label[:,nl] = PDF_Label[:,nl] / np.sum(PDF_Label[:,nl])
    print(' Done.')
    return PDF_Label
###########################################################################################################################################################################
def load_csv(csv_file):
    import csv
    data = {}
    with open(csv_file , 'r') as f:
        for r in csv.DictReader(f):
            for k in r.keys():
                try:
                    data[k].append(r[k])
                except KeyError:
                    data[k] = [r[k]]
    return data
###########################################################################################################################################################################
def get_Train_Test(Indices_G , K , IDs):
    i_train = 0
    i_test = 0
    ID_Train = np.empty(shape = (np.sum(Indices_G != K) , 1) , dtype = list , order = 'C')
    ID_Test = np.empty(shape = (np.sum(Indices_G == K) , 1) , dtype = list , order = 'C')        
    for i in range(0 , len(Indices_G)):
        if (Indices_G[i] != K):
            ID_Train[i_train] = IDs[i]
            i_train = i_train + 1
        if (Indices_G[i] == K):
            ID_Test[i_test] = IDs[i]
            i_test = i_test + 1
    return [ID_Train , ID_Test]
###########################################################################################################################################################################
def get_addressess(TestList):
    InputListInfo_Test = load_csv(TestList)    
    ID_Test = InputListInfo_Test['Subjects']
    if 'XFMs' in InputListInfo_Test:    
        XFM_Files_Test = InputListInfo_Test['XFMs']
        xfmf = 'exists'
    else:
        xfmf = ''
        XFM_Files_Test = ''
    if 'Masks' in InputListInfo_Test:    
        Mask_Files_Test = InputListInfo_Test['Masks']
        maskf = 'exists'
    else:
        maskf = ''
        Mask_Files_Test = ''
    if 'T1s' in InputListInfo_Test:    
        T1_Files_Test = InputListInfo_Test['T1s']
        t1 = 'exists'
    else:
        t1 =''
        T1_Files_Test = ''
    if 'T2s' in InputListInfo_Test:    
        T2_Files_Test = InputListInfo_Test['T2s']
        t2 = 'exists'
    else:
        t2 =''
        T2_Files_Test = ''
    if 'PDs' in InputListInfo_Test:    
        PD_Files_Test = InputListInfo_Test['PDs']
        pd = 'exists'
    else:
        pd = ''
        PD_Files_Test = ''
    if 'FLAIRs' in InputListInfo_Test:    
        FLAIR_Files_Test = InputListInfo_Test['FLAIRs']
        flair = 'exists'
    else:
        flair = ''
        FLAIR_Files_Test = ''
    if 'Labels' in InputListInfo_Test:    
        Label_Files_Test = InputListInfo_Test['Labels']
        Label = 'exists'
    else:
        Label = ''
        Label_Files_Test = ''

    return [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, Label_Files_Test, Label]
###########################################################################################################################################################################

def warp_and_read_prior(prior, ref_scan, xfm, tmp_file_location,clobber=False):
    """ Apply INVERSE xfm to prior , like ref_scan and store into tmp_file_location
    then read into numpy array. 
    Parameters:
        prior - input scan
        ref_scan - reference space
        xfm - transformation
        tmp_file_location - output file
        
    Returns: 
        numpy array of floats of the contents of output
    """
    run_command(f'itk_resample {prior} --like  {ref_scan} --transform {xfm} {tmp_file_location} --clobber')
    return sitk.GetArrayFromImage(sitk.ReadImage(tmp_file_location))

def main(argv):   
    # Default Values    
    n_folds=10
    image_range = 256
    subject = 0
    Classifier='RF'    
    doPreprocessingf = False
    path_trained_classifiers=''
    InputList=''
    TestList=''
    path_Temp=None
    path_nlin_files = ''
    ClassificationMode = ''
    path_output = ''
    TestList = ''
    n_labels = 3
    n_jobs = 1
    temp_dir = None
    nifti = False


    _help="""
BISON.py -c <Classifier (Default: LDA)> 
         -i <Input CSV File>  - for training only (TT or CV mode)
         -m <Templates directory> - location of template files, unless -p option is used 
         -f <Number of Folds in K-fold Cross Validation (Default=10)>
         -o <Output Path>
         -t <Temp Files Path>
         -e <Classification Mode> (CV/TT/PT)
         -n <New Data CSV File>  - for segmenting only (TT or PT mode)
         -p <Pre-trained Classifiers Path> - pretrained classfier and templates directory
         -d <Do Preprocessing>  - run nonlinear registration
         -l <The Number of Classes, default 3> 
         -j <n> maximum number of jobs (CPUs) to use for classification default 1, -1 - all possible
         --nifti output a nifti file

CSV File Column Headers: Subjects, XFMs, T1s, T2s, PDs, FLAIRs, Labels, Masks
Preprocessing Options: 
    Y:   Perform Preprocessing 
Classification Mode Options:
    CV:   Cross Validation (On The Same Dataset) 
    TT:   Train-Test Model (Training on Input CSV Data, Segment New Data, Needs an extra CSV file)
    PT:   Using Pre-trained Classifiers 

Classifier Options:
    NB:   Naive Bayes
    LDA:  Linear Discriminant Analysis
    QDA:  Quadratic Discriminant Analysis
    LR:   Logistic Regression
    KNN:  K Nearest Neighbors 
    RF:   Random Forest 
    SVM:  Support Vector Machines 
    Tree: Decision Tree
    Bagging
    AdaBoost
    """

    try:
        opts, args = getopt.getopt(argv,"hc:i:m:o:t:e:n:f:p:dl:j:",["cfile=","ifile=","mfile=","ofile=","tfile=","efile=","nfile=","ffile=","pfile=","dfile","lfile=","jobs=","nifti"])
    except getopt.GetoptError:
        print(_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(_help)
            sys.exit()
        elif opt in ("-c", "--cfile"):
            Classifier = arg
        elif opt in ("-i", "--ifile"):
            InputList = arg
        elif opt in ("-m", "--mfile"):
            path_nlin_files = arg
        elif opt in ("-o", "--ofile"):
            path_output = arg
        elif opt in ("-t", "--tfile"):
            if not os.path.exists(arg):
                os.makedirs(arg)
            path_Temp = arg+str(np.random.randint(1000000, size=1)).replace("[",'').replace("]",'').replace(" ",'').replace(" ",'')+'_BISON_'
        elif opt in ("-e", "--efile"):
            ClassificationMode = arg
        elif opt in ("-n", "--nfile"):
            TestList = arg
        elif opt in ("-f", "--ffile"):
            n_folds = int(arg)
        elif opt in ("-p", "--pfile"):
            path_trained_classifiers = arg
        elif opt in ("-d", "--dfile"):
            doPreprocessingf = True
        elif opt in ("-l", "--lfile"):
            n_labels = int(arg)
        elif opt in ("-j", "--jobs"):
            n_jobs = int(arg)
        elif opt in ("--nifti"):
            nifti = True
        else:
            print("Unknown option:",opt)
            print(_help)
            sys.exit(1)

    if path_Temp is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="BISON_")
        path_Temp = temp_dir.name + os.sep + 'temp_'

    print('The Selected Input CSV File is ', InputList)
    print('The Selected Test CSV File is ', TestList)
    print('The Selected Classifier is ', Classifier)
    print('The Classification Mode is ', ClassificationMode)
    print('The Selected Template Mask is ', path_nlin_files)
    print('The Selected Output Path is ', path_output)    
    print('The Assigned Temp Files Path is ', path_Temp)

    if doPreprocessingf:
        print('Preprocessing:  Yes')

    if (Classifier == 'NB'):
        # Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
    elif (Classifier == 'LDA'):
        # Linear Discriminant Analysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(solver = "svd" )  
    elif (Classifier == 'QDA'):
        # Quadratic Discriminant Analysis
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis()
    elif (Classifier == 'LR'):
        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C = 200 , penalty = 'l2', tol = 0.01)
    elif (Classifier == 'KNN'):
        # K Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors = 10)
    elif (Classifier == 'Bagging'):
        # Bagging
        from sklearn.ensemble import BaggingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        clf = BaggingClassifier(KNeighborsClassifier() , max_samples = 0.5 , max_features = 0.5)
    elif (Classifier == 'AdaBoost'):
        # AdaBoost
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators = 100)
    elif (Classifier == 'RF'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 100, n_jobs=n_jobs)
    elif (Classifier == 'RF0'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=10,verbose=True)
    elif (Classifier == 'RF1'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=20,verbose=True)
    elif (Classifier == 'RF2'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=40,verbose=True)
    elif (Classifier == 'RF3'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=100,verbose=True)
    elif (Classifier == 'RF4'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=1000,verbose=True)
    elif (Classifier == 'RF5'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=100, min_samples_split=100, min_samples_leaf=100, verbose=True)
    elif (Classifier == 'SVM'):
        # Support Vector Machines
        from sklearn import svm
        clf = svm.LinearSVC()
    elif (Classifier == 'Tree'):
        # Decision Tree
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
    else:
        print('The Selected Classifier Was Not Recognized')
        sys.exit()

    if (InputList != ''):
    	[IDs, XFM_Files, xfmf, Mask_Files, maskf, T1_Files, t1, T2_Files, t2, PD_Files, pd, FLAIR_Files, flair, Label_Files, Label] = get_addressess(InputList)

####################### Preprocessing ####################################################################################################################################
    if (path_nlin_files == ''):
	    print('No path has been defined for the template files')
	    sys.exit()
    if (path_nlin_files != ''):
    	path_nlin_mask = path_nlin_files + os.sep + 'Mask.mnc'
    	path_av_t1 = path_nlin_files +  os.sep + 'Av_T1.mnc'
    	path_av_t2 = path_nlin_files +  os.sep + 'Av_T2.mnc'
    	path_av_pd = path_nlin_files +  os.sep + 'Av_PD.mnc'
    	path_av_flair = path_nlin_files +  os.sep + 'Av_FLAIR.mnc'

    if ((path_trained_classifiers == '') & (ClassificationMode == 'PT')):
	    print('No path has been defined for the pretrained classifiers')
	    sys.exit()
    if (path_trained_classifiers != ''):    
        path_av_t1 = path_trained_classifiers + os.sep + 'Av_T1.mnc'
        path_av_t2 = path_trained_classifiers + os.sep + 'Av_T2.mnc'
        path_av_pd = path_trained_classifiers + os.sep + 'Av_PD.mnc'
        path_av_flair = path_trained_classifiers + os.sep + 'Av_FLAIR.mnc'   

    if (n_labels == 0):
        print('The number of classes has not been determined')
        sys.exit()

    if (ClassificationMode == ''):
        print('The classification mode has not been determined')
        sys.exit()
###########################################################################################################################################################################
    if ClassificationMode == 'CV':
        if doPreprocessingf:
            doPreprocessing(path_nlin_mask,path_Temp, IDs, Label_Files , Label, T1_Files , t1 , T2_Files , t2 , PD_Files , pd , FLAIR_Files, flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair)
            [IDs, XFM_Files, xfmf, Mask_Files, maskf, T1_Files, t1, T2_Files, t2, PD_Files, pd, FLAIR_Files, flair, Label_Files, Label] = get_addressess( path_Temp + 'Preprocessed.csv')

        if Label == '':    
            print('No Labels to Train on')
            sys.exit()

        Indices_G = np.random.permutation(len(IDs)) * n_folds / len(IDs)
        Kappa = np.zeros(shape = (len(IDs) , n_labels))
        ID_Subject = np.empty(shape = (len(IDs),1) , dtype = list, order = 'C')       

        for K in range(0 , n_folds):
            [ID_Train , ID_Test] = get_Train_Test(Indices_G , K , IDs)    
            [XFM_Files_Train , XFM_Files_Test] = get_Train_Test(Indices_G , K , XFM_Files)    
            [Label_Files_Train , Label_Files_Test] = get_Train_Test(Indices_G , K , Label_Files)    
            [Mask_Files_Train , Mask_Files_Test] = get_Train_Test(Indices_G , K , Mask_Files)    
            
            n_features=n_labels           
            if (t1 != ''):        
                [T1_Files_Train , T1_Files_Test] = get_Train_Test(Indices_G , K , T1_Files)    
                T1_PDF_Label = Calculate_Tissue_Histogram(T1_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                n_features = n_features + n_labels + 2
                
            if (t2 != ''):
                [T2_Files_Train , T2_Files_Test] = get_Train_Test(Indices_G , K , T2_Files)    
                T2_PDF_Label = Calculate_Tissue_Histogram(T2_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                n_features = n_features + n_labels + 3
                
            if (pd != ''):
                [PD_Files_Train , PD_Files_Test] = get_Train_Test(Indices_G , K , PD_Files)    
                PD_PDF_Label = Calculate_Tissue_Histogram(PD_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                n_features = n_features + n_labels + 3
                
            if (flair != ''):
                [FLAIR_Files_Train , FLAIR_Files_Test] = get_Train_Test(Indices_G , K , FLAIR_Files)    
                FLAIR_PDF_Label = Calculate_Tissue_Histogram(FLAIR_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                n_features = n_features + n_labels + 3
            
            path_sp = path_nlin_files + os.sep + 'SP_'
            
            X_All = np.empty(shape = (0 , n_features) , dtype = float , order = 'C')
            Y_All = np.empty(shape = (0 , ) , dtype = np.int32 , order = 'C')
            
            for i in range(0 , len(ID_Train)):
                str_Train = str(ID_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                print(('Extracting The Features: Subject ID = ' + str_Train))
                
                str_Mask = str(Mask_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
                ind_Mask = (Mask > 0)
                N=int(np.sum(Mask))
                Label = str(Label_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'') 
                WMT = sitk.GetArrayFromImage(sitk.ReadImage(Label))

                nl_xfm = str(XFM_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(path_sp + str(nl+1) + '.mnc',Label,nl_xfm, path_Temp + 'train_' + str(i) + '_' + str(K) + '_tmp_sp_'+str(nl+1)+'.mnc')
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]

                if (t1 != ''):
                    str_T1 = str(T1_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T1 = sitk.GetArrayFromImage(sitk.ReadImage(str_T1))
                    av_T1 = warp_and_read_prior(path_av_t1, Label, nl_xfm, path_Temp + 'train_' + str(i)+'_' + str(K) + '_tmp_t1.mnc')
                    T1[T1 < 1] = 1
                    T1[T1 > (image_range - 1)] = (image_range - 1)
                    T1_Label_probability = np.empty(shape = (N, n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        T1_Label_probability[:,nl] = T1_PDF_Label[np.round(T1[ind_Mask]).astype(np.int),nl]
                    X_t1 = np.zeros(shape = (N , 2))
                    X_t1[0 : N , 0] = T1[ind_Mask]
                    X_t1[0 : N , 1] = av_T1[ind_Mask]
                    X_t1 = np.concatenate((X_t1 , T1_Label_probability) , axis = 1)

                if (t2 != ''):
                    str_T2 = str(T2_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T2 = sitk.GetArrayFromImage(sitk.ReadImage(str_T2))
                    av_T2 = warp_and_read_prior(path_av_t2,Label,nl_xfm,path_Temp + 'train_' + str(i)+'_'+str(K) + '_tmp_t2.mnc')
                    T2[T2 < 1] = 1
                    T2[T2 > (image_range - 1)] = (image_range - 1)
                    T2_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        T2_Label_probability[:,nl] = T2_PDF_Label[np.round(T2[ind_Mask]).astype(np.int32),nl]
                    if (t1 == ''):
                        X_t2 = np.zeros(shape = (N , 2))
                        X_t2[0 : N , 0] = T2[ind_Mask]
                        X_t2[0 : N , 1] = av_T2[ind_Mask]
                    if (t1 != ''):
                        X_t2 = np.zeros(shape = (N , 3))
                        X_t2[0 : N , 0] = T2[ind_Mask]
                        X_t2[0 : N , 1] = av_T2[ind_Mask]
                        X_t2[0 : N , 2] = T2[ind_Mask] / T1[ind_Mask]
                    X_t2 = np.concatenate((X_t2 , T2_Label_probability) , axis = 1)

                if (pd != ''):
                    str_PD = str(PD_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    PD = sitk.GetArrayFromImage(sitk.ReadImage(str_PD))
                    av_PD = warp_and_read_prior(path_av_pd,Label,nl_xfm,path_Temp + 'train_' + str(i)+'_'+str(K) + '_tmp_pd.mnc')
                    PD[PD < 1] = 1
                    PD[PD > (image_range - 1)] = (image_range - 1)
                    PD_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        PD_Label_probability[:,nl] = PD_PDF_Label[np.round(PD[ind_Mask]).astype(np.int32),nl]
                    if (t1 == ''):
                        X_pd = np.zeros(shape = (N , 2))
                        X_pd[0 : N , 0] = PD[ind_Mask]
                        X_pd[0 : N , 1] = av_PD[ind_Mask]
                    if (t1 != ''):
                        X_pd = np.zeros(shape = (N , 3))
                        X_pd[0 : N , 0] = PD[ind_Mask]
                        X_pd[0 : N , 1] = av_PD[ind_Mask]
                        X_pd[0 : N , 2] = PD[ind_Mask] / T1[ind_Mask]
                    X_pd = np.concatenate((X_pd , PD_Label_probability ) , axis = 1)

                if (flair != ''):
                    str_FLAIR = str(FLAIR_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    FLAIR = sitk.GetArrayFromImage(sitk.ReadImage(str_FLAIR))
                    av_FLAIR = warp_and_read_prior(path_av_flair,Label,nl_xfm, path_Temp + 'train_' + str(i) + '_'+str(K) + '_tmp_flair.mnc')
                    FLAIR[FLAIR < 1] = 1
                    FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                    FLAIR_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        FLAIR_Label_probability[:,nl] = FLAIR_PDF_Label[np.round(FLAIR[ind_Mask]).astype(np.int32),nl]
                    if (t1 == ''):
                        X_flair = np.zeros(shape = (N , 3))
                        X_flair[0 : N , 0] = FLAIR[ind_Mask]
                        X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                    if (t1 != ''):
                        X_flair = np.zeros(shape = (N , 4))
                        X_flair[0 : N , 0] = FLAIR[ind_Mask]
                        X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                        X_flair[0 : N , 2] = FLAIR[ind_Mask] / T1[ind_Mask]
                    X_flair = np.concatenate((X_flair , FLAIR_Label_probability ) , axis = 1)

                else:
                    X = np.zeros(shape = (N , 0))
                    X = np.concatenate((X , spatial_priors) , axis = 1)
                if (t1 != ''):
                    X = np.concatenate((X , X_t1) , axis = 1)
                if (t2 != ''):
                    X = np.concatenate((X , X_t2) , axis = 1)
                if (pd != ''):
                    X = np.concatenate((X , X_pd) , axis = 1)
                if (flair != ''):
                    X = np.concatenate((X , X_flair) , axis = 1)

                X_All = np.concatenate((X_All , X) , axis = 0)
                Y = np.zeros(shape = (N , ),dtype=np.int32)
                Y[0 : N , ] = (WMT[ind_Mask])
                Y_All = np.concatenate((Y_All , Y) , axis = 0)

            print('Training The Classifier ...')
            clf = clf.fit(X_All , Y_All)

            for i in range(0 , len(ID_Test)):
                str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                print(('Segmenting Volumes: Subject: ID = ' + str_Test))
                Label = str(Label_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                WMT = sitk.GetArrayFromImage(sitk.ReadImage(Label))
                str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
                ind_Mask = (Mask > 0)
                N=int(np.sum(Mask))
                nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(path_sp + str(nl+1) + '.mnc', Label,nl_xfm, path_Temp + 'test_' + str(i) + '_' + str(K) + '_tmp_sp_'+str(nl+1)+'.mnc')
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]                
                               
                if (t1 != ''):
                    str_T1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T1 = sitk.GetArrayFromImage(sitk.ReadImage(str_T1))
                    av_T1 = warp_and_read_prior(path_av_t1, Label, str_T1, path_Temp + 'test_' +  str(i) + '_' + str(K) + '_tmp_t1.mnc')
                    T1[T1 < 1] = 1
                    T1[T1 > (image_range - 1)] = (image_range - 1)
                    T1_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        T1_Label_probability[:,nl] = T1_PDF_Label[np.round(T1[ind_Mask]).astype(np.int32),nl]
                    N = len(T1_Label_probability)
                    X_t1 = np.zeros(shape = (N , 2))
                    X_t1[0 : N , 0] = T1[ind_Mask]
                    X_t1[0 : N , 1] = av_T1[ind_Mask]
                    X_t1 = np.concatenate((X_t1 , T1_Label_probability) , axis = 1)
    
                if (t2 != ''):                
                    str_T2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T2 = sitk.GetArrayFromImage(sitk.ReadImage(str_T2))
                    av_T2 = warp_and_read_prior(path_av_t2, Label, nl_xfm, path_Temp + 'test_' +  str(i) + '_' + str(K) + '_tmp_t2.mnc')
                    T2[T2 < 1] = 1
                    T2[T2 > (image_range - 1)] = (image_range - 1)
                    T2_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        T2_Label_probability[:,nl] = T2_PDF_Label[np.round(T2[ind_Mask]).astype(np.int32),nl]
                    N = len(T2_Label_probability)
                    if (t1 == ''):
                        X_t2 = np.zeros(shape = (N , 2))
                        X_t2[0 : N , 0] = T2[ind_Mask]
                        X_t2[0 : N , 1] = av_T2[ind_Mask]
                    if (t1 != ''):
                        X_t2 = np.zeros(shape = (N , 3))
                        X_t2[0 : N , 0] = T2[ind_Mask]
                        X_t2[0 : N , 1] = av_T2[ind_Mask]    
                        X_t2[0 : N , 2] = T2[ind_Mask] / T1[ind_Mask]  
                    X_t2 = np.concatenate((X_t2 , T2_Label_probability) , axis = 1)
                    
                if (pd != ''):
                    str_PD = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    PD = sitk.GetArrayFromImage(sitk.ReadImage(str_PD))
                    av_PD = warp_and_read_prior(path_av_pd, Label, nl_xfm, path_Temp + 'test_' +  str(i) + '_' + str(K) + '_tmp_pd.mnc')
                    PD[PD < 1] = 1
                    PD[PD > (image_range - 1)] = (image_range - 1)
                    PD_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        PD_Label_probability[:,nl] = PD_PDF_Label[np.round(PD[ind_Mask]).astype(np.int32),nl]
                    N = len(PD_Label_probability)
                    if (t1 == ''):
                        X_pd = np.zeros(shape = (N , 2))
                        X_pd[0 : N , 0] = PD[ind_Mask]
                        X_pd[0 : N , 1] = av_PD[ind_Mask]
                    if (t1 != ''):
                        X_pd = np.zeros(shape = (N , 3))
                        X_pd[0 : N , 0] = PD[ind_Mask]
                        X_pd[0 : N , 1] = av_PD[ind_Mask]                        
                        X_pd[0 : N , 2] = PD[ind_Mask] / T1[ind_Mask]
                    X_pd = np.concatenate((X_pd , PD_Label_probability ) , axis = 1)                
                    
                if (flair != ''):
                    str_FLAIR = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    FLAIR = sitk.GetArrayFromImage(sitk.ReadImage(str_FLAIR))
                    av_FLAIR = warp_and_read_prior(path_av_flair, Label, nl_xfm, path_Temp + 'test_' +  str(i) + '_' + str(K) + '_tmp_flair.mnc')
                    FLAIR[FLAIR < 1] = 1
                    FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                    FLAIR_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        FLAIR_Label_probability[:,nl] = FLAIR_PDF_Label[np.round(FLAIR[ind_Mask]).astype(np.int32),nl]
                    N = len(FLAIR_Label_probability)
                    if (t1 == ''):
                        X_flair = np.zeros(shape = (N , 3))
                        X_flair[0 : N , 0] = FLAIR[ind_Mask]
                        X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                    if (t1 != ''):
                        X_flair = np.zeros(shape = (N , 4))
                        X_flair[0 : N , 0] = FLAIR[ind_Mask]
                        X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                        X_flair[0 : N , 2] = FLAIR[ind_Mask] / T1[ind_Mask]
                    X_flair = np.concatenate((X_flair , FLAIR_Label_probability ) , axis = 1)
                
                else:
                    X = np.zeros(shape = (N , 0))
                    X = np.concatenate((X , spatial_priors) , axis = 1)    
                if (t1 != ''):
                    X = np.concatenate((X , X_t1) , axis = 1)
                if (t2 != ''):
                    X = np.concatenate((X , X_t2) , axis = 1)
                if (pd != ''):
                    X = np.concatenate((X , X_pd) , axis = 1)
                if (flair != ''):
                    X = np.concatenate((X , X_flair) , axis = 1)
                        
                Y = np.zeros(shape = (N , ))
                Y[0 : N] = WMT[ind_Mask]
                Binary_Output = clf.predict(X)
                for nl in range(0 , n_labels):
                    Kappa[subject,nl] = 2 * np.sum((Y==(nl+1)) * (Binary_Output==(nl+1))) / (np.sum(Y==(nl+1)) + np.sum(Binary_Output==(nl+1)))
                    ID_Subject[subject] = ID_Test[i]
                if (np.sum(Y) + np.sum(Binary_Output)) == 0:
                    Kappa[subject] = 1
                print(Kappa[subject])
                subject = subject + 1
                        
                WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0,:]) , len(Mask[0 , 0 , :])),dtype=np.int32)
                WMT_auto[ind_Mask] = Binary_Output[0 : N]
                
                inputImage = sitk.ReadImage(str_Mask)
                result_image = sitk.GetImageFromArray(WMT_auto)
                result_image.CopyInformation(inputImage)
                sitk.WriteImage(result_image,  path_output + os.sep +  Classifier + '_' + str_Test + '_Label.mnc')             
        
        print('Cross Validation Successfully Completed. \nKappa Values:\n')        
        print(Kappa)
        print('Indices')
        print(Indices_G) 
        print(('Mean Kappa: ' + str(np.mean(Kappa)) + ' - STD Kappa: ' + str(np.std(Kappa))))  
###########################################################################################################################################################################    
    elif ClassificationMode == 'TT':
        K=0
        Indices_G=np.ones(shape = (len(IDs) , 1))
        ID_Train = IDs
        XFM_Files_Train = XFM_Files    
        Label_Files_Train = Label_Files   
        Mask_Files_Train = Mask_Files  
        if (t1 != ''):
            T1_Files_Train = T1_Files
        if (t2 != ''):        
            T2_Files_Train = T2_Files
        if (pd != ''):
            PD_Files_Train = PD_Files
        if (flair != ''):
            FLAIR_Files_Train = FLAIR_Files
        if doPreprocessingf:
            doPreprocessing(path_nlin_mask,path_Temp, IDs, Label_Files , Label, T1_Files , t1 , T2_Files , t2 , PD_Files , pd , FLAIR_Files, flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair)
            [IDs, XFM_Files, xfmf, Mask_Files, maskf, T1_Files, t1, T2_Files, t2, PD_Files, pd, FLAIR_Files, flair, Label_Files, Label] = get_addressess(path_Temp+'Preprocessed.csv')

        [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, Label_Files_Test, Label] = get_addressess(TestList)

        n_features=n_labels           
        if (t1 != ''):        
            [T1_Files_Train , tmp] = get_Train_Test(Indices_G , K , T1_Files)
            if os.path.exists(path_output+os.sep + 'T1_Label.pkl'):
                T1_PDF_Label = joblib.load(path_output+os.sep +'T1_Label.pkl')
            else:
                T1_PDF_Label = Calculate_Tissue_Histogram(T1_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                draw_histograms(T1_PDF_Label,path_output+os.sep +'T1_Label.png',"T1w")
            n_features = n_features + n_labels + 2
                
        if (t2 != ''):
            [T2_Files_Train , tmp] = get_Train_Test(Indices_G , K , T2_Files)
            if os.path.exists(path_output+os.sep +'T2_Label.pkl'):
                T2_PDF_Label=joblib.load(path_output+os.sep +'T2_Label.pkl')
            else:    
                T2_PDF_Label = Calculate_Tissue_Histogram(T2_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                draw_histograms(T2_PDF_Label,path_output+os.sep +'T2_Label.png',"T2w")
            n_features = n_features + n_labels + 3
                
        if (pd != ''):
            [PD_Files_Train , tmp] = get_Train_Test(Indices_G , K , PD_Files) 
            if os.path.exists(path_output+os.sep +'PD_Label.pkl'):
                PD_PDF_Label=joblib.load(path_output+os.sep +'PD_Label.pkl')
            else:
                PD_PDF_Label = Calculate_Tissue_Histogram(PD_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                draw_histograms(PD_PDF_Label,path_output+os.sep +'PD_Label.png',"T1w")
            n_features = n_features + n_labels + 3
                
        if (flair != ''):
            [FLAIR_Files_Train , tmp] = get_Train_Test(Indices_G , K , FLAIR_Files)
            if os.path.exists(path_output+os.sep +'FLAIR_Label.pkl'):
                PD_PDF_Label=joblib.load(path_output+os.sep +'FLAIR_Label.pkl')
            else:
                FLAIR_PDF_Label = Calculate_Tissue_Histogram(FLAIR_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                draw_histograms(FLAIR_PDF_Label,path_output+os.sep +'FLAIR_Label.png',"FLAIR")
            n_features = n_features + n_labels + 3            
                       
        path_sp = path_nlin_files + 'SP_'
            
        X_All = np.empty(shape = (0 , n_features ) , dtype = float , order = 'C')
        Y_All = np.empty(shape = (0 , ) , dtype = np.int32 , order = 'C')
    
        for i in range(0 , len(ID_Train)):
            str_Train = str(ID_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print(('Extracting The Features: Subject: ID = ' + str_Train))
                
            str_Mask = str(Mask_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
            ind_Mask = (Mask > 0)
            N=int(np.sum(Mask))
            Label = str(Label_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'') 
            WMT = sitk.GetArrayFromImage(sitk.ReadImage(Label))
                
            nl_xfm = str(XFM_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
            for nl in range(0 , n_labels):
                spatial_prior = warp_and_read_prior(f"{path_sp}{nl+1}.mnc", Label, nl_xfm, f"{path_Temp}train_{i}_{K}_sp_{nl+1}.mnc")
                spatial_priors[0:N,nl] = spatial_prior[ind_Mask]
                
            if (t1 != ''):
                str_T1 = str(T1_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T1 = sitk.GetArrayFromImage(sitk.ReadImage(str_T1))
                av_T1 = warp_and_read_prior(path_av_t1, Label,nl_xfm, f"{path_Temp}train_{i}_{K}_av_t1.mnc")
                T1[T1 < 1] = 1
                T1[T1 > (image_range - 1)] = (image_range - 1)
                T1_Label_probability = np.empty(shape = (N, n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    T1_Label_probability[:,nl] = T1_PDF_Label[np.round(T1[ind_Mask]).astype(np.int32),nl]
                X_t1 = np.zeros(shape = (N , 2))
                X_t1[0 : N , 0] = T1[ind_Mask]
                X_t1[0 : N , 1] = av_T1[ind_Mask]
                X_t1 = np.concatenate((X_t1 , T1_Label_probability) , axis = 1)
    
            if (t2 != ''):                
                str_T2 = str(T2_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T2 = sitk.GetArrayFromImage(sitk.ReadImage(str_T2))
                av_T2 = warp_and_read_prior(path_av_t2,Label,nl_xfm, f"{path_Temp}train_{i}_{K}_av_t2.mnc")
                T2[T2 < 1] = 1
                T2[T2 > (image_range - 1)] = (image_range - 1)
                T2_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    T2_Label_probability[:,nl] = T2_PDF_Label[np.round(T2[ind_Mask]).astype(np.int32),nl]
                if (t1 == ''):
                    X_t2 = np.zeros(shape = (N , 2))
                    X_t2[0 : N , 0] = T2[ind_Mask]
                    X_t2[0 : N , 1] = av_T2[ind_Mask]
                if (t1 != ''):
                    X_t2 = np.zeros(shape = (N , 3))
                    X_t2[0 : N , 0] = T2[ind_Mask]
                    X_t2[0 : N , 1] = av_T2[ind_Mask]    
                    X_t2[0 : N , 2] = T2[ind_Mask] / T1[ind_Mask]
                        
                X_t2 = np.concatenate((X_t2 , T2_Label_probability) , axis = 1)
                    
            if (pd != ''):
                str_PD = str(PD_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                PD = sitk.GetArrayFromImage(sitk.ReadImage(str_PD))
                av_PD = warp_and_read_prior(path_av_pd, Label,nl_xfm, f"{path_Temp}train_{i}_{K}_av_pd.mnc")
                PD[PD < 1] = 1
                PD[PD > (image_range - 1)] = (image_range - 1)
                PD_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    PD_Label_probability[:,nl] = PD_PDF_Label[np.round(PD[ind_Mask]).astype(np.int32),nl]
                if (t1 == ''):
                    X_pd = np.zeros(shape = (N , 2))
                    X_pd[0 : N , 0] = PD[ind_Mask]
                    X_pd[0 : N , 1] = av_PD[ind_Mask]
                if (t1 != ''):
                    X_pd = np.zeros(shape = (N , 3))
                    X_pd[0 : N , 0] = PD[ind_Mask]
                    X_pd[0 : N , 1] = av_PD[ind_Mask]                        
                    X_pd[0 : N , 2] = PD[ind_Mask] / T1[ind_Mask]
                        
                X_pd = np.concatenate((X_pd , PD_Label_probability ) , axis = 1)                
                    
            if (flair != ''):
                str_FLAIR = str(FLAIR_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                FLAIR = sitk.GetArrayFromImage(sitk.ReadImage(str_FLAIR))
                av_FLAIR = warp_and_read_prior(path_av_flair, Label,nl_xfm, f"{path_Temp}train_{i}_{K}_av_flair.mnc")
                FLAIR[FLAIR < 1] = 1
                FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                FLAIR_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    FLAIR_Label_probability[:,nl] = FLAIR_PDF_Label[np.round(FLAIR[ind_Mask]).astype(np.int32),nl]
                if (t1 == ''):
                    X_flair = np.zeros(shape = (N , 3))
                    X_flair[0 : N , 0] = FLAIR[ind_Mask]
                    X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                if (t1 != ''):
                    X_flair = np.zeros(shape = (N , 4))
                    X_flair[0 : N , 0] = FLAIR[ind_Mask]
                    X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                    X_flair[0 : N , 2] = FLAIR[ind_Mask] / T1[ind_Mask]
                        
                X_flair = np.concatenate((X_flair , FLAIR_Label_probability ) , axis = 1)
                
            X = np.zeros(shape = (N , 0))
            X = np.concatenate((X , spatial_priors) , axis = 1)  
            if (t1 != ''):
                X = np.concatenate((X , X_t1) , axis = 1)
            if (t2 != ''):
                X = np.concatenate((X , X_t2) , axis = 1)
            if (pd != ''):
                X = np.concatenate((X , X_pd) , axis = 1)
            if (flair != ''):
                X = np.concatenate((X , X_flair) , axis = 1)
                
            X_All = np.concatenate((X_All , X) , axis = 0)
            Y = np.zeros(shape = (N , ),dtype=np.int32)
            Y[0 : N , ] = (WMT[ind_Mask])    
            Y_All = np.concatenate((Y_All , Y) , axis = 0)
            
        print('Training The Classifier ...')
        clf = clf.fit(X_All , Y_All)
        print('Training Successfully Completed.')        

        saveFlag=1

        if saveFlag == 1:
            print('Saving the Classifier ...')  
            path_trained_classifiers = path_output 
            path_save_classifier = path_trained_classifiers + os.sep + Classifier + '_T1'+t1+'_T2'+t2+'_PD'+pd+'_FLAIR'+flair+'.pkl'    
            

            joblib.dump(clf,path_save_classifier)
            if (t1 != ''):
                joblib.dump(T1_PDF_Label,path_trained_classifiers+os.sep+'T1_Label.pkl')
            if (t2 != ''):
                joblib.dump(T2_PDF_Label,path_trained_classifiers+os.sep+'T2_Label.pkl')
            if (pd != ''):
                joblib.dump(PD_PDF_Label,path_trained_classifiers+os.sep+'PD_Label.pkl')
            if (flair != ''):
                joblib.dump(FLAIR_PDF_Label,path_trained_classifiers+os.sep+'FLAIR_Label.pkl')
            print("Trained Classifier Successfully Saved in: ",path_trained_classifiers)
            sys.exit()
            
        for i in range(0 , len(ID_Test)):
            str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print(('Segmenting Volumes: Subject: ID = ' + str_Test))
            str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'') .replace(" ",'')
            Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
            ind_Mask = (Mask > 0)
            nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            N=int(np.sum(Mask))
              
            if (t1 != ''):
		
                str_T1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(f"{path_sp}{nl+1}.mnc", str_T1, nl_xfm, f"{path_Temp}test_{i}_{K}_sp_{nl+1}.mnc")
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]
                
                T1 = sitk.GetArrayFromImage(sitk.ReadImage(str_T1))
                av_T1 = warp_and_read_prior(path_av_t1, str_T1, nl_xfm, f"{path_Temp}test_{i}_{K}_av_t1.mnc")
                T1[T1 < 1] = 1
                T1[T1 > (image_range - 1)] = (image_range - 1)
                T1_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    T1_Label_probability[:,nl] = T1_PDF_Label[np.round(T1[ind_Mask]).astype(np.int32),nl]
                N = len(T1_Label_probability)
                X_t1 = np.zeros(shape = (N , 2))
                X_t1[0 : N , 0] = T1[ind_Mask]
                X_t1[0 : N , 1] = av_T1[ind_Mask]
                X_t1 = np.concatenate((X_t1 , T1_Label_probability) , axis = 1)
    
            if (t2 != ''):                
                str_T2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T2 = sitk.GetArrayFromImage(sitk.ReadImage(str_T2))
                av_T2 = warp_and_read_prior(path_av_t2, str_T2, nl_xfm, f"{path_Temp}test_{i}_{K}_av_t2.mnc")
                T2[T2 < 1] = 1
                T2[T2 > (image_range - 1)] = (image_range - 1)
                T2_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    T2_Label_probability[:,nl] = T2_PDF_Label[np.round(T2[ind_Mask]).astype(np.int32),nl]
                N = len(T2_Label_probability)
                if (t1 == ''):
                    X_t2 = np.zeros(shape = (N , 2))
                    X_t2[0 : N , 0] = T2[ind_Mask]
                    X_t2[0 : N , 1] = av_T2[ind_Mask]
                if (t1 != ''):
                    X_t2 = np.zeros(shape = (N , 3))
                    X_t2[0 : N , 0] = T2[ind_Mask]
                    X_t2[0 : N , 1] = av_T2[ind_Mask]    
                    X_t2[0 : N , 2] = T2[ind_Mask] / T1[ind_Mask]                 
                X_t2 = np.concatenate((X_t2 , T2_Label_probability) , axis = 1)
                    
            if (pd != ''):
                str_PD = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                PD = sitk.GetArrayFromImage(sitk.ReadImage(str_PD))
                av_T2 = warp_and_read_prior(path_av_pd, str_PD, nl_xfm, f"{path_Temp}test_{i}_{K}_av_pd.mnc")
                PD[PD < 1] = 1
                PD[PD > (image_range - 1)] = (image_range - 1)
                PD_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    PD_Label_probability[:,nl] = PD_PDF_Label[np.round(PD[ind_Mask]).astype(np.int32),nl]
                N = len(PD_Label_probability)
                if (t1 == ''):
                    X_pd = np.zeros(shape = (N , 2))
                    X_pd[0 : N , 0] = PD[ind_Mask]
                    X_pd[0 : N , 1] = av_PD[ind_Mask]
                if (t1 != ''):
                    X_pd = np.zeros(shape = (N , 3))
                    X_pd[0 : N , 0] = PD[ind_Mask]
                    X_pd[0 : N , 1] = av_PD[ind_Mask]                        
                    X_pd[0 : N , 2] = PD[ind_Mask] / T1[ind_Mask]  
                X_pd = np.concatenate((X_pd , PD_Label_probability ) , axis = 1)                
                    
            if (flair != ''):
                str_FLAIR = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                FLAIR = sitk.GetArrayFromImage(sitk.ReadImage(str_FLAIR))
                av_FLAIR = warp_and_read_prior(path_av_flair, str_FLAIR, nl_xfm, f"{path_Temp}test_{i}_{K}_av_flair.mnc")
                FLAIR[FLAIR < 1] = 1
                FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                FLAIR_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    FLAIR_Label_probability[:,nl] = FLAIR_PDF_Label[np.round(FLAIR[ind_Mask]).astype(np.int32),nl]
                N = len(FLAIR_Label_probability)
                if (t1 == ''):
                    X_flair = np.zeros(shape = (N , 3))
                    X_flair[0 : N , 0] = FLAIR[ind_Mask]
                    X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                if (t1 != ''):
                    X_flair = np.zeros(shape = (N , 4))
                    X_flair[0 : N , 0] = FLAIR[ind_Mask]
                    X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                    X_flair[0 : N , 2] = FLAIR[ind_Mask] / T1[ind_Mask]     
                X_flair = np.concatenate((X_flair , FLAIR_Label_probability ) , axis = 1)
                
            else:
                X = np.zeros(shape = (N , 0))
                X = np.concatenate((X , spatial_priors) , axis = 1)
            if (t1 != ''):
                X = np.concatenate((X , X_t1) , axis = 1)
            if (t2 != ''):
                X = np.concatenate((X , X_t2) , axis = 1)
            if (pd != ''):
                X = np.concatenate((X , X_pd) , axis = 1)
            if (flair != ''):
                X = np.concatenate((X , X_flair) , axis = 1)
   
            Y = np.zeros(shape = (N , ), dtype=np.int32)
            Binary_Output = clf.predict(X)       
            Prob_Output=clf.predict_proba(X)            
            #### Saving results #########################################################################################################################            
            WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])),dtype=np.int32)
            WMT_auto[ind_Mask] = Binary_Output[0 : N]
            
            str_Labelo= path_output +os.sep + Classifier + '_' + str_Test
            
            inputImage = sitk.ReadImage(str_Mask)
            result_image = sitk.GetImageFromArray(WMT_auto)
            result_image.CopyInformation(inputImage)
            sitk.WriteImage(result_image,  str_Labelo + '_Label.mnc')
            
            Prob_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])))
            Prob_auto[ind_Mask] = Prob_Output[0 : N,1]
            
            result_image = sitk.GetImageFromArray(Prob_auto)
            result_image.CopyInformation(inputImage)
            sitk.WriteImage(result_image,  str_Labelo + '_P.mnc')
            
            if (t1 != ''):            
                new_command = 'minc_qc.pl ' + str_T1 + ' --mask ' + str_Labelo + '_Label.mnc ' + str_Labelo + '_Label.jpg --big --clobber --spectral-mask  --image-range 0 200 --mask-range 0 ' + str(n_labels)
                os.system(new_command)
            if (t2 != ''): 
                new_command = 'minc_qc.pl ' + str_T2  +' '+ str_Labelo + '_T2.jpg --big --clobber  --image-range 0 200 --mask-range 0 ' + str(n_labels)
                os.system(new_command)
            if (pd != ''): 
                new_command = 'minc_qc.pl ' + str_PD +' '+ str_Labelo + '_PD.jpg --big --clobber  --image-range 0 200 --mask-range 0 ' + str(n_labels)
                os.system(new_command)
            if (flair != ''): 
                new_command = 'minc_qc.pl ' + str_FLAIR+' '+ str_Labelo + '_FLAIR.jpg --big --clobber  --image-range 0 200 --mask-range 0 ' + str(n_labels)
                os.system(new_command)
###########################################################################################################################################################################    
    elif ClassificationMode == 'PT':
        path_sp   =path_trained_classifiers + os.sep + 'SP_'
        path_av_t1=path_trained_classifiers + os.sep + 'Av_T1.mnc'
        path_av_t2=path_trained_classifiers + os.sep + 'Av_T2.mnc'
        path_av_pd=path_trained_classifiers + os.sep + 'Av_PD.mnc'
        path_av_flair=path_trained_classifiers+os.sep + 'Av_FLAIR.mnc'
        
        [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, Label_Files_Test, Label] = get_addressess( TestList )
        path_saved_classifier = path_trained_classifiers + os.sep + Classifier+'_T1'+t1+'_T2'+t2+'_PD'+pd+'_FLAIR'+flair+'.pkl'
############## Preprocessing ####################################################################################################################################
        if doPreprocessingf:
            doPreprocessing(path_nlin_mask,path_Temp, ID_Test, Label_Files_Test , Label, T1_Files_Test , t1 , T2_Files_Test , t2 , PD_Files_Test , pd , FLAIR_Files_Test , flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair)
            [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, Label_Files_Test, Label] = get_addressess(path_Temp+'Preprocessed.csv')
########## Loading Trained Classifier ##########################################################################################################################            
        print('Loading the Pre-trained Classifier from: ' + path_saved_classifier)
        clf = joblib.load(path_saved_classifier)
        # set maximum jobs to run in parallel
        clf.n_jobs = n_jobs

        K=0
        if (t1 != ''):
            T1_PDF_Label=joblib.load(path_trained_classifiers+os.sep +'T1_Label.pkl')
        if (t2 != ''):
            T2_PDF_Label=joblib.load(path_trained_classifiers+os.sep +'T2_Label.pkl')
        if (pd != ''):
            PD_PDF_Label=joblib.load(path_trained_classifiers+os.sep +'PD_Label.pkl')
        if (flair != ''):
            FLAIR_PDF_Label=joblib.load(path_trained_classifiers+os.sep +'FLAIR_Label.pkl')

        for i in range(0 , len(ID_Test)):
            str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print(('Segmenting Volumes: Subject: ID = ' + str_Test))
            str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'') .replace(" ",'')
            Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
            ind_Mask = (Mask > 0)
            nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            N=int(np.sum(Mask))
            print('Extracting The Features ...')
            if (t1 != ''):
                str_T1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(f"{path_sp}{nl+1}.mnc", str_T1, nl_xfm, f"{path_Temp}test_{i}_{K}_sp_{nl+1}.mnc")
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]

                T1 = sitk.GetArrayFromImage(sitk.ReadImage(str_T1))
                av_T1 = warp_and_read_prior(path_av_t1, str_T1, nl_xfm, f"{path_Temp}test_{i}_{K}_av_t1.mnc")
                T1[T1 < 1] = 1
                T1[T1 > (image_range - 1)] = (image_range - 1)
                T1_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    T1_Label_probability[:,nl] = T1_PDF_Label[np.round(T1[ind_Mask]).astype(np.int32),nl]
                N = len(T1_Label_probability)
                X_t1 = np.zeros(shape = (N , 2))
                X_t1[0 : N , 0] = T1[ind_Mask]
                X_t1[0 : N , 1] = av_T1[ind_Mask]
                X_t1 = np.concatenate((X_t1 , T1_Label_probability) , axis = 1)
    
            if (t2 != ''):
                str_T2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T2 = sitk.GetArrayFromImage(sitk.ReadImage(str_T2))
                av_T2 = warp_and_read_prior(path_av_t2, str_T2, nl_xfm, f"{path_Temp}test_{i}_{K}_av_t2.mnc")
                T2[T2 < 1] = 1
                T2[T2 > (image_range - 1)] = (image_range - 1)
                T2_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    T2_Label_probability[:,nl] = T2_PDF_Label[np.round(T2[ind_Mask]).astype(np.int32),nl]
                N = len(T2_Label_probability)
                if (t1 == ''):
                    X_t2 = np.zeros(shape = (N , 2))
                    X_t2[0 : N , 0] = T2[ind_Mask]
                    X_t2[0 : N , 1] = av_T2[ind_Mask]
                if (t1 != ''):
                    X_t2 = np.zeros(shape = (N , 3))
                    X_t2[0 : N , 0] = T2[ind_Mask]
                    X_t2[0 : N , 1] = av_T2[ind_Mask]
                    X_t2[0 : N , 2] = T2[ind_Mask] / T1[ind_Mask]
                X_t2 = np.concatenate((X_t2 , T2_Label_probability) , axis = 1)

            if (pd != ''):
                str_PD = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                PD = sitk.GetArrayFromImage(sitk.ReadImage(str_PD))
                av_PD = warp_and_read_prior(path_av_pd, str_PD, nl_xfm, f"{path_Temp}test_{i}_{K}_av_pd.mnc")
                PD[PD < 1] = 1
                PD[PD > (image_range - 1)] = (image_range - 1)
                PD_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    PD_Label_probability[:,nl] = PD_PDF_Label[np.round(PD[ind_Mask]).astype(np.int32),nl]
                N = len(PD_Label_probability)
                if (t1 == ''):
                    X_pd = np.zeros(shape = (N , 2))
                    X_pd[0 : N , 0] = PD[ind_Mask]
                    X_pd[0 : N , 1] = av_PD[ind_Mask]
                if (t1 != ''):
                    X_pd = np.zeros(shape = (N , 3))
                    X_pd[0 : N , 0] = PD[ind_Mask]
                    X_pd[0 : N , 1] = av_PD[ind_Mask]
                    X_pd[0 : N , 2] = PD[ind_Mask] / T1[ind_Mask]
                X_pd = np.concatenate((X_pd , PD_Label_probability ) , axis = 1)

            if (flair != ''):
                str_FLAIR = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                FLAIR = sitk.GetArrayFromImage(sitk.ReadImage(str_FLAIR))
                av_FLAIR = warp_and_read_prior(path_av_flair, str_FLAIR, nl_xfm, f"{path_Temp}test_{i}_{K}_av_flair.mnc")
                FLAIR[FLAIR < 1] = 1
                FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                FLAIR_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    FLAIR_Label_probability[:,nl] = FLAIR_PDF_Label[np.round(FLAIR[ind_Mask]).astype(np.int32),nl]
                N = len(FLAIR_Label_probability)
                if (t1 == ''):
                    X_flair = np.zeros(shape = (N , 3))
                    X_flair[0 : N , 0] = FLAIR[ind_Mask]
                    X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                if (t1 != ''):
                    X_flair = np.zeros(shape = (N , 4))
                    X_flair[0 : N , 0] = FLAIR[ind_Mask]
                    X_flair[0 : N , 1] = av_FLAIR[ind_Mask]
                    X_flair[0 : N , 2] = FLAIR[ind_Mask] / T1[ind_Mask]
                X_flair = np.concatenate((X_flair , FLAIR_Label_probability ) , axis = 1)

            else:
                X = np.zeros(shape = (N , 0))
                X = np.concatenate((X , spatial_priors) , axis = 1)
            if (t1 != ''):
                X = np.concatenate((X , X_t1) , axis = 1)
            if (t2 != ''):
                X = np.concatenate((X , X_t2) , axis = 1)
            if (pd != ''):
                X = np.concatenate((X , X_pd) , axis = 1)
            if (flair != ''):
                X = np.concatenate((X , X_flair) , axis = 1)
            
            Y = np.zeros(shape = (N , ))
            print("Applying The Classifier ...")
            Binary_Output = clf.predict(X)
            #### Saving results #########################################################################################################################            
            WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])), dtype=np.int32)
            WMT_auto[ind_Mask] = Binary_Output[0 : N]
            str_Labelo = path_output + os.sep + Classifier + '_' + str_Test 
            
            inputImage = sitk.ReadImage(str_Mask)
            result_image = sitk.GetImageFromArray(WMT_auto)
            result_image.CopyInformation(inputImage)
            sitk.WriteImage(result_image,  str_Labelo + '_Label.mnc')
            run_command('mincresample ' + str_Labelo + '_Label.mnc -like ' + str_T1 + ' ' + str_Labelo + '_Labelr.mnc')
            if nifti:
                run_command('mnc2nii ' + str_Labelo + '_Label.mnc ' + str_Labelo + '_Label.nii')
            
            if (t1 != ''):            
                run_command('minc_qc.pl ' + str_T1 + ' --mask ' + str_Labelo + '_Labelr.mnc ' + str_Labelo + '_Label.jpg --big --clobber --spectral-mask  --image-range 0 200 --mask-range 0 ' + str(n_labels))
            if (t2 != ''): 
                run_command('minc_qc.pl ' + str_T2  +' '+ str_Labelo + '_T2.jpg --big --clobber  --image-range 0 200 --mask-range 0 ' + str(n_labels))
            if (pd != ''): 
                run_command('minc_qc.pl ' + str_PD +' '+ str_Labelo + '_PD.jpg --big --clobber  --image-range 0 200 --mask-range 0 ' + str(n_labels))
            if (flair != ''): 
                run_command('minc_qc.pl ' + str_FLAIR+' '+ str_Labelo + '_FLAIR.jpg --big --clobber  --image-range 0 200 --mask-range 0 ' + str(n_labels))

    print('Segmentation Successfully Completed. ')

if __name__ == "__main__":
   main(sys.argv[1:])   

