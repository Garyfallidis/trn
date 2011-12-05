import os
import glob

home_dir=os.path.expanduser('~')

data_dir=home_dir+'/Data/dipy_data/'

Resources={0:{'tag':'simulation phantom bezier SNR 30 2 curves analyze',\
            'path':[data_dir+'Marta/Software_Phantoms/bezier_phantom_2curves_SNR30_1.4_0.35_0.35.hdr']},\
               
           1:{'tag':'simulation phantom bezier SNR 30 2 lines analyze',\
            'path':[data_dir+'Marta/Software_Phantoms/bezier_phantom_2lines_SNR30_1.4_0.35_0.35.hdr']},\
               
           2:{'tag':'simulation phantom voxels 1 fibre text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_1fibre']},

           3:{'tag':'simulation phantom voxels 1 fibre iso text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_1fibre+iso']},

           4:{'tag':'simulation phantom voxels 2 fibres iso 15 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres+iso_15deg']},

           5:{'tag':'simulation phantom voxels 2 fibres iso 30 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres+iso_30deg']},
           
           6:{'tag':'simulation phantom voxels 2 fibres iso 60 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres+iso_60deg']},

           7:{'tag':'simulation phantom voxels 2 fibres iso 90 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres+iso_90deg']},
          
           8:{'tag':'simulation phantom voxels 2 fibres 15 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres_15deg']},

           9:{'tag':'simulation phantom voxels 2 fibres 30 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres_30deg']},
           
           10:{'tag':'simulation phantom voxels 2 fibres 60 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres_60deg']},

           11:{'tag':'simulation phantom voxels 2 fibres 90 degrees text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_2fibres_90deg']},

           12:{'tag':'simulation phantom voxels isotropic text',\
            'path':[data_dir+'Marta/SimData/results_SNR030_isotropic']},
           
           13:{'tag':'real Eleftherios Marta CBU100891 MPRAGE dicom',\
             'path':[data_dir+'Eleftherios_Marta/CBU100891_METHODS/20100707_083026/Series_002_CBU_MPRAGE']},
           
           14:{'tag':'real Eleftherios Marta CBU100891 STEAM 2.5mm 104 directions DTI dicom Trio',\
             'path':[data_dir+'Eleftherios_Marta/CBU100891_METHODS/20100707_083026/Series_003_DTI_25x25x25_1b_104dir']},

           15:{'tag':'real Eleftherios Marta CBU100891 STEAM 2.5mm 96 directions DTI dicom Trio',\
             'path':[data_dir+'Eleftherios_Marta/CBU100891_METHODS/20100707_083026/Series_004_DTI_96_CBU_STEAM_25x25x25']},

           16:{'tag':'real Eleftherios Marta CBU100891 STEAM 2mm 64 directions DTI dicom Trio',\
             'path':[data_dir+'Eleftherios_Marta/CBU100891_METHODS/20100707_083026/Series_005_CBU_DTI_64InLea_2x2x2']},

           17:{'tag':'real Eleftherios Marta CBU100891 STEAM 2.5mm ? 101 directions DSI dicom Trio',\
             'path':[data_dir+'Eleftherios_Marta/CBU100891_METHODS/20100707_083026/Series_011_ep2d_advdiff_101dir_DSI']},

           18:{'tag':'real Eleftherios Marta CBU100899 STEAM 2.5mm 114 directions DTI dicom Trio',\
             'path':[data_dir+'Eleftherios_Marta/CBU100899_METHODS/20100708_112612/Series_002_DTI_25x25x25_STEAM_114dir']},
           
           19:{'tag':'real Eleftherios Marta CBU100891 STEAM 2.5mm ? 2x58 directions DTI dicom Trio',\
             'path':[data_dir+'Eleftherios_Marta/CBU100899_METHODS/20100708_112612/Series_003_DTI_25x25x25_STEAM_58dir_2bvals']},
           
           20:{'tag':'real Eleftherios Marta Guy T2 Structural DSI nifti Verio ',\
             'path':[data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0002_echo00.nii',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0002_echo01.nii']},

           21:{'tag':'real Eleftherios Marta Guy MPRAGE DSI nifti Verio ',\
             'path':[data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0003.nii']},

           22:{'tag':'real Eleftherios Marta Guy DSI 101 directions max b-value 3000 2.5mm 12channels nifti Verio',\
             'path':[data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0004.nii',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0004.bvals',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0004.bvecs']},

           23:{'tag':'real Eleftherios Marta Guy DSI 101 directions max b-value 3000 2.5mm 32channels nifti Verio',\
             'path':[data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0006.nii',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0006.bvals',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0006.bvecs']},

           24:{'tag':'real Eleftherios Marta Guy DSI 101 directions max b-value 4000 2mm 32channels nifti Verio',\
             'path':[data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0007.nii',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0007.bvals',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0007.bvecs']},

           25:{'tag':'real Eleftherios Marta Guy DSI 101 directions max b-value 4000 2.5mm 32channels nifti Verio',\
             'path':[data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0008.nii',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0008.bvals',\
                     data_dir+'Eleftherios_Guy_Marta/DSI_dataVerio/18620_0008.bvecs']},

           26:{'tag':'real Eleftherios Marta CBU100891 MPRAGE nifti Trio',\
             'path':[data_dir+'Eleftherios_Marta/CBU100891_METHODS/20100707_083026/Series_002_CBU_MPRAGE.nii']}}


           



def get_paths(query):
    
    query_names=query.split()
    query_result=[]
    
    for row in Resources: #for every line in the dictionary
        i=0
        for word in query_names: #for every word in query 
            if Resources[row]['tag'].find(word) >= 0:
                i+=1
        if i==len(query_names):
            for filename in Resources[row]['path']:
                query_result.append(row)
                query_result.append(Resources[row]['tag'])
                query_result.append(filename)
    return query_result


def get_unique_tags():
    tags=[]
    for row in Resources:
        for word in Resources[row]['tag'].split():
            tags.append(word)
    
    return list(set(tags))
