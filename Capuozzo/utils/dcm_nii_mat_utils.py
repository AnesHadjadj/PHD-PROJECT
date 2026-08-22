import os
from glob import glob

import dicom2nifti
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import numpy
from tqdm import tqdm

import torch
#import torchio as tio
import torchvision
import skimage
from skimage import color

from torch.utils.data import DataLoader
import nibabel as nib

from scipy import io
from math import floor, ceil
import pandas as pd

from utils.dataset_utils import *

from dicom2nifti.exceptions import ConversionError


def get_patients_list_from_dcm_folder(dicom_input):
    folder = glob(dicom_input + '/*')

    pat_list = list()

    for patient_id in tqdm(folder):
        mri_types = glob(patient_id+'/*')
        for patient in mri_types:
            name = os.path.basename(os.path.dirname(patient))
            #print("NAME: "+name)
            #print("PATIENT: "+patient)
            dir_name = os.path.basename(patient)
            dicom_input = dicom2nifti.common.read_dicom_directory(patient)
            pat_list.append(name+"_"+dir_name+"_"+str(dicom_input[0].PixelSpacing))
            #print(name, dicom_input[0].PixelSpacing)
            #print(dicom_input[0].SpacingBetweenSlices)
    pat_list.sort()
    return pat_list

# def get_nii_from_dcm_folder(dicom_input, nifti_output):   #dorigine
#     file_error = list()

#     # Convert DICOM to NIFTI by reporting errors

#     folder = glob(dicom_input + '/*')

#     for patient_id in tqdm(folder):
#         mri_types = glob(patient_id+'/*')
#         for patient in mri_types:
#             name = os.path.basename(os.path.dirname(patient))
#             #print("NAME: "+name)
#             #print("PATIENT: "+patient)
#             dir_name = os.path.basename(patient)
#             try:
#                 #dicom_input = dicom2nifti.common.read_dicom_directory(patient)
#                 #print(dicom_input[0])
#                 if not os.path.exists(nifti_output):
#                     os.mkdir(nifti_output)
#                 if not os.path.exists(nifti_output+"/"+dir_name):
#                     os.mkdir(nifti_output+"/"+dir_name)
#                 dicom2nifti.dicom_series_to_nifti(patient, os.path.join(nifti_output, dir_name, name + '.nii.gz'))
#             except dicom2nifti.exceptions.ConversionValidationError as err:
#                 print("Error with "+name+" in "+dir_name)
#                 print(err)
#                 file_error.append(name+" - "+dir_name+" - "+str(err))
#                 pass 
#             #except Exception as err:
#                 #print("Error with "+name+" in "+dir_name)
#                 #print(err)
#                 #file_error.append(name+" - "+dir_name)
#                 #pass

#     print(file_error)
    
#     num_list = list()
#     for elem in file_error:
#         num_list.append(elem.split(" - ")[0])

#     folder = glob(nifti_output + '/*')
#     for mri_type in folder:
#         patients = glob(mri_type + '/*')
#         for patient in patients:
#             if os.path.basename(patient)[:5] in num_list:
#                 print("Moving "+os.path.basename(patient))
#                 os.rename(patient, nifti_output+"_removed/"+os.path.basename(mri_type)+"/"+os.path.basename(patient))



#____________________________________________________________________________________________
def get_nii_from_dcm_folder(dicom_input, nifti_output):    #ta3i
    file_error = []

    folder = glob(dicom_input + '/*')

    for patient_id in tqdm(folder):
        mri_types = glob(patient_id + '/*')

        for patient in mri_types:
            name = os.path.basename(os.path.dirname(patient))
            dir_name = os.path.basename(patient)

            try:
                # créer les dossiers si nécessaires
                if not os.path.exists(nifti_output):
                    os.mkdir(nifti_output)

                out_dir = os.path.join(nifti_output, dir_name)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                out_file = os.path.join(out_dir, name + '.nii.gz')

                # ✅ NE PAS refaire si déjà converti
                if os.path.exists(out_file):
                    print(f"[OK] Déjà converti : {name} ({dir_name})")
                    continue

                dicom2nifti.dicom_series_to_nifti(
                    patient,
                    out_file
                )

            except dicom2nifti.exceptions.ConversionError as err:
                print(f"[SKIPPED] {name} ({dir_name}) -> {err}")
                file_error.append(name + " - " + dir_name + " - " + str(err))
                continue

            except Exception as err:
                print(f"[ERROR] {name} ({dir_name}) -> {err}")
                file_error.append(name + " - " + dir_name + " - " + str(err))
                continue

    print(file_error)

#____________________________________________________________________________________________



def group_nii_files_by_class(nifti_output,out_dict):
    folder = glob(nifti_output + '/*')

    for mri_type in folder:
        if not os.path.exists(mri_type+"/0"):
            os.makedirs(mri_type+"/0")
        if not os.path.exists(mri_type+"/1"):
            os.makedirs(mri_type+"/1")
        patients = glob(mri_type + '/*')
        for patient in patients:
            if os.path.basename(patient) != "0" and os.path.basename(patient) != "1":
                value = out_dict[os.path.basename(patient)[:5]]
                print("Moving "+os.path.basename(patient)+" into "+value)
                os.rename(patient, mri_type+"/"+value+"/"+os.path.basename(patient))
                
def get_patients_list_from_nii_folder(nifti_output):
    pat_list_2 = list()

    folder = glob(nifti_output + '/*')
    for patient_id in folder:
        mri_types = glob(patient_id+'/*')
        for out in mri_types:
            o = glob(out+'/*')
            for patient in tqdm(o):
                nii = nib.load(patient)
                dims = nii.get_fdata().shape
                #print(dims)
                ss = nii.header.get_zooms()
                pat_list_2.append(os.path.basename(patient)[:5]+"_"+os.path.basename(patient_id)+"_"+str(ss)+"_"+str(dims))
                #print(os.path.basename(patient_id), os.path.basename(patient),sx, sy, sz)

    pat_list_2.sort()
    #for elem in pat_list_2:
    #    print(elem)
    return pat_list_2

def group_nii_files_by_unknown_class(nifti_output):
    folder = glob(nifti_output + '/*')

    for mri_type in folder:
        if not os.path.exists(mri_type+"/unknown"):
            os.makedirs(mri_type+"/unknown")
        patients = glob(mri_type + '/*')
        for patient in patients:
            if os.path.basename(patient) != "unknown":
                print("Moving "+os.path.basename(patient)+" into unknown")
                os.rename(patient, mri_type+"/unknown/"+os.path.basename(patient))
                
def isometric_numpy_transform(sample, scale_factors, img_size):
    #print(scale_factors)
    #print(sample.shape)
    # With Tensor3D Transformation, idx 0 becomes 2, idx 1 becomes 0 and idx 2 becomes 1
    sample = sample - np.min(sample)
    sample = sample[:,:,:,np.newaxis]
    s1 = int(sample.shape[0]*scale_factors[0])
    s2 = int(sample.shape[1]*scale_factors[1])
    s3 = int(sample.shape[2]*scale_factors[2])
    transform = transforms.Compose([ToTensor3D(),
                                    Resize3D(size=(s3,s1,s2)),
                                    Rotation3D90Degrees(times=2, axis=1),
                                    Rotation3D90Degrees(times=2, axis=3),
                                    Pad3D(size=(img_size,img_size,img_size))
    ])
    sample = transform(sample)
    sample = sample.squeeze(0).numpy()
    return sample

def load_nifti_image(path, img_size=256):
    nifti = nib.load(path)
    data = nifti.get_fdata().astype(np.float32)
    scale_factors = nifti.header.get_zooms()
    data = isometric_numpy_transform(data, scale_factors, img_size)
    
    return data
                
"""
def save_nii_images_3d_as_mat(            #on a modifié, voir en bas
    folder,
    mri_type,
    case_id,
    target,
    main_path,
    img_size=256
):
    case_id = str(case_id).zfill(5)
    
    if folder == "test":
        target = "unknown"

    nii_path = f"{main_path}/{folder}_nii/{mri_type}/{target}/{case_id}.nii.gz"
    
    if not os.path.exists(f"{main_path}/{folder}_mat/{mri_type}/{target}/{case_id}_{mri_type}.mat"):
        img3d = np.around(load_nifti_image(nii_path)).astype(np.int16)

        data = {'X': img3d, 'y': target}
        if not os.path.exists(f"{main_path}/{folder}_mat"):
            os.makedirs(f"{main_path}/{folder}_mat")
        if not os.path.exists(f"{main_path}/{folder}_mat/{mri_type}"):
            os.makedirs(f"{main_path}/{folder}_mat/{mri_type}")  
        if not os.path.exists(f"{main_path}/{folder}_mat/{mri_type}/{target}"):
            os.makedirs(f"{main_path}/{folder}_mat/{mri_type}/{target}")  
        if not os.path.exists(f"{main_path}/{folder}_mat/{mri_type}/{target}/{case_id}_{mri_type}.mat"):
            io.savemat(f"{main_path}/{folder}_mat/{mri_type}/{target}/{case_id}_{mri_type}.mat", data, do_compression=True)
"""
            
def import_train_and_test_from_csv(train_csv, test_csv):
    data = {}
    data["train"] = pd.read_csv(train_csv)
    data["test"] = pd.read_csv(test_csv)
    return data

"""
def convert_nii_data_to_mat(data, main_path):  #on a modifié, voir en bas
    missing = {}
    missing["train"] = []
    missing["test"] = []
    for folder,value in data.items():
        for index in tqdm(range(len(value))):
            row = value.loc[index]
            case_id = int(row.BraTS21ID)
            target = int(row.MGMT_value)
            case_id_str = str(case_id).zfill(5)
            if folder == "test":
                target_str = "unknown"
            else:
                target_str = target
            if os.path.exists(f"{main_path}/{folder}_nii/FLAIR/{target_str}/{case_id_str}.nii.gz"):
                for mri_type in ["FLAIR", "T1w", "T1wCE", "T2w"]:
                    #save_dicom_images_3d_as_mat(folder,mri_type,case_id,target)
                    save_nii_images_3d_as_mat(folder,mri_type,case_id,target,main_path)
            else:
                missing[folder].append(case_id_str)

    print("Missing train ids:")
    print(missing["train"])
    print("Missing test ids:")
    print(missing["test"])"""



def save_nii_images_3d_as_mat(folder, mri_type, case_id, target, main_path, img_size=256):
    """
    Convertit un fichier NIfTI en .mat si le fichier existe et n'a pas déjà été converti.
    """
    case_id_str = str(case_id).zfill(5)
    
    if folder == "test":
        target_str = "unknown"
    else:
        target_str = str(target)

    nii_path = f"{main_path}/{folder}_nii/{mri_type}/{target_str}/{case_id_str}.nii.gz"
    mat_path = f"{main_path}/{folder}_mat/{mri_type}/{target_str}/{case_id_str}_{mri_type}.mat"

    if not os.path.exists(nii_path):
        print(f"[SKIPPED] {nii_path} -> NIfTI file not found")
        return

    if os.path.exists(mat_path):
        print(f"[SKIPPED] {mat_path} -> .mat already exists")
        return

    # Charger et convertir
    img3d = np.around(load_nifti_image(nii_path, img_size)).astype(np.int16)
    data = {'X': img3d, 'y': target}

    # Créer dossiers si nécessaire
    os.makedirs(os.path.dirname(mat_path), exist_ok=True)

    io.savemat(mat_path, data, do_compression=True)
    print(f"[DONE] {mat_path} created")


def convert_nii_data_to_mat(data, main_path, img_size=256):
    """
    Parcourt le dataset, convertit tous les NIfTI existants en .mat.
    Ignore les fichiers manquants ou déjà traités.
    """
    missing = {"train": [], "test": []}

    for folder, df in data.items():
        for index in tqdm(range(len(df))):
            row = df.loc[index]
            case_id = int(row.BraTS21ID)
            target = int(row.MGMT_value) if folder == "train" else "unknown"

            # Vérifie l'existence du NIfTI FLAIR pour décider de continuer
            nii_path_check = f"{main_path}/{folder}_nii/FLAIR/{target}/{str(case_id).zfill(5)}.nii.gz"
            if not os.path.exists(nii_path_check):
                missing[folder].append(str(case_id).zfill(5))
                print(f"[SKIPPED] {nii_path_check} -> FLAIR NIfTI missing")
                continue

            # Convertit toutes les séquences MRI
            for mri_type in ["FLAIR", "T1w", "T1wCE", "T2w"]:
                save_nii_images_3d_as_mat(folder, mri_type, case_id, target, main_path, img_size)

    print("\n=== Résumé des fichiers manquants ===")
    print("Missing train ids:", missing["train"])
    print("Missing test ids:", missing["test"])



