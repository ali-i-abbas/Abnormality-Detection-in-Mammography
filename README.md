# Abnormality-Detection-in-Mammography

This is a fork from https://github.com/AidenFather/Abnormality-Detection-in-Mammography. I made some modification so it can be run with python scripts. Please follow the following instructions to run it:

1- Go to https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

Download Images (DICOM, 163.6GB) from

https://wiki.cancerimagingarchive.net/download/attachments/22516629/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia?version=1&modificationDate=1534787024127&api=v2

Download and Install NBIA Data Retriever from

https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images

Open the downloaded "CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia" with NBIA Data Retriever and start downloading the files

Make sure the files are downloaded to folder "CBIS-DDSM" in the same folder as the code

Download these Files from https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM and put them in the same folder as the code

Mass-Training-Description (csv)	https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_train_set.csv?version=1&modificationDate=1506796355038&api=v2
Calc-Training-Description (csv)	https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_train_set.csv?version=1&modificationDate=1506796349666&api=v2
Mass-Test-Description (csv)	https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_test_set.csv?version=1&modificationDate=1506796343175&api=v2
Calc-Test-Description (csv) https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_test_set.csv?version=1&modificationDate=1506796343686&api=v2

2- Run "cbis_ddsm_rename_dicom_files.py". This will delete files from "CBIS-DDSM" folder, if you think you might need the data later, make a backup copy of it somewhere else.

3- Run "data_labeling.py" and "cbis_ddsm_patch_extraction_256x256.py"

4- Download the following folders from DDSM dataset at http://www.eng.usf.edu/cvprg/Mammography/Database.html

normal_07
normal_08
normal_09
normal_10
normal_11
normal_12

Convert them to PNG format using this:
https://github.com/yodhcn/DDSM-LJPEG-Converter

Create a folder named "Normal" in the code folder. Move normal_07,normal_08, ... folders to "Normal" folder.
Your folder structure should look like this:
Normal/normal_07/case*
Normal/normal_08/case*
...

5- run "ddsm_patch_extraction_256x256.py". This will make changes to "Normal" folder, so if you need the original folder structures make a copy of it somewhere else.

6- run "data_preparation_for_cnn.py".

7- run "cnn_mammography_multi_train.py". This will train CNN Model No. 5.2 4 VGG Blocks with Dropout and BatchNormalization.
I changed batch_size=16 from 32 because I was getting out of memmory errors. With better GPU you can change it back to 32.

8- run "cnn_mammography_binary_train.py". This will train Binary Classification CNN Model No. 5.2 - 4 VGG Blocks with BatchNormalization, Droput, and Weight Decay.

9- run "cnn_mammography_multi_evaluation.py". This will run evaluation for CNN Model No. 5.2 4 VGG Blocks with Dropout and BatchNormalization.

10- run "cnn_mammography_binary_evaluation.py". This will run evaluation for Binary Classification CNN Model No. 5.2 - 4 VGG Blocks with BatchNormalization, Droput, and Weight Decay.

9- You can find all the models and evaluations in "cnn_development_and_evaluation.py" but I haven't cleaned it up from errors. You still can copy some useful code from it and try it in new scripts.




Notes:
- You need a strong PC with strong GPU to do deep learning. Also you need at least 16GB RAM and 250GB free Hard disk.
- If you have NVidia GPU, it's better to install CUDA. Follow this guide:
https://towardsdatascience.com/python-environment-setup-for-deep-learning-on-windows-10-c373786e36d1



