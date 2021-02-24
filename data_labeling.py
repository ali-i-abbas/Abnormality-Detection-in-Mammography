
import pandas as pd

#Labels for train and test data for both calc and mass cases

## Train data labels
calc_train = pd.read_csv("calc_case_description_train_set.csv")
calc_train['image_name'] = calc_train.patient_id + '_' + calc_train['left or right breast'] + '_' + calc_train['image view'] + '.png'
calc_train.drop(["image file path","cropped image file path","ROI mask file path"], axis=1, inplace=True)
calc_train.columns = ["Patient_ID","Breast_Density","Side_L_R","Image View","Abnormality_ID","Abnormality_Type","Mass_Shape","Mass_Margins","Assessment","Pathology", "Subtlety","Image_Name"]

mass_train = pd.read_csv("mass_case_description_train_set.csv")
mass_train['image_name'] = mass_train.patient_id + '_' + mass_train['left or right breast'] + '_' + mass_train['image view'] + '.png'
mass_train.drop(["image file path","cropped image file path","ROI mask file path"], axis=1, inplace=True)
mass_train.columns = ["Patient_ID","Breast_Density","Side_L_R","Image View","Abnormality_ID","Abnormality_Type","Mass_Shape","Mass_Margins","Assessment","Pathology", "Subtlety","Image_Name"]


## Test data labels
calc_test = pd.read_csv("calc_case_description_test_set.csv")
calc_test['image_name'] = calc_test.patient_id + '_' + calc_test['left or right breast'] + '_' + calc_test['image view'] + '.png'
calc_test.drop(["image file path","cropped image file path","ROI mask file path"], axis=1, inplace=True)
calc_test.columns = ["Patient_ID","Breast_Density","Side_L_R","Image View","Abnormality_ID","Abnormality_Type","Mass_Shape","Mass_Margins","Assessment","Pathology", "Subtlety","Image_Name"]

mass_test = pd.read_csv("mass_case_description_test_set.csv")
mass_test['image_name'] = mass_test.patient_id + '_' + mass_test['left or right breast'] + '_' + mass_test['image view'] + '.png'
mass_test.drop(["image file path","cropped image file path","ROI mask file path"], axis=1, inplace=True)
mass_test.columns = ["Patient_ID","Breast_Density","Side_L_R","Image View","Abnormality_ID","Abnormality_Type","Mass_Shape","Mass_Margins","Assessment","Pathology", "Subtlety","Image_Name"]


# Train and Test label
train_label = pd.concat([calc_train, mass_train], axis = 0)
train_label['Pathology'][train_label['Pathology'] == 'BENIGN_WITHOUT_CALLBACK'] = 'BENIGN'
train_label['Class'] = train_label['Pathology'] + '_' + train_label['Abnormality_Type']

test_label = pd.concat([calc_test, mass_test], axis = 0)
test_label['Pathology'][test_label['Pathology'] == 'BENIGN_WITHOUT_CALLBACK'] = 'BENIGN'
test_label['Class'] = test_label['Pathology'] + '_' + test_label['Abnormality_Type']

# Set image_name to be the index
train_label.set_index("Image_Name", inplace = True)
test_label.set_index("Image_Name", inplace = True)

#save labels
train_label.to_pickle("train_label.pkl")
test_label.to_pickle("test_label.pkl")

