import torch
import numpy as np
import pandas 


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer

def clean_sex(val):
    if val != "male" and val != "female":
        return np.nan
    if val == "man":
        return "male"
    if val == "woman":
        return "female"

    # dummy numbers
    try:
        # wil raise exception if it is a string 
        int(val)
        return np.nan
    except:
        return val
    

train = pandas.read_csv("titanic.csv", sep=",")


data = train["Age"].to_numpy().reshape(train.shape[0],1)

sex = train["Sex"]
sex = sex.apply(clean_sex)
sex = sex.to_numpy().reshape(train.shape[0],1)

# Cleaning 
# Simple
clean_missing_values = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
# KNN
# clean_missing_values = KNNImputer(missing_values= np.nan, n_neighbors= 2)

sex = clean_missing_values.fit_transform(sex)
print(sex)

# Label Encoding
# label_encoding = LabelEncoder()
ordinal_encoder = OrdinalEncoder()
sex_encoded = ordinal_encoder.fit_transform(sex)

print(sex_encoded)



 

