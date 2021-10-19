# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load dataset
train_data = pd.read_csv("../Dataset/titanic/train.csv")
test_data = pd.read_csv("../Dataset/titanic/test.csv")


# split data in features and target variable
def dataset_splitter(data):
    feature_cols = []
    for feature_col in data.head(0):
        if feature_col == "Survived":
            continue
        else:
            feature_cols.append(feature_col)
    return feature_cols


# copy original dataset in case we need it latter
original_train = train_data.copy()

full_data = [train_data, test_data]

# feature that tells whether a passenger had a cabin on the titanic
train_data["Has_Cabin"] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_data["Has_Cabin"] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

