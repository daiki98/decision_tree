# Load libraries
import re

import pandas as pd
import numpy as np
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
train_data["HasCabin"] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_data["HasCabin"] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# create new feature family size as a combination of sibsp and parch
for dataset in full_data:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

# create new feature isAlone from FamilySize
for dataset in full_data:
    dataset["isAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "isAlone"] = 1


# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset["Embarked"] = dataset["Embarked"].fillna("S")

# Remove all NULLS in the Fare column value is "median of train_data fare"
for dataset in full_data:
    dataset["Fare"] = dataset["Fare"].fillna(train_data["Fare"].median())

# Remove all NULLS in the Age column
for dataset in full_data:

    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # np.isnan: If null is included, return true
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    # astype: change type to int
    dataset['Age'] = dataset['Age'].astype(int)


# Define function to extract titles from passenger name
def get_title(name):
    title_search = re.search(" ([A-Za-z]+)\.", name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset["Title"] = dataset["Name"].apply(get_title)

# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset["Title"] = dataset["Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
                                                "Sir", "Jonkheer", "Dona"], "Rare")

    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
    dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")


for dataset in full_data:
    # Mapping Sex
    dataset["Sex"] = dataset["Sex"].map( {"female": 0, "male": 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

    # Mapping Embarked
    dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 31, "Fare"] = 3
    dataset["Fare"] = dataset["Fare"].astype(int)



