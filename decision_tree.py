# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load dataset
train_data = pd.read_csv("../Dataset/titanic/train.csv")

# split dataset in features and target variable
feature_cols = []
for feature_col in train_data.head(0):
    if feature_col == "Survived":
        continue
    else:
        feature_cols.append(feature_col)
x = feature_cols
y = train_data["Survived"]
print(x)
print(y)