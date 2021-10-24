# Load libraries
# Imports needed for the script
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

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


# remove variables from dataset
def variable_remover(label_list, data):
    for label in label_list:
        data = data.drop(label, axis=1)
    return data


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

    # Mapping Age
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[dataset["Age"] > 64, "Age"] = 4

# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train_data = variable_remover(drop_elements, train_data)
test_data = variable_remover(drop_elements, test_data)

color_map = plt.cm.viridis
plt.figure(figsize=(12, 12))
plt.title("Person Correlation of Features", y=1.05, size=15)
# what is sns.heatmap => notion
sns.heatmap(train_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=color_map, linecolor='white',
            annot=True)
# plt.show()


# Since "Survived is a binary class, these metrics grouped by the Title feature represent:
# MEAN: survival rate
# COUNT: total observations
# SUM: people survived

# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
# print(train_data[["Title", "Survived"]].groupby(["Title"], as_index=False).agg(["mean", "count", "sum"]))

# sex_mapping = {{'female': 0, 'male': 1}}
# print(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum']))


# I use copy() again to prevent modifications in out original_train dataset
title_and_sex = original_train.copy()[['Name', 'Sex']]

# Create 'Title' feature
title_and_sex['Title'] = title_and_sex['Name'].apply(get_title)

# Map 'Sex' as binary feature
title_and_sex['Sex'] = title_and_sex['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Table with 'Sex' distribution grouped by 'Title'
# print(title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum']))


# Define function to calculate Gini Impurity
def get_gini_impurity(survived_count, total_count):
    random_observation_survived_prob = survived_count/total_count
    random_observation_not_survived_prob = 1 - random_observation_survived_prob
    mislabelling_survived_prob = random_observation_not_survived_prob * random_observation_survived_prob
    mislabelling_not_survived_prob = random_observation_survived_prob * random_observation_not_survived_prob
    gini_impurity = mislabelling_survived_prob + mislabelling_not_survived_prob
    return gini_impurity


gini_impurity_starting_node = get_gini_impurity(342, 891)
# print("gini_impurity_starting_node" + str(gini_impurity_starting_node))

gini_impurity_men = get_gini_impurity(109, 577)
# print("gini_impurity_men: " + str(gini_impurity_men))

gini_impurity_women = get_gini_impurity(233, 314)
# print("gini_impurity_women: " + str(gini_impurity_women))

# Gini impurity decrease if node is split by Sex
men_weight = 577 / 891
women_weight = 314 / 891
weight_gini_impurity_sex_split = (gini_impurity_men * men_weight) + (gini_impurity_women * women_weight)

sex_gini_decrease = weight_gini_impurity_sex_split - gini_impurity_starting_node
# print("sex_gini_decrease: " + str(sex_gini_decrease))

gini_impurity_title_1 = get_gini_impurity(81, 517)
# print("gini_impurity_title_1: " + str(gini_impurity_title_1))

gini_impurity_title_others = get_gini_impurity(261, 374)
# print("gini_impurity_title_others" + str(gini_impurity_title_others))

# Gini Impurity decrease if node is split for observations with Title == 1 == Mr
title_1_weight = 517 / 891
title_others_weight = 374 / 891
weight_gini_impurity_title_split = (gini_impurity_title_1 * title_1_weight) + \
                                   (gini_impurity_title_others * title_others_weight)
title_gini_decrease = weight_gini_impurity_title_split - gini_impurity_starting_node
# print("title_gini_decrease: " + str(title_gini_decrease))

# Desired number of Cross Validation folds
cv = KFold(n_splits=10)

accuracies = list()
max_attributes = len(list(test_data))
depth_range = range(1, max_attributes + 1)

# Testing max_depths from 1 to max attributes
# comment out prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth=depth)
    # print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(train_data):
        # Extract train data with cv indices
        f_train = train_data.loc[train_fold]
        # Extract valid data with cv indices
        f_valid = train_data.loc[valid_fold]

        # I fit the model with the fold train data
        model = tree_model.fit(X=f_train.drop(["Survived"], axis=1), y=f_train["Survived"])

        # I calculate accuracy with the fold validation data
        valid_acc = model.score(X=f_valid.drop(["Survived"], axis=1), y=f_valid["Survived"])
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n")

# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
# print(df.to_string(index=False))

# Create Numpy arrays of train, test and Survived dataframes to feed into our models
y_train = train_data["Survived"]
x_train = train_data.drop(["Survived"], axis=1).values
x_test = test_data.values

# Create Decision Tree with max_depth = 3
decision_tree = tree.DecisionTreeClassifier(max_depth=3)
decision_tree.fit(x_train, y_train)

# Predicting results for test dataset
y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({
    "PassengerId": full_data[1]["PassengerId"].values,
    "Survived": y_pred
})

submission.to_csv("submission.csv", index=False)

# Export our trainded model as a .dot fole
with open("tree1.dot", "w") as f:
    f = tree.export_graphviz(decision_tree, out_file=f, max_depth=3, impurity=True,
                             feature_names=list(train_data.drop(["Survived"], axis=1)), class_names=["Died", "Survived"],
                             rounded=True, filled=True)

# Convert .dot to .png to allow display in web notebook
check_call(["dot", "-Tpng", "tree1.dot", "-o", "tree1.png"])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
draw.text((10, 0), "\"Title <= 1.5\" corresponds to \"Mr.\" title", (0, 0, 255))
img.save("titanic_decision_tree_depth_3.png")
PImage("titanic_decision_tree_depth_3.png")

