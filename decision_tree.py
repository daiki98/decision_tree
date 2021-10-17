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


# training data
x_train = dataset_splitter(train_data)
y_train = train_data["Survived"]

print(x_train[0])
# test data
x_test = dataset_splitter(test_data)

# create decision tree classifier object
classifier = DecisionTreeClassifier()

# train decision tree classifier
# classifier = classifier.fit(x_train, y_train)

# predict the response for test dataset
# y_prediction = classifier.predict(x_test)
