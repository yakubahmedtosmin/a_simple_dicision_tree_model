import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print (train.head())
print (test.head())

train.describe()

datadict = pd.DataFrame(train.dtypes)
datadict

datadict['MissingVal'] = train.isnull().sum()
datadict

datadict['NUnique']=train.nunique()
datadict

datadict['Count']=train.count()
datadict

train.shape

print(train["Survived"].value_counts())
print(train["Survived"].value_counts(normalize = True))
print(train["Survived"][train["Sex"] == 'male'].value_counts())
print(train["Survived"][train["Sex"] == 'female'].value_counts())

train["Child"] = float('NaN')
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

print(train.columns.values)

from sklearn import tree

#for DT
train["Age"] = train["Age"].fillna(train["Age"].median())
train.head()
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train.head()

# Target and Features : target, features1
target = train["Survived"].values
features1 = train[["Pclass", "Sex", "Age", "Fare"]].values

# DT1: tree_one
tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(features1, target)

print(tree_one.feature_importances_)
print(tree_one.score(features1, target))

train.describe(include=['object'])
train.describe(include=['number'])
 
test.isnull().sum()

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Extract  TEST SET: Pclass, Sex, Age, and Fare.
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# Prediction TEST SET
prediction = tree_one.predict(test_features)
print(prediction)

# DF : PassengerId & Survived. Survived for predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# DF
print(my_solution.shape)

my_solution.describe(include=['number'])
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


#VISUALIZATION ON TRAIN SET DATA
train.Survived.value_counts(normalize=True)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived',data=train,ax=axes[0,0])
sns.countplot('Pclass',data=train,ax=axes[0,1])
sns.countplot('Sex',data=train,ax=axes[0,2])
sns.countplot('SibSp',data=train,ax=axes[0,3])
sns.countplot('Parch',data=train,ax=axes[1,0])
sns.countplot('Embarked',data=train,ax=axes[1,1])
sns.distplot(train['Fare'], kde=True,ax=axes[1,2])
sns.distplot(train['Age'].dropna(),kde=True,ax=axes[1,3])
