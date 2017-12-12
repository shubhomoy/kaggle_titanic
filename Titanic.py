#!/usr/bin/python

__author__ = 'Shubhomoy Biswas'

import warnings

import pandas as pd

warnings.filterwarnings('ignore')

# importing train and test csv files
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')

# combine train and test set for one time cleaning
combine = [dataset, testset]

# Checking null values
for data in combine:
    print(data.isnull().sum())

# Checking how many survived
dataset['Survived'].value_counts()

# Female and male passenger count
dataset.groupby('Sex')['PassengerId'].count()

# Comparing female and male survival rate
dataset.groupby(['Sex', 'Survived'])['Survived'].count()

# Comparing survival ratio with respect to Pclass
pd.crosstab(dataset.Pclass, dataset.Survived, margins=True)

# Combining the above two and analyse!
pd.crosstab([dataset.Sex, dataset.Survived], dataset.Pclass, margins=True)


# Need to fill NaN values.
# Lets see how we can fill the age.
print 'Min age {}'.format(dataset['Age'].min())
print 'Max age {}'.format(dataset['Age'].max())
print 'Mean age {}'.format(dataset['Age'].mean())

# Extracting the initials from the name. This may help to determine age from Initials
for data in combine:
    data['Initials'] = data.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(dataset['Sex'], dataset['Initials'])
# hmm..got the sex from uncommon initials

# We don't need so many initials for now. Let's group together some initials
for data in combine:
    data['Initials'].replace(['Dona', 'Miss', 'Mlle', 'Mme', 'Countess', 'Lady', 'Countess', 'Ms'],
                             ['Mrs', 'Ms', 'Mrs', 'Mrs', 'Mrs', 'Ms', 'Ms', 'Ms'], inplace=True)
    
    data['Initials'].replace(['Sir', 'Don', 'Jonkheer', 'Rev'],
                             ['Mr', 'Mr', 'Mr', 'Mr'], inplace=True)
    
    data['Initials'].replace(['Capt', 'Col', 'Dr', 'Major'],
                             ['Other', 'Other', 'Other', 'Other'], inplace=True)
                             
# Now let's see the difference!
pd.crosstab(dataset['Sex'], dataset['Initials'])
pd.crosstab(testset.Sex, testset.Initials)

# Checking the mean age of these Initials
dataset.groupby('Initials')['Age'].mean()

# Setting the age of Null values according to the mean age categorized by initials
for data in combine:
    data.loc[(data.Age.isnull())&(data.Initials=='Master'), 'Age'] = 5
    data.loc[(data.Age.isnull())&(data.Initials=='Mr'), 'Age'] = 33
    data.loc[(data.Age.isnull())&(data.Initials=='Mrs'), 'Age'] = 36
    data.loc[(data.Age.isnull())&(data.Initials=='Ms'), 'Age'] = 22
    data.loc[(data.Age.isnull())&(data.Initials=='Other'), 'Age'] = 48

# Just for fun and ofcourse analysing!
dataset.groupby(['Initials', 'Survived'])['Pclass'].count()


# Now we need to fill null values for embarked.<br />
# First let's see how Embarked is related with the prediction of survival
pd.crosstab([dataset.Embarked, dataset.Pclass], [dataset.Sex, dataset.Survived], margins=True)
pd.crosstab(dataset.Embarked, dataset.Survived, margins=True)

# We see that most of the passengers boarded from 'S', thus filling null values of Embarked with 'S'
# Easy way out!
for data in combine:
    data['Embarked'].fillna('S', inplace=True)

# Now let's check how parents and siblings help in survival prediction.
# First with siblings
pd.crosstab(dataset.SibSp, dataset.Survived, margins=True)
pd.crosstab(dataset.SibSp, dataset.Pclass, margins=True)
# We can see that having 2-3 siblings helped in survival.

# Now having parents...
pd.crosstab(dataset.Parch, [dataset.Pclass, dataset.Survived], margins=True)
# Looks like same result as that of siblings.

# Now let's check fare
print 'Highest fare is {}'.format(dataset['Fare'].max())
print 'Lowest fare is {}'.format(dataset['Fare'].min())
print 'Average fare is {}'.format(dataset['Fare'].mean())

# Finding correlation of all the above factors
dataset.corr()


# Categorizing continuous values
# Let's categorize age first into 5 categories
for data in combine:
    data['Age_band'] = 0
    data.loc[dataset['Age']<=16, 'Age_band'] = 0
    data.loc[(data['Age']>16)&(data['Age']<=32), 'Age_band'] = 1
    data.loc[(data['Age']>32)&(data['Age']<=48), 'Age_band'] = 2
    data.loc[(data['Age']>48)&(data['Age']<=64), 'Age_band'] = 3
    data.loc[(data['Age']>64)&(data['Age']<=80), 'Age_band'] = 4

pd.crosstab(dataset.Age_band, dataset.Survived, margins=True)


# Feature engineering
# Create new feature and drop unnecessary features.

# Let's combine Parch and SibSp into Family_size
for data in combine:
    data['Family_size'] = data['Parch'] + data['SibSp']
    data['Alone'] = 0
    data.loc[data['Family_size'] == 0, 'Alone'] = 1

pd.crosstab(dataset.Family_size, [dataset.Survived, dataset.Pclass], margins=True)


# Since fare is also a continuous feature, we can categorize it into 4 parts
for data in combine:
    data['Fare_range'] = 0
    data['Fare_range'] = pd.qcut(data['Fare'], 4)

dataset.groupby('Fare_range')['Survived'].mean()

for data in combine:
    data['Fare_cat'] = 0
    data.loc[data['Fare']<=7.91, 'Fare_cat'] = 0
    data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454), 'Fare_cat'] = 1
    data.loc[(data['Fare']>14.454)&(data['Fare']<=31.0), 'Fare_cat'] = 2
    data.loc[(data['Fare']>31.0)&(data['Fare']<=512.329), 'Fare_cat'] = 3

dataset.groupby('Fare_cat')['Survived'].mean()
# We can see that more expensive fare, lead to more survival probability

# Transform string to numbers. Since strings are scary!
for data in combine:
    data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    data['Initials'].replace(['Master', 'Mr', 'Mrs', 'Ms', 'Other'], [0, 1, 2, 3, 4], inplace=True)

# Time to free up space!
# Dropping unnecessary features.
dataset.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_range', 'PassengerId'], axis=1, inplace=True)
testset.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_range'], axis=1, inplace=True)

dataset.corr()

# Let's check the shape
print(dataset.shape)
print(testset.shape)

# Prediction!

# Here we go!
# Importing necessary libs
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Split data into train and test set
train, test = train_test_split(dataset, test_size=0.3, random_state=0, stratify=dataset['Survived'])
train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]

# Feature scaling is a breeze :P
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.fit_transform(test_X)

X = dataset[dataset.columns[1:]]
Y = dataset['Survived']

# ### Logistic Regression
model = LogisticRegression()
model.fit(train_X, train_Y)
prediction_1 = model.predict(test_X)
print 'Accuracy for Logistic Regression {}'.format(metrics.accuracy_score(prediction_1, test_Y))


# ### Linear - SVM
model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
model.fit(train_X, train_Y)
prediction_2 = model.predict(test_X)
print 'Accuracy of Linear SVM {}'.format(metrics.accuracy_score(prediction_2, test_Y))


# ### Radial - SVM
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(train_X, train_Y)
prediction_3 = model.predict(test_X)
print 'Accuracy of Radial - SVM {}'.format(metrics.accuracy_score(prediction_3, test_Y))


# ### Decision Tree
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
prediction_4 = model.predict(test_X)
print 'Accuracy of Decision Tree {}'.format(metrics.accuracy_score(prediction_4, test_Y))


# ### Random forest
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X, train_Y)
prediction_5 = model.predict(test_X)
print 'Accuracy of Random forest with 100 estimators {}'.format(metrics.accuracy_score(prediction_5, test_Y))

# Hmm..SVM seems to give higher accuracy. Let's try giving possible values for C and gamma
C = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
kernel = ['rbf', 'linear']
hyper = {
    'kernel': kernel,
    'C': C,
    'gamma': gamma
}
# gd = GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
# t_X = sc.fit_transform(X)
# gd.fit(t_X, Y)
# print gd.best_score_
# print gd.best_estimator_

from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)),
                                               ('RBF', svm.SVC(probability=True, kernel='rbf', C=1, gamma=0.1)),
                                               ('RFor', RandomForestClassifier(n_estimators=100, random_state=0)),
                                               ('LR', LogisticRegression(C=0.05))], voting='soft').fit(train_X, train_Y)
print 'Accuracy of ensemble model is {}'.format(ensemble_lin_rbf.score(test_X, test_Y))

# ## Evaluation and Submission
# Using Radial - SVM
sc = StandardScaler()
test = testset.loc[:, testset.columns != 'PassengerId']
test = sc.fit_transform(test)

# model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
# model.fit(train_X, train_Y)
prediction_final = ensemble_lin_rbf.predict(test)

df = pd.DataFrame.from_records(zip(testset['PassengerId'].values, prediction_final), columns=['PassengerId', 'Survived'])
df.to_csv('submit.csv', index=False)

