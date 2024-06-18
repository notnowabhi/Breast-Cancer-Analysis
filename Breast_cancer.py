import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression() # initializing the learning model/algorithm

# reading the csv file as a Panda Dataframe

df = pd.read_csv("breast-cancer.csv")
df.drop(["id"], axis = 1, inplace = True)

# the cleaning and analysis of the dataset was done prior.

df["diagnosis"] = np.where(df["diagnosis"].str.contains("M"), 1, 0)

# separating labels from the dataset

dflabel = df.copy()
df.drop('diagnosis', axis = 1, inplace = True)
dflabel.drop(df[df.columns[0:31]], axis = 1, inplace = True)

# performing the train_test_split the test size being 30% of the data

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df, dflabel, test_size = 0.3)

# fitting both the features and labels to the regressor initialized above

regressor.fit(X_train, Y_train)

# accuracy for the model 

score = regressor.score(X_test,Y_test)
print("score = ", score)
