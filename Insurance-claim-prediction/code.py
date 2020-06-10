# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
#loading data
df = pd.read_csv(path)
print(df.head())

#storing features in X
X = df.iloc[:,:-1]
#print(X.head())

#storing target variable in y
y = df.iloc[:,-1]
#print(y.head())

#spliting data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 6)



# --------------
import matplotlib.pyplot as plt


# Code starts here
#box plotting and display
plt.boxplot(df['bmi'])
plt.show()

#taking quantile of X_train['bmi']
q_value = X_train['bmi'].quantile(q=0.95)
print(q_value)

#check value counts of y_train
print(y_train.value_counts())


# --------------
import seaborn as sns
# Code starts here

#finding correlation between features stored in X_train
relation = X_train.corr()
print(relation)

#plot pairplot
sns.pairplot(X_train)
plt.show()



# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
#creating a list to store columns
cols = ['children','sex','region','smoker']

#creating subplot
fig, axes = plt.subplots(nrows = 2, ncols = 2)

#filling subplots
for i in range(2):
    for j in range(2):
        col = cols[i*2+j]
        sns.countplot(x = X_train[col],hue = y_train, ax = axes[i,j] )

#display
plt.show()



# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
#instantiate logistic regression model
lr = LogisticRegression(random_state = 9)

#making grid search model
grid = GridSearchCV(lr,param_grid=parameters)

#fitting values
grid.fit(X_train,y_train)

#predict class
y_pred = grid.predict(X_test)

#accuracy checking
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)



# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
#calculating roc_auc_score
score = roc_auc_score(y_test,y_pred)

#predict probability
y_pred_proba = grid.predict_proba(X_test)[:,1]
print(y_pred_proba)

#calculating fpr tpr
fpr,tpr,_ = metrics.roc_curve(y_test,y_pred)
#print(fpr,tpr)

#calculating roc_auc_score 
roc_auc = roc_auc_score(y_test,y_pred_proba)

#plot roc_curve
plt.plot(fpr,tpr,label = "Logistic model,auc=" + str(roc_auc))
plt.show()
# Code ends here


