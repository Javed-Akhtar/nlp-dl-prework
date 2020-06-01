# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
#Reading dataset
df = pd.read_csv(path)

#display
print(df.head())

#Storing all independent features in X
X = df.copy()
X.drop(['list_price'],axis=1,inplace=True)

#Storing target variable in y
y = df['list_price']

#Split dataframe into X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
#Storing all The X_train columns in cols
cols = X_train.columns

#Create 3*3 subplot
fig,axes = plt.subplots(nrows=3,ncols=3)

#To subplot scatter plots for loops are used
for i in range(3):
    for j in range(3):
        col=cols[i*3+j]
        axes[i][j].scatter(X[col],y)

# code ends here



# --------------
# Reduce feature redundancies

#Creating correlation table of X_train
corr = X_train.corr()
print(corr)

#dropping columns from X_train and X_test which are having a correlation higher than (+/-)0.75
X_train.drop(['play_star_rating','val_star_rating','val_star_rating'],axis=1,inplace=True)
X_test.drop(['play_star_rating','val_star_rating','val_star_rating'],axis=1,inplace=True)




# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Price Prediction

#Instantiate linear regression model
regressor = LinearRegression()

#Fitting model on the X_train and y_train
regressor.fit(X_train,y_train)

#Making predictions on the X_test
y_pred = regressor.predict(X_test)

#Finding MSE
mse = mean_squared_error(y_test,y_pred)
print(mse)

#Finding r^2 score
r2 = r2_score(y_test,y_pred)
print(r2)




# --------------
#Residual Check

#calculate residual
residual = y_test - y_pred
#print(residual)

#Histogram making
plt.hist(residual)

#display histogram
plt.show()



