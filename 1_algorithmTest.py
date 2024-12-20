import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression    ##for linear_regression
from sklearn.linear_model import LogisticRegression    ##for logistic regression
from sklearn.neighbors import KNeighborsRegressor       ##for knn regression
from sklearn.tree import DecisionTreeRegressor          ##for DecisionTreeRegression


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# import dataset
myData = pd.read_csv('finaldataset/updatedataset.csv')
print(myData.head())

##data columns:
print("\n\n column names: \n" , myData.columns)

# split input and output and make in array format
x = np.array(myData.drop(["Class"], axis=1))
y = np.array((myData["Class"]))

# check x, y
print("\n\n x \n",x)
print("\n\n y \n",y)


# split test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# check train and test data length
print("x_test :",  len(x_test))
print("y_test :",  len(y_test))
print("x_trian :", len(x_train))
print("y_trian :", len(y_train))


# applay algorithm
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
acc =  linear_regression.score(x_test, y_test)
print("Linar Regression : ", acc)

##apply logistic regression:

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train , y_train)
acc = logistic_regression.score(x_test , y_test)
print("logistic regression acc  :" , acc )

##apply k-nearest neighbours:
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train , y_train)
acc = knn.score(x_test , y_test)
print("knn acc : " , acc)


##apply DecisionTreeRegressor
dtregression = DecisionTreeRegressor()
dtregression.fit(x_train , y_train)
acc = dtregression.score(x_test , y_test)
print("decision tree regression acc :" , acc)


"""
###not executing / taking 2,3hrs for execution:
##apply Random forest regression:
rf_regression = RandomForestRegressor()
rf_regression.fit(x_train , y_train)
acc = rf_regression.score(x_test, y_test)
print("random forest regression acc :" , acc)
"""


###    classifiers:   ####

print("---------------classifier---------------")

##apply Decision tree clssifier:
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train , y_train)
acc  = dt_clf.score(x_test ,y_test)
print("decisin tree clasifier:" , acc)



##apply random_forest classifier:
rf_clf = RandomForestClassifier(n_estimators=10 , criterion="entropy")
rf_clf.fit(x_train , y_train)
acc = rf_clf.score(x_test ,y_test)
print("randomforest_clf :" , acc)




## naivebayes clasifier::
naivebayes_clf = GaussianNB()
naivebayes_clf.fit(x_train ,y_train)
acc = naivebayes_clf.score(x_test , y_test)
print("naivebayes classifier acc :" ,acc)

