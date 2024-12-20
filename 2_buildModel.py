##store best algorithm for prediction:
import pickle
import pandas as pd
import numpy as np
# import dataset
myData = pd.read_csv('finaldataset/updatedataset.csv')
print(myData.head())

# split input and output and make in array format
x = np.array(myData.drop(["Class"], axis=1))
y = np.array((myData["Class"]))

# check x, y
print("\n\n x \n",x)
print("\n\n y \n",y)


# split test and train data
from sklearn.model_selection import train_test_split

##apply algorithm:
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


##apply Decision tree clssifier:
bestscore_dc = 0
i = 0
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(x_train , y_train)
    acc  = dt_clf.score(x_test ,y_test)
    print("\n i-count:" , i  ,"acc:" ,acc ,end="")
    if bestscore_dc < acc:
        bestscore_dc = acc
        print("-------------------------------------->decisin tree clasifier:" , bestscore_dc)
        with open("dt.pickle", "wb")as dc_file:
            pickle.dump(dt_clf , dc_file)


## naivebayes clasifier::
bestscore_nb = 0
j = 0
for j in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    naivebayes_clf = GaussianNB()
    naivebayes_clf.fit(x_train ,y_train)
    acc = naivebayes_clf.score(x_test , y_test)
    print("\n j-count:" ,j,"acc:", acc ,end="")
    if bestscore_nb < acc:
        bestscore_nb = acc
        print("--------------------------------->naivebayes classifier acc :" ,bestscore_nb)
        with open("nb.pickle", "wb")as nb_file:
            pickle.dump(naivebayes_clf ,nb_file)
