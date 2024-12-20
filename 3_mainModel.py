###import pickle model
import pickle
import pandas as pd
import numpy as np
# import dataset
pred = pd.read_csv('finaldataset/pred.csv')
print(pred.head())

###convert to array format:
pred = pred.drop(columns='Class')
pred_test = np.array(pred)

##import pickle
pickle_in = open("dt.pickle" ,"rb")
model = pickle.load(pickle_in)

###prediction for x-test:
print("\n\n prediction: " , model.predict(pred_test))