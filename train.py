#import main libraries

#processing & viz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data splitting
from sklearn.model_selection import train_test_split

#load data
df = pd.read_csv("predictive_maintenance.csv")

#rename headers& data cleanup
df = df.rename(columns={'Product ID':'product_id', 'Air temperature [K]':'air_temperature', 'Process temperature [K]':'process_temperature', 'Rotational speed [rpm]':'rot_speed', 'Torque [Nm]':'torque','Tool wear [min]':'tool_wear','Failure Type':'fail_type','Type':'type','Target':'target' })
del df['UDI']

#data split
#split according to random state = 1
df_trainval, df_test = train_test_split(df, test_size=0.2, random_state=1) #split train+val [80%], test [20%]
df_train, df_val = train_test_split(df_trainval, test_size=0.25, random_state=1,  stratify=df_trainval["fail_type"].values) #split train[60%] val [20%]

#reset, drop index
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_test = df_test["fail_type"].values
y_train = df_train["fail_type"].values
y_val = df_val["fail_type"].values

del df_test["fail_type"]
del df_train["fail_type"]
del df_val["fail_type"]


del df_test["target"]
del df_train["target"]
del df_val["target"]

#libraries
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False) #import encoder

from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

#auc
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay



# Serialize non-numerical data using DictVectorizer (aka One-Hot Encoding)
train_dict = df_train.to_dict(orient='records') # get categorial variables from train db, sort them by x and put them into dictionary 
X_train = dv.fit_transform(train_dict) #one-hot encoding 
val_dict = df_val.to_dict(orient='records') #apply same 
X_val = dv.transform(val_dict)


#decision tree
dt = make_pipeline(RobustScaler(),
                DecisionTreeClassifier(max_depth=20,criterion='gini',splitter='random'))
dt.fit(X_train,y_train)


# decision forest
rf = RandomForestClassifier(n_estimators=59, max_depth=25, n_jobs=-1, random_state=1)
rf.fit(X_train,y_train)



# SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
clf = make_pipeline(RobustScaler(),
                     LinearSVC(random_state=0, tol=1e-2))
clf.fit(X_train, y_train)

#linear
rc = RidgeClassifier(alpha=400, normalize=False)
rc.fit(X_train,y_train)


#Stack
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
kf.get_n_splits(X_val,y_val)

from sklearn.ensemble import StackingClassifier
stck = StackingClassifier(
     estimators=[
       ('dt', dt), ('rf', rf),('svc',clf)], final_estimator=rf,cv=kf)
stck = stck.fit(X_train, y_train)       


print("Class Accuracy: %2.3f \n Val Accuracy: %2.3f" %( sum(y_train==stck.predict(X_train))/len(y_train)*100,  sum(y_val==stck.predict(X_val))/len(y_val)*100))

# export pickle
import pickle
with open('model.bin','wb') as f: pickle.dump(stck,f)
with open("dv.bin", 'wb') as f: pickle.dump(dv,f)