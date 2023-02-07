import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
sns.set()
import warnings
warnings.filterwarnings('ignore')

import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#PIMA diabetes dataset
diabetes_data = pd.read_csv('diabetes.csv') 

copy_ds=diabetes_data.copy()

copy_ds[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = copy_ds[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

def median_target(col):   
    temp = copy_ds[copy_ds[col].notnull()]
    temp = temp[[col, 'Outcome']].groupby(['Outcome'])[[col]].median().reset_index()
    return temp
columns = copy_ds.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    copy_ds.loc[(copy_ds['Outcome'] == 0 ) & (copy_ds[i].isnull()), i] = median_target(i)[i][0]
    copy_ds.loc[(copy_ds['Outcome'] == 1 ) & (copy_ds[i].isnull()), i] = median_target(i)[i][1]

from scipy import stats
import numpy as np
#z-score provides provides how far a datapoint from mean or relationship to the mean of group of values.
z = np.abs(stats.zscore(copy_ds))
copy_ds= copy_ds[(z < 3).all(axis=1)] #along column

#a quantile determines how many values in a distribution are above or below a certain limit.
Q1 = copy_ds.quantile(0.25)
Q2 = copy_ds.quantile(0.75)
IQR = Q2 - Q1
copy_ds = copy_ds[~((copy_ds < (Q1 - 1.5 * IQR)) | (copy_ds > (Q2 + 1.5 * IQR))).any(axis=1)]

from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
X = copy_ds.drop(columns = ['Outcome'], axis=1)
Y = copy_ds['Outcome']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)

clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train, Y_train)

filename='diabetes'
pickle.dump(clf,open(filename,'wb'))