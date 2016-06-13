import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import float32
#from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("/home/jeff/Documents/workspace/Prudential_LI/train.csv")
clf = DecisionTreeClassifier()	

#mk_cols = train.loc[:, ["Medical_Keyword_"+str(i) for i in range(1,49)]]
#mk_list = list(mk_cols)
mk_cols = train.iloc[:,79:127]
mk_sum = mk_cols.sum(axis=0)
mk_cols = mk_cols.append(mk_cols, ignore_index=True)

category_dict_PI2 = {}
PI2_unique = train["Product_Info_2"].unique()
category_dict_PI2 = dict([(j,i) for i,j in enumerate(PI2_unique)])
train["Product_Info_2"].replace(category_dict_PI2, inplace=True)
print(train)

X = train.drop(['Response'], axis=1)
y = train['Response']
print(X.shape)
print(y.shape)
#Indicate if NaN's part of dataframe
#print(np.any(np.isnan(X)))
X = X.fillna(1e6).astype(np.float32)
y = y.fillna(1e6).astype(np.float32)
#Replace only columns with NaN's
#bad_flats = X.select_dtypes(include=['float64'])
#X = X.replace(bad_flats.astype('float32'))
print(X.dtypes)
clf.fit(X,y)
#clf.predict_proba(train[1,:])
