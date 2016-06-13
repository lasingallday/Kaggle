import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from numpy import float32
#from sklearn.feature_extraction.text import CountVectorizer

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tsvg", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

train = pd.read_csv("C:/Users/jeffrey.thomas/workspace/Prudential_LI/train.csv")
dt = DecisionTreeClassifier()

#mk_cols = train.loc[:, ["Medical_Keyword_"+str(i) for i in range(1,49)]]
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
X_list = list(X.columns[:])
#print(y)
#Indicate if NaN's part of features dataframe
#print(np.any(np.isnan(X)))
X = X.fillna(1e6).astype(np.float32)
y = y.fillna(1e6).astype(np.float32)
#Replace only columns with NaN's
#bad_flats = X.select_dtypes(include=['float64'])
#X = X.replace(bad_flats.astype('float32'))
#print(X.dtypes)
dt.fit(X,y)
visualize_tree(dt, X_list)
