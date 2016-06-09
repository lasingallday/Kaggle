import pandas as pd
import numpy as np
import seaborn as sns
#from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("C:/Users/jeffrey.thomas/workspace/Prudential_LI/train.csv")
#print(train.head(15))

#mk_cols = train.loc[:, ["Medical_Keyword_"+str(i) for i in range(1,49)]]
#mk_list = list(mk_cols)
mk_cols = train.iloc[:,79:127]
mk_sum = mk_cols.sum(axis=0)
mk_cols = mk_cols.append(mk_result, ignore_index=True)

for column in train:
	if (column == "Product_Info_2"):

