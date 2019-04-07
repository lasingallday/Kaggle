import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

user_data = pd.read_csv('/home/jeff/Documents/workspace/Airbnb/train_users_2.csv')
#Try different C (greater to change amount of error allowed) and gamma values.
lin_clf = LinearSVC()

user_data = user_data.drop(['id','date_account_created','timestamp_first_active','date_first_booking','age','first_affiliate_tracked'], axis=1)

for column in user_data:
	stringy_columns = ['gender','signup_method','language','affiliate_channel','affiliate_provider','signup_app','first_device_type','first_browser','country_destination']
	if (column in stringy_columns):
		category_dict = {}
		user_data_unique = user_data[column].unique()
		category_dict = dict([(j,i) for i,j in enumerate(user_data_unique)])
		print(category_dict)
		user_data[column] = user_data[column].map(category_dict)

print(user_data.head())			
X = user_data.drop(['country_destination'], axis=1)
y = user_data['country_destination']

lin_clf.fit(X,y)
print(lin_clf.coef_)

#plt.figure(figsize=(6,5))
#plt.title("Weights of the model")
#plt.plot(lin_clf.coef_, 'b-', label="LinearSVC model")
#plt.xlabel("Features")
#plt.ylabel("Values of the weights")
#plt.legend(loc="best", prop=dict(size=12))
#plt.show()

