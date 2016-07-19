import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn import preprocessing, cross_validation, neighbors, datasets

user_data = pd.read_csv('/home/jeff/Documents/workspace/Airbnb/train_users_2.csv')
h = .1 # step size in the mesh
n = 5

# Try different C (greater to change amount of error allowed) and gamma values.
##lin_clf = LinearSVC()
clf = neighbors.KNeighborsClassifier(n_neighbors=n)

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
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf.fit(X_train,y_train)

#print(clf.predict(X.iloc[[1]]))
#print(X[['affiliate_provider']].min(), X[['affiliate_provider']].min())
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
#x_min, x_max = train[['affiliate_provider']].min() - 1, train[['affiliate_provider']].max() + 1
#y_min, y_max = y_train.min() - 1, y_train.max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                     np.arange(y_min, y_max, h))
#print(np.c_[xx.ravel(),yy.ravel()].shape)
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#preds = clf.predict(test)
##accuracy_1 = np.where(preds==y_test, 1, 0).sum() / float(len(test))
accuracy_2 = clf.score(X_test, y_test)
print("Neighbors: %d, Accuracy: %3f" % (n, accuracy_2))
#n=5:acy_2=.578; n=10:acy_1=.581:acy_2=.583; n=15:acy_1=.587; n=20:acy_1=.577, .591, .598:acy_2=.588
#print(Z)

#Z = Z.reshape(xx.shape)
#plt.figure()
#plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the training points
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())

##lin_clf.fit(X,y)
##print(lin_clf.coef_)

#plt.figure(figsize=(6,5))
#plt.title("Weights of the model")
#plt.plot(lin_clf.coef_, 'b-', label="LinearSVC model")
#plt.xlabel("Features")
#plt.ylabel("Values of the weights")
#plt.legend(loc="best", prop=dict(size=12))
##plt.show()

