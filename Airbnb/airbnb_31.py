import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn import preprocessing, cross_validation, neighbors, datasets, decomposition


user_data = pd.read_csv('/home/jeff/Documents/workspace/Airbnb/train_users_2.csv')
h = .02 # step size in the mesh
n = 5
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Try different C (greater to change amount of error allowed) and gamma values.
##lin_clf = LinearSVC()
clf = neighbors.KNeighborsClassifier(n_neighbors=n)
# Use a kernel I have created myself
#clf = svm.SVC(kernel=my_kernel)

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
X_norm = np.mean(X_train, axis=0)
X_train_centered = X_train - X_norm
svd = decomposition.TruncatedSVD(n_components=2)
X_2d = svd.fit_transform(X_train_centered)
clf.fit(X_2d, y_train)

#plt.scatter(X_2d[:,0], X_2d[:,1], c=y_train, s=50, cmap=plt.cm.Paired)
#plt.colorbar()
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#plt.title('First two PC\'s using digits data')
#plt.show()


#print(X[['affiliate_provider']].min(), X[['affiliate_provider']].min())
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_2d[:,0].min() - 1, X_2d[:,1].max() + 1
y_min, y_max = y_train.min() - 1, y_train.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
#print(np.c_[xx.ravel(),yy.ravel()].shape)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
#plt.figure(figsize=(6,5))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the training points
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

