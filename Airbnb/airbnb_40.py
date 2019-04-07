import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn import preprocessing, cross_validation, neighbors, datasets, metrics
from sklearn.multiclass import OneVsRestClassifier   #OneVsRest algorithm
from sklearn.grid_search import GridSearchCV   #Performing grid search
from sklearn.preprocessing import LabelEncoder
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['country_destination'])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['country_destination'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['country_destination'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['country_destination'], dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
	plt.show()


user_data = pd.read_csv('/home/jeff/Documents/workspace/Airbnb/train_users_2.csv')
n = 5

# Try different C (greater to change amount of error allowed) and gamma values.
lin_clf = SVC(C=1.0,kernel='linear',probability=True)
#clf = neighbors.KNeighborsClassifier(n_neighbors=n)

user_data = user_data.drop(['id','date_account_created','timestamp_first_active','date_first_booking','age','first_affiliate_tracked'], axis=1)

for column in user_data:
	stringy_columns = ['gender','signup_method','language','affiliate_channel','affiliate_provider','signup_app','first_device_type','first_browser','country_destination']
	if (column in stringy_columns):
		category_dict = {}
		user_data_unique = user_data[column].unique()
		category_dict = dict([(j,i) for i,j in enumerate(user_data_unique)])
		print(category_dict)
		user_data[column] = user_data[column].map(category_dict)

predictors = [x for x in user_data.columns if x not in ['country_destination','id']]
modelfit(lin_clf, user_data, predictors)

print(user_data.head())
X = user_data.drop(['country_destination'], axis=1)
y = user_data['country_destination']
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#clf.fit(X_train,y_train)
#preds = clf.predict(test)
#accuracy_1 = np.where(preds==y_test, 1, 0).sum() / float(len(test))
#accuracy_2 = clf.score(X_test, y_test)
#print("Neighbors: %d, Accuracy: %3f" % (n, accuracy_2))
#n=5:acy_2=.578; n=10:acy_1=.581:acy_2=.583; n=15:acy_1=.587; n=20:acy_1=.577, .591, .598:acy_2=.588


param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
if __name__ == '__main__':
	
	gsearch3 = GridSearchCV(estimator = OneVsRestClassifier(lin_clf), 
	param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	gsearch3.fit(user_data[predictors],user_data[target])
	print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
	modelfit(gsearch3.best_estimator_, user_data, predictors)

