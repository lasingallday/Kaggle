#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.preprocessing import LabelEncoder

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

user_data = pd.read_csv('C:/Repos/Kaggle/Dataset/train_modified.csv')
user_data = user_data.dropna()
target = 'Disbursed'
IDcol = 'ID'

#Choose all predictors except target & IDcols
predictors = [x for x in user_data.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)

for column in user_data:
	user_data_unique = user_data[column].unique()
	D = dict([(j,i) for i,j in enumerate(user_data_unique)])
	user_data[column] = user_data[column].map(D)

modelfit(gbm0, user_data, predictors)
param_test1 = {'n_estimators':range(20,81,10)}
if __name__ == '__main__':
	gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
	param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	gsearch1.fit(user_data[predictors],user_data[target])
	gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
