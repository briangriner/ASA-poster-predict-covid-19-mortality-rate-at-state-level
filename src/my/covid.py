
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[35]:


import sklearn


# In[36]:


data = pd.read_csv('a.csv')


# In[37]:


data.head()


# In[38]:


cor = data.corr()
cor = abs(cor['mortality_rate'])
print(cor[cor > 0.3])


# In[39]:


data.drop([33, 47], inplace=True) # Get rid of Guam/Puerto Rico
y = data['mortality_rate'] # Labels
states = data['state'] # If we want to look a state up later
data.drop(columns=['mortality_rate', 'Locationdesc', 'country_region', 'last_update', 'lat', 'long', 'confirmed', 'deaths',
                  'recovered', 'active', 'people_tested', 'people_hospitalized', 'testing_rate', 'incident_rate', 'hospitalization_rate',
                  'state'], inplace=True)
data.fillna(data.mean(), inplace=True)


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaled = StandardScaler().fit_transform(data)
X = pd.DataFrame(scaled, columns=data.columns)


# In[50]:


from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer

def evaluate_model(model, param_dict, passes=10):
    min_test_err = 1e10
    best_hyperparams = {}
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    for i in range(passes):
        print('Pass {}/10 for model {}'.format(i + 1, model))
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
        
        default_model = model()
        model_gs = GridSearchCV(default_model, param_dict, cv=3, n_jobs=16, verbose=1, scoring=scorer)
        model_gs.fit(X_train, y_train)
        optimal_model = model(**model_gs.best_params_)
        optimal_model.fit(X_train, y_train)
        err = mean_squared_error(y_test, optimal_model.predict(X_test))
        #print('MSE for {}: {}'.format(model, err))
        if err < min_test_err:
            min_test_err = err
            best_hyperparams = model_gs.best_params_
    print('Model {} with hyperparams {} yielded error {}'.format(model, best_hyperparams, min_test_err))
        
    
evaluate_model(LassoCV, {'eps': [0.0005, 0.001, 0.002, 0.003, 0.005, 0.006], 
                                'n_alphas':[50, 100, 200, 300, 400, 500],
                                'tol': [0.001, 0.005, 0.01],
                                'max_iter': [4000, 7000, 10000]})

#evaluate_model(Ridge, {'alpha' : [(0.1, 0.3, 0.7, 1.0, 2.0, 5.0)]})

evaluate_model(KNeighborsRegressor, {'n_neighbors' : np.arange(1, 10)})


# In[53]:


evaluate_model(SVR, {'kernel': ['linear', 'poly', 'rbf'], 
                                'degree': [2, 3, 5, 7],
                                'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]})

evaluate_model(GradientBoostingRegressor, {
                                'learning_rate': [0.1, 0.05, 0.02], 
                                'n_estimators': [100, 200, 400, 800],
                                'max_depth': [1, 2, 3, 4, 5],
                                'max_features' : ['auto', 'sqrt', 'log2']})

evaluate_model(DecisionTreeRegressor, {'splitter': ['best', 'random'], 
                                'criterion': ['mse', 'friedman_mse', 'mae'],
                                'max_depth': [None, 2, 3, 4, 5],
                                'max_features' : ['auto', 'sqrt', 'log2']})

evaluate_model(RandomForestRegressor, {'n_estimators': [100, 200, 400, 800], 
                                'max_depth': [None, 2, 3, 4, 5],
                            'min_samples_split': [2, 3, 4],
                                'max_features' : ['auto', 'sqrt', 'log2']})

evaluate_model(MLPRegressor, {'hidden_layer_sizes': [(100,) * 3, (100,) * 10, (100,) * 30, (100,) * 100]})

