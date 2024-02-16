from Analyzer.analyze import RFMSegmentaion

import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

_model_params = {
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [1, 5, 10]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [5, 10, 15],
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {
        }
    },
    'svc': {
        'model': SVC(),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['linear', 'poly', 'rbf']
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, 15]
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 200, 500],
            'criterion': ['gini', 'entropy']
        }
    },
    'xgb': {
        'model': XGBClassifier(),
        'params': {
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 500],
            'max_depth': [5, 10]
        }
    },
    'mlp': {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes': [(32,128),(64, 128)]
        }
    }
}

class CustomerLTVPredictor:
    def __init__(self, analyzer = RFMSegmentaion):
        rfm = analyzer.rfm.copy()
        basic_df = analyzer.input.groupby('customer')['sale'].sum().reset_index()
        basic_df.columns = ['customer', 'Total Revenue']
        self.df = rfm.merge(basic_df, how = 'left', on = 'customer')

    def cluster_hint(self):
        sse = {}
        for k in range(1, 10):
            kmeans = KMeans(n_init = 10, n_clusters = k, max_iter = 1000)
            kmeans.fit(self.df[['Total Revenue']])
            sse[k] = kmeans.inertia_
        px.line(x = sse.keys(), y = sse.values())

    def chose_best_predictor(self, revenue_clusters = 3, 
                             remove_outlier_quantile = 1,
                             cv = 5,
                             use_randomsearch = True,
                             only_modern_model = True):
        self.df = self.df[self.df['Total Revenue'] < self.df['Total Revenue'].quantile(remove_outlier_quantile)]
        kmeans = KMeans(n_init = 10, n_clusters = revenue_clusters, max_iter = 1000)
        self.df['LTV Cluster'] = kmeans.fit_predict(self.df[['Total Revenue']])

        mean_revenue_ltv = self.df.groupby(['LTV Cluster'])['Total Revenue'].mean().reset_index()
        mean_revenue_ltv.rename(columns = {'Total Revenue': 'Life Time Value (Revenue)'})
        count_ltv_cluster = self.df['LTV Cluster'].value_counts().reset_index()
        self.ltv_cluster_info = mean_revenue_ltv.merge(count_ltv_cluster, how = 'left', on = 'LTV Cluster')
        self.ltv_cluster_info = self.ltv_cluster_info.rename(columns = {'Total Revenue': 'Life Time Value (Revenue)',
                                                                        'count': 'Customer Count'})
        model_params = _model_params

        if only_modern_model == True:
            modern_model = ['random_forest', 'xgb', 'mlp']
            model_params = {key: _model_params[key] for key in modern_model}
        
        X = self.df.drop(['LTV Cluster', 'Total Revenue'], axis = 1)
        y = self.df['LTV Cluster']

        cat_col = [col for col in X.columns if X[col].dtype == 'object']
        num_col = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

        if X.isna().sum().sum() > 0:
            print('Warning: Your data have NA values. Predictor will use "mean" for numerical values\
                  and "most_frequent" for categorical values')
        
        num_transformer = Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'mean')),
            ('scale', MinMaxScaler())
        ])

        cat_transformer = Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'most_frequent')),
            ('encode', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
        ])

        preprocessor = ColumnTransformer(
            transformers = [
                ('num', num_transformer, num_col),
                ('cat', cat_transformer, cat_col)
            ],
            remainder = 'drop'
        )

        self.preprocessor = preprocessor

        X = preprocessor.fit_transform(X)


        #SearchCV
        
        scores = []
        for model_name, mp in model_params.items():
            if use_randomsearch == False:
                clf = GridSearchCV(mp['model'], mp['params'], cv = cv, return_train_score = False)
                clf.fit(X, y)
                s = {
                'model': model_name,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_,
                'best_estimator': clf.best_estimator_
                }
            else:
                clf = RandomizedSearchCV(mp['model'], mp['params'], cv = cv, return_train_score = False)
                clf.fit(X, y)
                s = {
                'model': model_name,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_,
                'best_estimator': clf.best_estimator_
                }
            scores.append(s)

        self.model_params = pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params', 'best_estimator'])
        self.best_estimator = self.model_params[self.model_params['best_score'] == self.model_params['best_score'].max()]['best_estimator'].values[0]
        self.pred_pipeline = Pipeline(steps = [
            ('preprocess', self.preprocessor),
            ('model', self.best_estimator)
        ])
        
    def run_best_predictor(self, analyzer = RFMSegmentaion):
        if analyzer.mode != 'rfm_cluster':
            print('You must run rfm cluster before using CustomerLTVPredictor')
        pred =  self.pred_pipeline.predict(analyzer.df)
        return pred
