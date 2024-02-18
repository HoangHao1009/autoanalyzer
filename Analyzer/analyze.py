import pandas as pd
from autoanalyzer.Analyzer import Column
import plotly.express as px


from sklearn.cluster import KMeans
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


def _merge(component = list):
    shortest = min(component, key = len)
    result = shortest.data
    for i in [i for i in component if i != shortest]:
        result = pd.merge(result, i.data, how = 'inner', left_index = True, right_index = True)   
    return result

def _to_bool(col):
    return col.apply(lambda x: 1 if x > 0 else 0)

class AllAnalyze:
    def __init__(self, customer = Column.mainColunm, sale = Column.Sale, date = Column.Date):
        self.basicinfo = BasicInfo(customer, sale, date)
        self.growth = Growth(customer, sale, date)
        self.newexisting = NewExisting(customer, sale, date)
        self.retention = Retention(customer, sale, date)
        self.cohort = Cohort(customer, sale, date)
        self.segmentation = RFMSegmentaion(customer, sale, date)
        self.predictor = CustomerLTVPredictor(self.segmentation)

    def get_full_result(self):
        return {
            'Basic Info': self.basicinfo, 
            'Growth': self.growth,
            'New Existing Customer': self.newexisting,
            'Retention': self.retention,
            'Cohort': self.cohort,
            'Segmentation': self.segmentation,
        }

    def get_analyze_data(self):
        return {
            'Basic Info data': self.basicinfo.all_data, 
            'Growth data': self.growth.all_data,
            'New Existing Customer data': self.newexisting.all_data,
            'Retention data': self.retention.all_data,
            'Cohort data': self.cohort.all_data,
            'Segmentation data': self.segmentation.all_data,
        }

    def get_all_chart(self):
        return {
            'Basic Info chart': self.basicinfo.all_px, 
            'Growth chart': self.growth.all_px,
            'New Existing Customer chart': self.newexisting.all_px,
            'Retention chart': self.retention.all_px,
            'Cohort chart': self.cohort.all_px,
            'Segmentation chart': self.segmentation.all_px,
        }

class BasicInfo:
    def __init__(self, customer = Column.mainColunm, sale = Column.Sale, date = Column.Date):
        self.input = _merge([i for i in [customer, sale, date] if i is not None])
        self.get_data()
        self.get_chart()
        self.all_data = {
            'total_revenue': self.total_revenue,
            'unique_customer': self.unique_customer,
            'customer_revenue': self.customer_revenue,
            'avg_customer_revenue': self.avg_customer_revenue,
            'segment_revenue': self.segment_revenue,
            'segment_unique_customer': self.segment_unique_customer
        }
        self.all_px = {
            'customer_revenue_px': self.customer_revenue_px,
            'segment_revenue_px': self.segment_revenue_px,
            'segment_unique_customer_px': self.segment_unique_customer_px
        }

    def get_data(self):
        data = self.input
        self.total_revenue = data['sale'].sum()
        self.unique_customer = data['customer'].nunique()
        self.customer_revenue = data.groupby(['customer'])['sale'].sum().reset_index().sort_values(by = ['sale'], ascending = True)
        self.avg_customer_revenue = self.customer_revenue['sale'].mean()
        self.segment_revenue = data.groupby(['customer_segment'])['sale'].sum().reset_index().sort_values(by = ['sale'])
        self.segment_revenue['percent'] = self.segment_revenue[['sale']].apply(lambda x: x/ x.sum())
        self.segment_unique_customer = data.groupby(['customer_segment'])['customer'].nunique().reset_index().sort_values(by = ['customer'])
        self.segment_unique_customer['percent'] = self.segment_unique_customer[['customer']].apply(lambda x: x/ x.sum())

    def get_chart(self):
        self.customer_revenue_px = px.bar(self.customer_revenue, orientation = 'h',
                                    x = 'sale', y = 'customer', 
                                    title = 'Revenue by Unique Customer')
        self.segment_revenue_px = px.pie(self.segment_revenue, 
                                    names = 'customer_segment', values = 'sale',
                                    title = 'Revenue Percentage by Customer Segment')
        self.segment_unique_customer_px = px.pie(self.segment_unique_customer, 
                                                 names = 'customer_segment', values = 'customer',
                                                 title = 'Number of Unique customer Percentage by Customer Segment')

class Growth:
    def __init__(self, customer = Column.mainColunm, sale = Column.Sale, date = Column.Date):
        self.input = _merge([i for i in [customer, sale, date] if i is not None])
        self.segment_by = customer.chose
        self.get_data()
        self.get_chart()
        self.all_data = {
            'monthly_revenue': self.monthly_revenue,
            'monthly_customer': self.monthly_unique_customer,
            'segment_monthly_revenue': self.segment_monthly_revenue
        }
        self.all_px = {
            'monthly_revenue_px': self.monthly_revenue_px,
            'monthly_unique_customer_px': self.monthly_unique_customer_px,
            'segment_monthly_revenue_px': self.segment_monthly_revenue_px,
            'segment_monthly_customer_px': self.segment_monthly_customer_px
        }

    def get_data(self):
        data = self.input

        self.monthly_revenue = data.groupby(['y_m'])['sale'].sum().reset_index().sort_values(by = ['y_m'])
        self.monthly_unique_customer = data.groupby(['y_m'])['customer'].nunique().reset_index().sort_values(by = ['y_m'])

        self.segment_monthly_revenue = data.groupby(['y_m', 'customer_segment'])['sale'].sum().reset_index()
        self.segment_monthly_customer = data.groupby(['y_m', 'customer_segment'])['customer'].nunique().reset_index()

    def get_chart(self):
        self.monthly_revenue_px = px.line(self.monthly_revenue, 
                                          x = 'y_m', y = 'sale',
                                          title = 'Revenue by Month')
        self.monthly_unique_customer_px = px.line(self.monthly_unique_customer, 
                                                  x = 'y_m', y = 'customer',
                                                  title = 'Number of Unique customer by Month')
        self.segment_monthly_revenue_px = px.line(self.segment_monthly_revenue, 
                                                  x = 'y_m', y = 'sale', color = 'customer_segment',
                                                  title = 'Monthly Revenue by Customer Segment')
        self.segment_monthly_customer_px = px.line(self.segment_monthly_customer, 
                                                   x = 'y_m', y = 'customer', color = 'customer_segment',
                                                   title = 'Monthly Number of Unique customer by Customer Segment')
        
class NewExisting:
    def __init__(self, customer = Column.mainColunm, sale = Column.mainColunm, date = Column.Date):
        input = _merge([customer, date, sale])
        input['FirstPurchase'] = input.groupby(['customer'])['date'].transform('min')
        input['FirstPurchase'] = input['FirstPurchase'].dt.strftime('%y_%m')
        input['type'] = 'New'
        input.loc[input['FirstPurchase'] != input['y_m'], 'type'] = 'Existing'
        self.input = input.drop(['FirstPurchase'], axis = 1)
   
        self.get_data()
        self.get_chart()

        self.all_data = {
            'count': self.count,
            'month_new_percent': self.monthly_new_percent,
            'type_percent': self.type_percent,
            'segment_count': self.segment_count,
            'segment_monthly_new_percent': self.segment_monthly_new_percent
        }
        self.all_px = {
            'count_px': self.count_px,
            'month_new_percent_px': self.monthly_new_percent_px,
            'type_percent_px': self.type_percent_px,
            'segment_count_px': self.segment_count_px,
            'segment_monthly_new_percent_px': self.segment_monthly_new_percent_px
        }

    def get_data(self):
        def take_new_pct(data):
            data['New Percent'] = data['New']/ data['Existing']
            data = data.fillna(0)
            return data
        self.count = self.input.groupby(['y_m', 'type'])['customer'].nunique().reset_index()
        self.count.columns = ['y_m', 'type', 'count type']

        self.monthly_new_percent = self.input.groupby(['y_m', 'type'])['customer'].nunique().unstack().reset_index()
        self.monthly_new_percent = take_new_pct(self.monthly_new_percent)
        self.type_percent = self.input.groupby(['type'])['customer'].nunique().reset_index()
        self.type_percent[['customer']] = self.type_percent[['customer']].apply(lambda x: x/ x.sum(), axis = 0)
        self.type_percent = self.type_percent.fillna(value = 0)
        self.type_percent.columns = ['type', 'type_percent']

        self.segment_count = self.input.groupby(['y_m', 'type', 'customer_segment'])['customer'].nunique().reset_index()
        self.segment_monthly_new_percent = self.segment_count.pivot_table(values = 'customer', index = ['y_m', 'customer_segment'], columns = 'type').reset_index()
        self.segment_monthly_new_percent = take_new_pct(self.segment_monthly_new_percent)

    
    def get_chart(self):
        self.count_px = px.line(self.count, x = 'y_m', y = 'count type', color = 'type',
                                category_orders = {'y_m': self.segment_count['y_m'].sort_values()},
                                title = 'Number of New/ Existing Customer by Month')
        self.monthly_new_percent_px = px.line(self.monthly_new_percent, x = 'y_m', y = 'New Percent',
                                              title = 'New Customer Percent by Month')
        self.type_percent_px = px.pie(self.type_percent, values = 'type_percent', names = 'type',
                                      title = 'Number of New/ Existing Customer by Order')
        self.segment_count_px = px.line(self.segment_count.sort_values(by = ['y_m']), x = 'y_m', y = 'customer', color = 'customer_segment', line_dash = 'type',
                                        category_orders = {'y_m': self.segment_count['y_m'].sort_values()},
                                        title = 'Number of New/ Existing Customer by Segment/Month'
                                        )
        self.segment_monthly_new_percent_px = px.line(self.segment_monthly_new_percent, x = 'y_m', y = 'New Percent', color = 'customer_segment',
                                                      title = 'New Customer Percent by Segment/ Month')

class Retention:
    def __init__(self, customer = Column.mainColunm, sale = Column.Sale, date = Column.Date):
        input = _merge([customer, date, sale])
        input = input.groupby(['customer', 'y_m'])['sale'].sum().unstack()
        self.input = input.apply(_to_bool, axis = 1)
        self.get_data()
        self.get_chart()
        self.all_data = {
            'by_customer': self.by_customer,
            'by_month': self.by_month
        }
        self.all_px = {
            'retention_px': self.retention_px,
            'retention_pct_px': self.retention_pct_px
        }

    def get_data(self):
        data = self.input
        self.by_customer = self.input
        di = {'y_m': [], 'Total unique customer': [], 'Retained customer': []}
        for i in range(len(data.columns)):
            current_month = data.columns[i]
            if i == 0:
                prev_month = None
            else:
                prev_month = data.columns[i - 1]
            di['y_m'].append(current_month)
            di['Total unique customer'].append(data[current_month].sum())

            retain = 0
            try:
                for c, p in zip(data[current_month], data[prev_month]):
                    if c == p and p == 1:
                        retain += 1
            except:
                retain = 0
            di['Retained customer'].append(retain)
        
        data = pd.DataFrame(di)
        data['Retention rate'] = data['Retained customer'] / data['Total unique customer']
        self.by_month = data
        
    def get_chart(self):
        self.retention_px = px.bar(
            self.by_month.drop(['Retention rate'], axis = 1).melt(id_vars='y_m', 
                                                                  var_name='type', 
                                                                  value_name='count'),
            x='y_m', y='count', color='type', barmode='group',
            title = 'Count of Retained Customer / Total Customer by Month'
        )

        self.retention_pct_px = px.line(self.by_month, x='y_m', y='Retention rate',
                                        title = 'Retention Rate by Month')

class Cohort:
    def __init__(self, customer = Column.mainColunm, sale = Column.mainColunm, date = Column.Date):
        if not customer.type == 'customer':
            raise TypeError('customer requires customer columns')
        if not sale.type == 'sale':
            raise TypeError('sale requires sales columns')
        self.input = _merge([customer, date, sale])
        self.get_data()
        self.get_chart()
        self.all_data = {
            'by_revenue': self.by_revenue,
            'by_revenue_pct': self.by_revenue_pct,
            'by_retention': self.by_retention,
            'by_retention_pct': self.by_retention_pct
        }
        self.all_px = {
            'by_revenue_px': self.by_revenue_px,
            'by_revenue_pct_px': self.by_revenue_pct_px,
            'by_retention_px': self.by_retention_px,
            'by_retention_pct_px': self.by_retention_pct_px
        }

    def get_data(self):
        def take_cohort(data, type, percentage = False):
            data['Cohort Month'] = data.groupby(['customer'])['month'].transform('min')
            data['Cohort Index'] = data['month'] - data['Cohort Month'] + 1
            if type == 'revenue':
                data = data.groupby(['Cohort Month', 'Cohort Index'])['sale'].sum()
            elif type == 'retention':
                data = data.groupby(['Cohort Month', 'Cohort Index'])['customer'].nunique()
            cohort_table = data.unstack()
            if percentage:
                cohort_table = cohort_table.apply(lambda x: x / cohort_table.iloc[:, 0], axis = 0)
            return cohort_table
        self.by_revenue = take_cohort(self.input, type = 'revenue')
        self.by_revenue_pct = take_cohort(self.input, type = 'revenue', percentage = True)
        self.by_retention = take_cohort(self.input, type = 'retention')
        self.by_retention_pct = take_cohort(self.input, type = 'retention', percentage = True)

    def get_chart(self):
        self.by_revenue_px = px.imshow(self.by_revenue,
                                       title = 'Cohort Analysis by Revenue')
        self.by_revenue_pct_px = px.imshow(self.by_revenue_pct,
                                           title = 'Cohort Analysis by Revenue Percent')
        self.by_retention_px = px.imshow(self.by_retention,
                                         title = 'Cohort Analysis by Retention')
        self.by_retention_pct_px = px.imshow(self.by_retention_pct,
                                             title = 'Cohort Analysis by Retention Percent')
        for i in [self.by_revenue_px, self.by_revenue_pct_px, self.by_retention_px, self.by_retention_pct_px]:
            i.update_layout(xaxis_side='top')

class RFMSegmentaion:
    def __init__(self, customer = Column.mainColunm, sale = Column.mainColunm, date = Column.Date,
                 r_clusters = 4, f_clusters = 4, m_clusters = 4):
        self.input = _merge([customer, date, sale])
        self.get_data(r_clusters, f_clusters, m_clusters)
        self.get_chart()
        self.all_data = {
            'rfm': self.rfm,
            'segment': self.segment
        }
        self.all_px = {
            'recency_px': self.recency_px,
            'frequency_px': self.frequency_px,
            'monetary_px': self.monetary_px,
            'recency_frequency_px': self.recency_frequency_px,
            'recency_monetary_px': self.recency_monetary_px,
            'frequency_monetary_px': self.frequency_monetary_px
        }

    def get_data(self, r_clusters, f_clusters, m_clusters):
        def get_rfm(data):
            n_ym = data['y_m'].nunique()
            recency = data.groupby(['customer'])['date'].max().reset_index()
            recency.columns = ['customer', 'MaxPurchaseDate']
            recency['Recency'] = (recency['MaxPurchaseDate'].max() - recency['MaxPurchaseDate']).dt.days

            frequency = data.groupby(['customer'])['date'].count().reset_index()
            frequency.columns = ['customer', 'Frequency']

            monetary = data.groupby(['customer'])['sale'].sum().reset_index()
            monetary.columns = ['customer', 'Monetary']
            data = recency.merge(frequency, how = 'left', on = 'customer')\
                        .merge(monetary, how = 'left', on = 'customer')
            data[['Recency', 'Frequency', 'Monetary']] = data[['Recency', 'Frequency', 'Monetary']].map(lambda x: x/ n_ym)
            return data
        
        def get_segment(data, r_clusters, f_clusters, m_clusters):
            cluster_cols = ['Recency cluster', 'Frequency cluster', 'Monetary cluster']
            rfm_col = ['Recency', 'Frequency', 'Monetary']
            rfm = [data[[i]] for i in rfm_col]
            kmeans_list = []
            for i in [r_clusters, f_clusters, m_clusters]:
                kmeans = KMeans(n_init = 10, n_clusters = i)
                kmeans_list.append(kmeans)
            for col, cluster_col, kmeans in zip(rfm, cluster_cols, kmeans_list):
                data[cluster_col] = kmeans.fit_predict(col)
            
            for rc, cc in zip(rfm_col, cluster_cols):
                temp = data.groupby([cc])[rc].mean().reset_index()
                if rc == 'Recency':
                    temp = temp.sort_values(by = rc, ascending = False).reset_index(drop = True)
                else:
                    temp = temp.sort_values(by = rc, ascending = True).reset_index(drop = True)
                temp['index'] = temp.index
                data = data.merge(temp[['index', cc]], how = 'left', on = cc)
                data = data.drop(cc, axis = 1)
                data = data.rename(columns = {'index': cc})

            data['RFM Overall Score'] = data['Recency cluster'] + data['Frequency cluster'] + data['Monetary cluster']
            segment_score = max(data['RFM Overall Score'])
            data['RFM Segment'] = 'Low Value'
            data.loc[data['RFM Overall Score'] > int(segment_score / 3), 'RFM Segment'] = 'Mid Value'
            data.loc[data['RFM Overall Score'] > int(segment_score / 3) * 2, 'RFM Segment'] = 'High Value'
            return data
        self.rfm = get_rfm(self.input)
        self.segment = get_segment(self.rfm, r_clusters, f_clusters, m_clusters)


    def get_chart(self):
        self.recency_px = px.histogram(self.rfm['Recency'])
        self.frequency_px = px.histogram(self.rfm['Frequency'])
        self.monetary_px = px.histogram(self.rfm['Monetary'])
        self.recency_frequency_px = px.scatter(
            self.segment,
            x = 'Recency', y = 'Frequency',
            color = 'RFM Segment',
            trendline = 'ols',
            title = 'Recency - Frequency Relationship'
        )
        self.recency_monetary_px = px.scatter(
            self.segment,
            x = 'Recency', y = 'Monetary',
            color = 'RFM Segment',
            trendline = 'ols',
            title = 'Recency - Monetary Relationship'
        )
        self.frequency_monetary_px = px.scatter(
            self.segment,
            x = 'Frequency', y = 'Monetary',
            color = 'RFM Segment',
            trendline = 'ols',
            title = 'Frequency - Monetary Relationship'
        )

class CustomerLTVPredictor:
    def __init__(self, analyzer = RFMSegmentaion):
        rfm = analyzer.segment.copy()
        customer_segment = analyzer.input[['customer', 'customer_segment']].drop_duplicates()
        basic_df = analyzer.input.groupby('customer')['sale'].sum().reset_index()
        basic_df.columns = ['customer', 'Total Revenue']
        self.df = basic_df.merge(rfm, how = 'left', on = 'customer')\
                        .merge(customer_segment, how = 'left', on = 'customer')
        self._model_params = {
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
    def cluster_hint(self):
        sse = {}
        for k in range(1, 10):
            kmeans = KMeans(n_init = 10, n_clusters = k, max_iter = 1000)
            kmeans.fit(self.df[['Total Revenue']])
            sse[k] = kmeans.inertia_
        return px.line(x = sse.keys(), y = sse.values())

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
        model_params = self._model_params

        if only_modern_model == True:
            modern_model = ['random_forest', 'xgb', 'mlp']
            model_params = {key: self._model_params[key] for key in modern_model}
        
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

        self.predictor_scores = pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params', 'best_estimator'])
        self.best_estimator = self.predictor_scores[self.predictor_scores['best_score'] == self.predictor_scores['best_score'].max()]['best_estimator'].values[0]
        self.pred_pipeline = Pipeline(steps = [
            ('preprocess', self.preprocessor),
            ('model', self.best_estimator)
        ])
        
    def run_best_predictor(self, analyzer = RFMSegmentaion):
        rfm = analyzer.segment
        customer_segment = analyzer.input[['customer', 'customer_segment']].drop_duplicates()
        df = pd.merge(rfm, customer_segment, how = 'left', on = 'customer')

        pred = self.pred_pipeline.predict(df)
        return pd.concat([df['customer'], pd.Series(pred, name = 'Life Time Value Predicted')], axis = 1)
