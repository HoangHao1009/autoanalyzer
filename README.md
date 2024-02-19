# autoanalyzer
Auto analyze - visualize marketing data

You can use it by python or use web interface: https://mktanalyze.streamlit.app/

# Table of contents
- Library installation
- Usage
  - Set up columns
- Analyze
  - Analyze all for one
  - Or analyze what you need
- Predict Customer Life Time Value
  - Set up predictor
  - Using predictor
  - Take your result
  - Using best predictor you have chosen
  

## I. Library installation
To use marketing analyzer, run this code:
```python
git clone https://github.com/HoangHao1009/autoanalyzer
cd autoanalyzer
pip install -e .
```
## II. Usage
### 1. Set up columns
```python
from Analyzer import Column, analyze
#For instance: you have a df with columns: 'Customer ID', 'Segment', 'Sales', 'Order Date'

customer = Column.mainColunm(
    df['Customer ID'], df['Segment'], ['Consumer', 'Corporate'], type = 'customer'
)
sale = Column.Sale(df['Sales'])
date = Column.Date(df['Order Date'], '%d/%m/%Y')
```
### 2. Analyze
#### 2.1. Analyze all for one
```python
analyze = analyze.AllAnalyze(customer, sale, date)
```
#### 2.2. Or analyze what you need
```python
#This make for you to custom your data and analyze flexibly
#You can change your customer columns to another segment like:
#customer = Column.mainColunm(df['Customer ID'], df['Segment'], ['Consumer', 'Home-Office'], type = 'customer')
#and see what different by using detailed analysis
basicinfo = analyze.BasicInfo(customer, sale, date)
growth = analyze.Growth(customer, sale, date)
newexisting = analyze.NewExisting(customer, sale, date)
retention = analyze.Retention(customer, sale, date)
cohort = analyze.Cohort(customer, sale, date)
segmentation = analyze.RFMSegmentaion(customer, sale, date)
```
#### 2.3. Take your results
```python
#all result
analyze.get_full_result()
#all analyze data
analyze.get_analyze_data()
#all visualize chart
analyze.get_all_chart()
#See results by detail analysis
basicinfo.all_data
basicinfo.all_px
#or growth.all_data, newexisting.all_px, ...

```
### 3. Predict Customer Life Time Value
#### 3.1. Set up predictor
```python
#take predictor by using allanalyze:
analyze = analyze.AllAnalyze(customer, sale, date)
predictor = analyze.predictor
#Or set up it by using RFMSegmentaion
predictor = analyze.CustomerLTVPredictor(segmentation)
```
#### 3.2. Using predictor
```python
#Get a hint of how many segment of LTV you would like to set up
predictor.cluster_hint()
#Chose best predictor (machine learning algorithm) of predicting your data
predictor.chose_best_predictor()
#-> params of this:
#remove_outlier_quantile: how many percent of outliner cross to remove it
#cv: the chosing algorithm using Grid/Random Search CV, so you can decide how many cv
#use_randomsearch: use RandomSearchCV or GridSearchCV
#only_modern_model: Use modern models or or even the old models
```
#### 3.3. Take your result
```python
#info of LTV cluster you've decide
predictor.ltv_cluster_info
#the performance of models you use
predictor.predictor_scores
#best predict
best_estimator
```
#### 3.4. Using best predictor you have chosen
```python
predictor.run_best_predictor(#put another RFMSegmentaion here to predict)
```


