
# coding: utf-8

# ### Loading the libraries

# In[484]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[485]:


data=pd.read_csv("TrainingData - Training.csv")


# In[486]:


data.dtypes


# In[487]:


data.describe(include="all").T


# In[488]:


#data.drop(['Period'],axis=1,inplace=True)


# In[489]:


data.shape


# In[490]:


na_columns=[col for col in data.columns if data[col].isnull().any()]
na_columns


# In[491]:


data.isna().sum()


# ### Replace missing values by mean and drop columns where nearly or more than 30 percent values are missing

# In[492]:


mean_PIA_40=data['Print_Impressions.Ads40'].mean()
mean_PWCA_50=data['Print_Working_Cost.Ads50'].mean()


# In[493]:


data['Print_Impressions.Ads40']=data['Print_Impressions.Ads40'].replace(np.nan,mean_PIA_40)
data['Print_Working_Cost.Ads50']=data['Print_Working_Cost.Ads50'].replace(np.nan,mean_PWCA_50)


# In[494]:


[col for col in data.columns if data[col].isnull().any()]


# In[495]:


data_new=data.dropna(axis='columns')


# In[496]:


data_new


# In[497]:


data_new.drop(['Period'],axis=1,inplace=True)


# In[498]:


data_new.columns


# In[499]:


sns.pairplot(data_new)


# In[500]:


corr = data_new.corr()
ax = sns.heatmap(corr,vmin=-1, vmax=1,center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')


# In[501]:


x_data=data_new[['Print_Impressions.Ads40', 'Print_Working_Cost.Ads50', 'SOS_pct',
       'CCFOT', 'Median_Temp', 'Median_Rainfall', 'Fuel_Price', 'Inflation',
       'Trade_Invest', 'Brand_Equity', 'Avg_EQ_Price', 'Any_Promo_pct_ACV',
       'EQ_Base_Price', 'Est_ACV_Selling', 'pct_ACV', 'Avg_no_of_Items',
       'pct_PromoMarketDollars_Category', 'RPI_Category', 'Competitor1_RPI',
       'Competitor2_RPI', 'Competitor3_RPI', 'Competitor4_RPI', 'EQ_Category',
       'EQ_Subcategory', 'pct_PromoMarketDollars_Subcategory',
       'RPI_Subcategory']]
y_data=data_new['EQ']


# In[502]:


data_new.describe(percentiles = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]).T


# ### Outlier Treatment

# In[503]:


x_data = x_data.div(x_data.quantile(.99)).clip_upper(1)
print (x_data)


# ### Train Test Split & Standizing x variable/regressor variable

# In[504]:


x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.20,random_state=0)


# In[505]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ### Fitting Random Forest Regressor Model

# In[506]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[507]:


m=[]
rmse_test=[]
n=range(4,500)
for n in n:
    regressor=RandomForestRegressor(n_estimators=n,min_samples_leaf=1,n_jobs=-1,oob_score=True,random_state=0)
    regressor.fit(x_train,y_train)
    y_pred_test=regressor.predict(x_test)
    m.append(n)
    rmse_test.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred_test)))


# In[508]:


m=pd.DataFrame(m)
rmse_test=pd.DataFrame(rmse_test)
f_results=pd.concat([m,rmse_test], axis=1, ignore_index=True)
f_results.columns=['Number of Estimators','Root Mean Square Error Testing']


# In[509]:


f_results.sort_values(by='Root Mean Square Error Testing', ascending=True).head()


# ### Feature Importance of the variables

# In[510]:


feat_importances = pd.Series(regressor.feature_importances_, index=x_data.columns)
feat_importances.nlargest(15).plot(kind='barh')


# In[511]:


feat_importances.nlargest(15)


# ### Correlation Matrix for x variables / regressor variables

# In[512]:


x_data.corr()[1:13].T


# In[513]:


x_data.corr()[14:26].T


# ### After Checking Variable Importance and Correlation Matrix we select the following list of regressor variables

# In[514]:


x_data_new=x_data[['Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price']]


# In[515]:


x_train_new,x_test_new,y_train_new,y_test_new=train_test_split(x_data_new,y_data,test_size=0.20,random_state=0)


# In[516]:


x_train_new = sc.fit_transform(x_train_new)
x_test_new = sc.transform(x_test_new)


# In[517]:


m_new=[]
rmse_test_new=[]
n_new=range(4,500)
for n in n_new:
    regressor=RandomForestRegressor(n_estimators=n,min_samples_leaf=1,n_jobs=-1,oob_score=True,random_state=0)
    regressor.fit(x_train_new,y_train_new)
    y_pred_test_new=regressor.predict(x_test_new)
    m_new.append(n)
    rmse_test_new.append(np.sqrt(metrics.mean_squared_error(y_test_new,y_pred_test_new)))


# In[518]:


m_new=pd.DataFrame(m_new)
rmse_test_new=pd.DataFrame(rmse_test_new)
f_results_new=pd.concat([m_new,rmse_test_new], axis=1, ignore_index=True)
f_results_new.columns=['Number of Trees','Root Mean Square Error Testing']


# In[519]:


f_results_new.sort_values(by='Root Mean Square Error Testing', ascending=True).head()


# ## Testing Data Set

# In[520]:


test_data=pd.read_csv('TestData - Test.csv')


# In[521]:


[col for col in test_data.columns if test_data[col].isnull().any()]


# In[522]:


test_data=test_data.drop(['Period','Social_Search_Impressions',
 'Social_Search_Working_cost',
 'Digital_Impressions',
 'Digital_Working_cost',
 'Print_Impressions.Ads40',
 'Print_Working_Cost.Ads50',
 'OOH_Impressions',
 'OOH_Working_Cost',
 'Digital_Impressions_pct',
 'Any_Feat_pct_ACV',
 'Any_Disp_pct_ACV',
 'Magazine_Impressions_pct',
 'TV_GRP'],axis=1)


# In[523]:


x_validation=test_data[['Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price']]
y_validation_actual=test_data['EQ']


# In[524]:


x_validation = x_validation.div(x_validation.quantile(.99)).clip_upper(1)


# In[525]:


x_validation = sc.transform(x_validation)


# In[526]:


regressor=RandomForestRegressor(n_estimators=143,min_samples_leaf=1,n_jobs=-1,oob_score=True,random_state=0)
regressor.fit(x_train_new,y_train_new)
y_pred_validation=regressor.predict(x_validation)


# In[527]:


print("Root Mean Squared Error is:",np.sqrt(metrics.mean_squared_error(y_validation_actual,y_pred_validation)))


# #### Comment : The root mean square error of the validation set is 26.378 which is really good.

# ## Let's move on to Time Series Forecasting Model

# In[528]:


data.shape


# In[529]:


data_xreg1= pd.DataFrame(data=data, columns=['Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price'])
data_xreg2=pd.DataFrame(data=test_data,columns=['Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price'])

data_xreg_Final=data_xreg1.append(data_xreg2)


# In[530]:


data_xreg_Final.head()


# In[531]:


data_yreg1=pd.DataFrame(data=data, columns=['Period','EQ'])
data_yreg2=pd.DataFrame(data=test_data, columns=['Period','EQ'])
data_yreg_Final=data_yreg1.append(data_yreg2)


# In[532]:


data_yreg_Final.head()


# In[533]:


data_ARIMAX=pd.concat([data_yreg_Final,data_xreg_Final], axis=1, ignore_index=True)


# In[534]:


data_ARIMAX= pd.DataFrame(data=data, columns=['Period','EQ','Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price'])


# In[535]:


data_ARIMAX.shape


# In[536]:


from sklearn.preprocessing import MinMaxScaler


# In[537]:


scaler=MinMaxScaler()


# In[538]:


scaler.fit(data[['EQ','Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price']])



d=scaler.transform(data[['EQ','Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price']])


# In[539]:


d=pd.DataFrame(d,columns=['EQ','Inflation',
'Est_ACV_Selling',
'pct_ACV',
'Fuel_Price',
'Avg_no_of_Items',
'pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI',
'Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct',
'EQ_Subcategory',
'Trade_Invest',
'Avg_EQ_Price'])


# In[540]:


d[['EQ','Est_ACV_Selling',
'pct_ACV']].plot()
plt.show()


# #### We can clearly observe how the changes in Est_ACV_Selling and pct_ACV brings about a change in EQ in the same direction with certain time lags

# In[541]:


d[['EQ','Fuel_Price',
'Avg_no_of_Items']].plot()
plt.show()


# #### As fuel price increases over time EQ in the same manner tends to decrease over time.Thereby showing a particular trend in the opposite direction whereas Avg_no_of_Items over time doesn't show any significant trend

# In[542]:


d[['EQ','pct_PromoMarketDollars_Category',
'EQ_Base_Price',
'EQ_Category',
'Competitor1_RPI']].plot()
plt.show()


# In[543]:


d[['EQ','Competitor2_RPI',
'Any_Promo_pct_ACV',
'SOS_pct']].plot()
plt.show()


# In[544]:


from scipy.misc import factorial


# In[545]:


import statsmodels.api as sm
sm.tsa.stattools.adfuller(d['EQ'])


# ### Next Steps : I had thought of performing the Augmented Dicky Fuller Test (null hypothesis: The series is sationary,alternative hypothesis: the series is  not stationary) to check whether the series is stationary or not over time.If not I would have differenced the series and took the corresponding lags of the variables for ehich the series become stationary. I would then use the corresponding lags of the variables as the regressor variables in my ARIMA model and similar checks of stationarity I would do for my forecast variable. With the help of the ACF(Auto Correlation Function)plot I would get MA(Moving Average) order i.e. under which lag correlation between the event at time point 't' and time point 't-1' crosses +1 or -1 for the first time,similarly AR(Auto Regressive) order can be found from the Partial autocorrelation function (PACF) plot where we calculate the corrletaion of the event happening at time point t with respect to time point 't-1' keeping all the time points till 't-2' as constant.After we have run an ARIMAX model finally we will try to see the goodness of fit of the model with a part of the existing data (as in we forecast for the latest few time points of our existing data with respect to the model fit based on time points in the recent past).Then we calculate the MAPE(Mean Absolute Percentage Error) of the model.Generally a good value of MAPE should not be avove 15 percent.
