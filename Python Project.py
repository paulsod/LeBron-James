from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import numpy.random as npr
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
path = 'C:\Users\User\Documents\Actuary\Fourth Year\Python\Project\LeBron/'
#READ IN DATA

LB16_17 = pd.read_csv(path+'LeBron 2016-2017.csv')
LB15_16 = pd.read_csv(path+'LeBron 2015-2016.csv')
LB14_15 = pd.read_csv(path+'LeBron 2014-2015.csv')
LB13_14 = pd.read_csv(path+'LeBron 2013-2014.csv')
LB12_13 = pd.read_csv(path+'LeBron 2012-2013.csv')
LB11_12 = pd.read_csv(path+'LeBron 2011-2012.csv')
LB10_11 = pd.read_csv(path+'LeBron 2010-2011.csv')
LB09_10 = pd.read_csv(path+'LeBron 2009-2010.csv')
LB08_09 = pd.read_csv(path+'LeBron 2008-2009.csv')
LB07_08 = pd.read_csv(path+'LeBron 2007-2008.csv')
LB06_07 = pd.read_csv(path+'LeBron 2006-2007.csv')
LB05_06 = pd.read_csv(path+'LeBron 2005-2006.csv')
LB04_05 = pd.read_csv(path+'LeBron 2004-2005.csv')
LB03_04 = pd.read_csv(path+'LeBron 2003-2004.csv')
#MERGE EACH SEASON'S STATA
LeBron = LB03_04.append([LB04_05,LB05_06,LB06_07,LB07_08,LB08_09,LB09_10,LB10_11,LB11_12,LB12_13,LB13_14,LB14_15,LB15_16,LB16_17])

#DATA VALIDATION
LeBron = LeBron.rename(columns = {'Unnamed: 5':'Home/Away','Unnamed: 7':'Margin'})#add missing headers
LeBron = LeBron[np.isfinite(LeBron['G'])]
#REMOVE UNNECESSARY COLUMNS
del LeBron['+/-']
del LeBron['GmSc']
del LeBron['G']
del LeBron['Tm']
del LeBron['Age']
del LeBron['Date']
del LeBron['GS']
del LeBron['FT%']
del LeBron['3P%']
del LeBron['FG%']
del LeBron['MP']

LeBron['Home/Away'].fillna('H',inplace = True)#make category into home and away games
LeBron.isnull().sum()#check there are no more Nulls
LeBron = LeBron.reset_index(drop=True)
import re
#EDIT THE STRINGS IN MARGIN COLUMN
LeBron['Margin'].replace('[^0-9+-]+','',inplace=True,regex=True) # does by using regex
LeBron


#stacked plot showing contribution of each type of scoring to total across his career 
import matplotlib.patches as mpatches
plt.figure(figsize=(32,24))
plt.stackplot(LeBron.index, LeBron['FT'],(LeBron['FG']-LeBron['3P'])*2,LeBron['3P']*3, colors=['m','c','r'])#multiplied by 2 and 3 to show contribution to scoring total
plt.xlabel('Games')
plt.ylabel('Points')
plt.title('Scoring Breakdown')
plt.legend([mpatches.Patch(color='m'),  mpatches.Patch(color='c'), mpatches.Patch(color='r')], ['FTs','Two Pointers','Three Pointers'])
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Python\Project\LeBron/Scoring_Breakdown',bbox_inches='tight')
plt.show()


del LeBron['FT']
del LeBron['3P']
del LeBron['FG']
#SUMMARIES
LeBron.corr()
LeBron.describe()

#get dummies for Home/Away, Opponent
dummies = pd.get_dummies(LeBron[['Home/Away','Opp']])
dummies = dummies.drop(['Home/Away_@','Opp_ATL'],axis=1)#drop one entry from each to avoid dummy variable trap: perfect multicollinearity
X = pd.concat([LeBron, dummies],axis =1) 
X = X.drop(['Home/Away','Opp'],axis=1)

#Linear Regressions create X and Y
X.corr()
X = X.apply(pd.to_numeric)
X = (X - X.mean())/X.std()
y = X['PTS']
X_reg = sm.add_constant(X.drop('PTS',axis=1))

########MACHINE LEARNING
from sklearn import linear_model
train_size = 746 #75% of data used for training
np.random.seed(123)
train_select = np.random.permutation(range(len(y)))
X_train = X_reg.ix[train_select[:train_size],:].reset_index(drop=True) #training set to train the package
X_test = X_reg.ix[train_select[train_size:],:].reset_index(drop=True) # test set to see how accurate predictions are
y_train = y[train_select[:train_size]].reset_index(drop=True)
y_test = y[train_select[train_size:]].reset_index(drop=True)
#LINREG
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
reg.score(X_test, y_test)#returns R^2
reg_test_pred = reg.predict(X_test)
RSS_reg = np.mean(pow((reg_test_pred - y_test),2))#residual SS

#LASSO
lasso = linear_model.Lasso(alpha = 0.05)
lasso.fit(X_train,y_train)
lasso.score(X_test, y_test)
lasso_test_pred = lasso.predict(X_test)
RSS_lasso = np.mean(pow((lasso_test_pred - y_test),2))

#RF REGRESSOR
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X_train,y_train)
RF.score(X_test, y_test)
RF_test_pred = RF.predict(X_test)
RSS_RF = np.mean(pow((RF_test_pred - y_test),2))

#### LARS
lars = linear_model.Lars(n_nonzero_coefs=9)
lars.fit(X_train,y_train)
lars.score(X_test, y_test)
lars_test_pred = lars.predict(X_test)
RSS_lars = np.mean(pow((lars_test_pred - y_test),2))


#LARS Lasso - not good
lars_lasso = linear_model.LassoLars(alpha= 1)
lars_lasso.fit(X_train,y_train)
lars_lasso.score(X_test,y_test)
lars_lasso_test_pred = lars_lasso.predict(X_test)
RSS_lars_lasso = np.mean(pow((lars_lasso_test_pred - y_test),2))

#SVM
from sklearn.svm import SVR
SVM = SVR(C = 0.5)
SVM.fit(X_train, y_train) 
SVM.score(X_test, y_test)
SVM_test_pred = SVM.predict(X_test)
RSS_SVM = np.mean(pow((SVM_test_pred - y_test),2))

#kernel ridge
from sklearn.kernel_ridge import KernelRidge
KR = KernelRidge(alpha = 1)
KR.fit(X_train, y_train) 
KR.score(X_test, y_test)
KR_test_pred = KR.predict(X_test)
RSS_KR = np.mean(pow((KR_test_pred - y_test),2))

#neural network 
from sklearn.neural_network import MLPRegressor
NN = MLPRegressor(hidden_layer_sizes = 20)
NN.fit(X_train, y_train) 
NN.score(X_test, y_test)
NN_test_pred = NN.predict(X_test)
RSS_NN = np.mean(pow((NN_test_pred - y_test),2))


#Huber Regression
Hub = linear_model.HuberRegressor(epsilon = 1)
Hub.fit(X_train, y_train) 
Hub.score(X_test, y_test)
Hub_test_pred = Hub.predict(X_test)
RSS_Hub = np.mean(pow((Hub_test_pred - y_test),2))

#SGDRegressor
SGD = linear_model.SGDRegressor(loss = 'epsilon_insensitive')
SGD.fit(X_train, y_train) 
SGD.score(X_test, y_test)
SGD_test_pred = SGD.predict(X_test)
RSS_SGD = np.mean(pow((SGD_test_pred - y_test),2))


#GRADIENT BOOSTER
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(loss = 'ls',n_estimators = 80)
GBR.fit(X_train, y_train) 
GBR.score(X_test, y_test)
GBR_test_pred = GBR.predict(X_test)
RSS_GBR = np.mean(pow((GBR_test_pred - y_test),2))



#######TIME SERIES######

pts = Series(LeBron['PTS']) 
plt.figure(figsize=(20,4))
plot(pts)
from statsmodels.tsa.stattools import acf 
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

#check for seasonality to do with the 82 game season. not significant
decomposition = seasonal_decompose(np.array(pts), freq=82) 
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


from statsmodels.tsa.stattools import adfuller
#function for assessing stationarity
def test_stationarity(timeseries):
    
    
    rolmean = pd.rolling_mean(timeseries, window=82)
    rolstd = pd.rolling_std(timeseries, window=82)

    #Plot statistics for graphs:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Python\Project\LeBron/Stationarity',bbox_inches='tight')
    plt.show()
    
    #Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

test_stationarity(pts)

pts_log= pts.apply(lambda x: np.log(x))  
test_stationarity(pts_log)

pts['first_difference'] = pts - pts.shift(1)  
test_stationarity(pts.first_difference.dropna(inplace=False))#first difference looks best

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(pts.first_difference.dropna(inplace=False), nlags=10)
lag_pacf = pacf(pts.first_difference.dropna(inplace=False), nlags=10, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='maroon')
plt.axhline(y=-1.96/np.sqrt(len(pts.first_difference)),linestyle='--',color='gold')
plt.axhline(y=1.96/np.sqrt(len(pts.first_difference)),linestyle='--',color='gold')
plt.xlabel('Lags')
plt.ylabel('Correlation')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122) 
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='maroon')
plt.axhline(y=-1.96/np.sqrt(len(pts.first_difference)),linestyle='--',color='gold')
plt.axhline(y=1.96/np.sqrt(len(pts.first_difference)),linestyle='--',color='gold')
plt.xlabel('Lags')
plt.ylabel('Correlation')
plt.title('Partial Autocorrelation Function')
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Python\Project\LeBron/acf',bbox_inches='tight')

#ARIMA MODEL
from statsmodels.tsa.arima_model import ARIMA
pts_first_diff_array = np.array(pts.first_difference.dropna(inplace=False))
model = ARIMA(pts_first_diff_array, order=(1, 0, 0)) 
results_AR = model.fit() 
print results_AR.summary() 
plt.plot(pts.first_difference)
plt.plot(results_AR.fittedvalues, color='red')

pts['predict'] = Series(results_AR.predict(start = 950, end = 994))

#plot prediction against last 45 games
plt.figure(facecolor='white')
plt.plot(pts.first_difference[:45],lw=2)#with added line width
plt.plot(pts['predict'],lw=2) 
plt.title('ARIMA Forecast on Differenced Series')
plt.xlabel('Final 45 Games')
plt.ylabel('Differences in Points Scored')
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Python\Project\LeBron/ARIMA Prediction',bbox_inches='tight')



