"""
Tabular Playground Series — Time Series Forecasting with XGBoost
A production-oriented time series forecasting project predicting multi-country retail book sales using calendar feature decomposition, lag features, and gradient-boosted regression.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/time-series-tps-eda-xgb-simplified
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Machine Learning Estimators
import xgboost as xgb

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set_style('darkgrid')

import warnings
warnings.filterwarnings("ignore")

# --- Cell 2 ---
df_train =pd.read_csv('../input/tabular-playground-series-sep-2022/train.csv',index_col=0,parse_dates=['date']) # index col is set to zero as it is the index, so no need to have 2 indexes, just set it as index.
df_train.head()

# --- Cell 3 ---
df_test =pd.read_csv('../input/tabular-playground-series-sep-2022/test.csv',index_col=0,parse_dates=['date'])
df_test.head()

# --- Cell 4 ---
df_train.info()

# --- Cell 5 ---
df_train.shape, df_test.shape 

# --- Cell 6 ---
#checking for null values in train data
df_train.isnull().sum()

# --- Cell 7 ---
#checking for null values in test data
df_test.isnull().sum()

# --- Cell 8 ---
#Checking for duplicate rows
df_train.duplicated().sum()

# --- Cell 9 ---
df_train.describe()

# --- Cell 10 ---
display('Train Data',df_train.describe(include='object')),display('Test Data',df_test.describe(include='object'))

# --- Cell 11 ---
# checking for timeline of train and test data
display('Train_data:' ,'---------',df_train.date.min(),'---------',df_train.date.max())

# --- Cell 12 ---
display('Test_data:','---------',df_test.date.min(),'---------',df_test.date.max())

# --- Cell 13 ---
df_train.nunique()

# --- Cell 14 ---
countries = df_train['country'].unique().tolist()
countries

# --- Cell 15 ---
products = df_train['product'].unique().tolist()
products

# --- Cell 16 ---
stores =  df_train['store'].unique().tolist()
stores

# --- Cell 17 ---
df_train.head()

# --- Cell 18 ---
df_train["year"] = df_train["date"].dt.year
df_train["month"] = df_train["date"].dt.month
df_train["day_of_week"] = df_train["date"].dt.dayofweek
df_train["day_of_year"] = df_train["date"].dt.dayofyear

# --- Cell 19 ---
df_test["year"] = df_test["date"].dt.year
df_test["month"] = df_test["date"].dt.month
df_test["day_of_week"] = df_test["date"].dt.dayofweek
df_test["day_of_year"] = df_test["date"].dt.dayofyear

# --- Cell 20 ---
sns.histplot(df_train['num_sold']);

# --- Cell 21 ---
product_df = df_train.groupby(["date","product"])["num_sold"].sum().reset_index()
product_df

# --- Cell 22 ---
product_df.describe()

# --- Cell 23 ---
plt.figure(figsize=(12,8))
sns.lineplot(data=product_df, x="date", y="num_sold", hue="product");

# --- Cell 24 ---
sns.boxplot(x='store',y='num_sold',data=df_train);

# --- Cell 25 ---
sns.boxplot(x='product',y='num_sold',data=df_train)
plt.xticks(rotation=90);

# --- Cell 26 ---
sns.boxplot(x='country',y='num_sold',data=df_train);

# --- Cell 27 ---
data=df_train.groupby('country').sum('num_sold')
sns.barplot(data=data,x=data.index,y='num_sold')
plt.title('Total Sales by Country');

# --- Cell 28 ---
weekly_df = df_train.groupby(["country","store", "product", pd.Grouper(key="date", freq="W")])["num_sold"].sum().rename("num_sold").reset_index()
monthly_df = df_train.groupby(["country","store", "product", pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()

# --- Cell 29 ---
def plot_all(df):
    f,axes = plt.subplots(2,2,figsize=(20,15), sharex = True, sharey=True)
    f.tight_layout()
    for n,prod in enumerate(df["product"].unique()):
        plot_df = df.loc[df["product"] == prod]
        sns.lineplot(data=plot_df, x="date", y="num_sold", hue="country", style="store",ax=axes[n//2,n%2])
        axes[n//2,n%2].set_title("Product: "+str(prod))

# --- Cell 30 ---
plot_all(weekly_df)

# --- Cell 31 ---
plot_all(monthly_df)

# --- Cell 32 ---
df_train.info()

# --- Cell 35 ---
train = pd.get_dummies(df_train, columns=['country', 'store', 'product'], drop_first=True)
test =  pd.get_dummies(df_test, columns=['country', 'store', 'product'], drop_first=True)

# --- Cell 36 ---
# Splitting the data
X = train.drop(['num_sold', 'date'], axis=1)
y = train.num_sold

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

# --- Cell 37 ---
# Function to calculate metric results
def calculate_results(y_true, y_pred):
    # Calculate model Mean Absolute Error (MAE)
    model_mae = mean_absolute_error(y_val, y_pred)
    # Calculate model Mean Squrared Error (MSE)
    model_mse = mean_squared_error(y_val, y_pred)
    # Calculate model Root Mean Squared Error (RMSE)
    model_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    # Calculate Adjusted R2_score
    model_r2 = r2_score(y_val, y_pred)
    # Calculate Root Mean Squared Log Error
    model_rmsle = np.log(np.sqrt(mean_squared_error(y_val, y_pred)))
    
    
    model_results = {"Mean Absolute Error (MAE)": model_mae,
                     "Mean Squared Error (MSE)": model_mse,
                     "Root Mean Squared Error (RMSE)": model_rmse,
                     "Adjusted R^2 Score": model_r2,
                     "Root Mean Squared Log Error": model_rmsle}
    return model_results

# --- Cell 38 ---
# Predict using Ridge Regression
xgb_model = xgb.XGBRegressor(params={'regressor__n_estimators': 120, 'regressor__max_depth': 6, 'regressor__learning_rate': 0.1})
y_pred = xgb_model.fit(X_train, y_train).predict(X_val)
    
xgb_results = calculate_results(y_val, y_pred)
pd.DataFrame(xgb_results ,index=['values']).T

# --- Cell 39 ---
predictions= xgb_model.fit(X,y).predict(test.drop('date',axis=1))

# --- Cell 40 ---
# Feature importances
pd.DataFrame({'Feature': X.columns,'Importance': xgb_model.feature_importances_}).sort_values(by=['Importance'], ascending=False).reset_index(drop=True)

# --- Cell 41 ---
submissions=pd.read_csv('../input/tabular-playground-series-sep-2022/sample_submission.csv')
submissions['num_sold']=predictions//1

# --- Cell 42 ---
submissions.head()

# --- Cell 43 ---
submissions.to_csv('submission.csv', index = False)



if __name__ == "__main__":
    print("Pipeline execution complete.")
