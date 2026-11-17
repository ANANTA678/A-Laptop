#1. run the code
#2. in terminal run this code->  python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly xgboost lightgbm statsmodels joblib streamlitimport joblib
#3. 
import pandas as pd
import matplotlib.pyplot as plt                     
import seaborn as sns            
import numpy as np              

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: "{:.2f}".format(x))
np.set_printoptions(suppress=True)
df_customers = pd.read_csv(r'D:\Practice\Project 2\customers.csv')
df_loans = pd.read_csv(r"D:\Practice\Project 2\loans.csv")
df_bureau = pd.read_csv(r'D:\Practice\Project 2\bureau_data.csv')

print(df_customers.shape, df_loans.shape, df_bureau.shape )
print(df_customers.head(2), df_loans.head(2), df_bureau.head(2) )
df=pd.merge(df_customers, df_loans, on = "cust_id").merge(df_bureau, on = 'cust_id')
df
df.info()
df['default'] = df['default'].astype('int')
df['default'].value_counts()
df.isnull().sum()
df.residence_type.mode()[0]
df['residence_type'] = df.residence_type.fillna(value=df.residence_type.mode()[0], inplace=True)
df.processing_fee.describe()
df[ df.processing_fee == df.processing_fee.max()] [['loan_amount', 'processing_fee']]
df[ df.processing_fee > df.loan_amount] [['loan_amount', 'processing_fee']]
# processing_fee on the loan amount is 3% only 

df[ (df.processing_fee/df.loan_amount) > 0.03 ][['loan_amount', 'processing_fee']]
df = df[ (df.processing_fee/df.loan_amount) < 0.03 ]
# Use other business rules for data validation
# Rule 1: GST should not be more than 20%

df[(df.gst/df.loan_amount)>0.2].shape
# Rule 2: Net disbursement should not be higher than loan_amount
df[df.net_disbursement>df.loan_amount].shape
for col in df.select_dtypes(include="object").columns:
    print("\nColumn Name -:", df[col].unique())
   
### Fix Errors in Loan Purpose Column
df['loan_purpose'] = df['loan_purpose'].replace('Personaal', 'Personal')
df['loan_purpose'].unique()
df['loan_purpose'] = df['loan_purpose'].replace('Personaal', 'Personal')
df['loan_purpose'].unique()
df.groupby("default")['age'].describe()
columns_continuous = df.select_dtypes(exclude="object").columns
df["loan_to_income"] = round(df['loan_amount']/df["income"],2)
df["loan_to_income"].describe()
df['delinquent_ratio'] = (df["delinquent_months"] *100 / df["total_loan_months"] ).round(2)
df[["delinquent_months", "total_loan_months", 'delinquent_ratio']].head(4)
df['avg_dpd_per_delinquency'] = np.where(
     df['delinquent_months'] != 0,
    (df['total_dpd'] / df['delinquent_months']).round(2),
    0)

df.columns
df = df.drop(['cust_id', 'loan_id', 'disbursal_date', 'installment_start_dt', 'loan_amount', 'income', 
                              'total_loan_months', 'delinquent_months', 'total_dpd'], axis="columns")
df.columns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif_df
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    # 1. Create a new DataFrame using a dictionary for cleaner syntax
    vif_df = pd.DataFrame({"Feature": data.columns})
    
    # 2. Calculate the VIF score for each column index
    vif_df["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    
    # 3. Sort values to easily see the highest multicollinearity at the top
    return vif_df.sort_values(by="VIF", ascending=False).reset_index(drop=True)
data = df.select_dtypes(['int64', 'float64'])
calculate_vif(data)
df = df.drop(columns = ['sanction_amount', 'processing_fee', 'gst', 'net_disbursement','principal_outstanding'])
df.head(1)

df.head(2)
def calculate_woe_iv(df, feature, target):
    grouped = df.groupby(feature)[target].agg(['count','sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'good'})
    grouped['bad']=grouped['total']-grouped['good']
    
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    grouped['good_pct'] = grouped['good'] / total_good
    grouped['bad_pct'] = grouped['bad'] / total_bad
    grouped['woe'] = np.log(grouped['good_pct']/ grouped['bad_pct'])
    grouped['iv'] = (grouped['good_pct'] -grouped['bad_pct'])*grouped['woe']
    
    grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0)
    grouped['iv'] = grouped['iv'].replace([np.inf, -np.inf], 0)
    
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv

grouped, total_iv = calculate_woe_iv(df, 'loan_purpose', 'default')
grouped
df.head(1)
X_binned = pd.cut(df['age'], bins=50, labels=False)
X_binned
df.dtypes
iv_values = {}

for feature in df.columns:
    if df[feature].dtype in('str','object'):
        _, iv = calculate_woe_iv(df, feature, 'default' )
    else:
        _, iv = calculate_woe_iv(df, feature, 'default' )
    iv_values[feature] = iv
        
iv_values
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV'])
iv_df = iv_df.sort_values(by='IV', ascending=False)
iv_df
# select features that has IV > 0.02
selected_features_iv = [feature for feature, iv in iv_values.items() if iv > 0.02]
selected_features_iv
data = df[selected_features_iv + ['default']]
data

for col in data.select_dtypes(include='object').columns:
    print(data[col].unique())

data = pd.get_dummies(data)
data
X = data.drop('default', axis=1)
Y = data["default"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.25, stratify=Y)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. Define models and their respective hyperparameter grids
models = {
    "LogisticRegression": {
        "model": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(class_weight='balanced', random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [1, 2, 5] # Helps with class imbalance
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42, verbose=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.1, 0.2],
            "is_unbalance": [True, False] # Helps with class imbalance
        }
    }
}

best_score = 0
best_model_name = ""
ultimate_best_model = None
trained_models = {}

# 2. Iterate, Train, and Tune
for model_name, config in models.items():
    print(f"Training and Tuning {model_name}...")
    
    # Using RandomizedSearchCV for efficient hyperparameter exploration
    search = RandomizedSearchCV(
        estimator=config["model"], 
        param_distributions=config["params"], 
        n_iter=10,        # Number of parameter settings that are sampled
        cv=3,             # 3-fold cross-validation
        scoring='roc_auc',# Optimize for ROC-AUC due to imbalanced data
        random_state=42,
        n_jobs=-1         # Use all available CPU cores
    )
    
    # Fit the search model
    search.fit(X_train, y_train)
    trained_models[model_name] = search.best_estimator_
    
    # Evaluate on the unseen test set
    y_pred = search.predict(X_test)
    y_pred_proba = search.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  -> Best Params: {search.best_params_}")
    print(f"  -> Test ROC-AUC: {auc:.4f}\n")
    
    # Track the absolute best performing model
    if auc > best_score:
        best_score = auc
        best_model_name = model_name
        ultimate_best_model = search.best_estimator_

# 3. Final Evaluation of the Best Model
print("="*50)
print(f"🏆 Overall Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
print("="*50)

print("\nClassification Report for Best Model:")
y_pred_best = ultimate_best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# 4. Export the Model using Joblib
joblib_filename = f"best_model_{best_model_name.lower()}.joblib"
joblib.dump(ultimate_best_model, joblib_filename)
print(f"✅ Best model successfully exported to disk as: '{joblib_filename}'")
