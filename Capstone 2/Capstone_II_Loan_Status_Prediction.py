#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


# Load the dataset
url = "https://raw.githubusercontent.com/thendralvanans/Thinkful-Project/main/Capstone%202/loan_data.csv" 
df = pd.read_csv(url)
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# # Handling the missing values

# In[5]:


df.isnull().sum()


# In[6]:


df=df.drop(["Loan_ID"],axis=1)
df['Gender'] = df['Gender'].fillna(df['Gender'].mode().iloc[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode().iloc[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median()).astype(int)
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode().iloc[0]).astype(int)
df['Dependents'] = df['Dependents'].replace(['0', '1', '2', '3+'], [0,1,2,3])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode().iloc[0])
df['CoapplicantIncome'] = df['CoapplicantIncome'].astype(int)
df['LoanAmount'] = df['LoanAmount'].astype(int)
df['Dependents'] = df['Dependents'].astype(int)


# In[7]:


df.isnull().sum()


# In[8]:


for column_name in df.columns:
    print("Unique values in column {} are: {}".format(column_name, df[column_name].unique()))


# # Exploratory data analysis

# In[9]:


categorical_columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
fig, ax = plt.subplots(2, 3, figsize=(10,7))

for index, cat_col in enumerate(categorical_columns):
    row, col = index//3, index%3
    sns.countplot(x=cat_col, data=df, hue='Loan_Status', ax=ax[row, col], palette='Set1')

plt.suptitle('Loan Approval Distribution Across Categorical Variables', fontsize=16, y=1.02)
plt.subplots_adjust(hspace=1)


# In[10]:


numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

fig, axes = plt.subplots(1, 4, figsize=(25, 7))
for idx, num_col in enumerate(numerical_columns):
    sns.boxplot(x='Loan_Status', y=num_col, data=df, ax=axes[idx], palette='Set1')
    axes[idx].set_title(f'Distribution of {num_col}', fontsize=14)
    axes[idx].set_ylabel(num_col, fontsize=12)
plt.suptitle('Distribution of Numerical Variables by Loan Approval', fontsize=30, y=1.02)
plt.subplots_adjust(wspace=0.4)
print(df[numerical_columns].describe())

plt.show()


# # Converting categorical data into numerical form

# In[11]:


convert_columns = ['Gender', 'Married', 'Education','Self_Employed', 'Loan_Status']
for col in convert_columns:
    uniques_value = df[col].unique()
    df[col].replace(uniques_value, [0, 1], inplace=True)
df['Property_Area'].replace(df['Property_Area'].unique(), [0, 1, 3], inplace=True)


# In[12]:


for column_name in df.columns:
    print("Unique values in column {} are: {}".format(column_name, df[column_name].unique()))


# In[13]:


plt.figure(figsize=(10,8))
corr_df = df.corr()
sns.heatmap(corr_df,annot=True)


# #    Preprocessing Data

# In[14]:


X = df.drop(['Loan_Status'], axis=1)
y = df['Loan_Status']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[17]:


X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# In[18]:


X_train


# # Model Selection

# ## DecisionTreeClassifier

# In[19]:


start = time.time()
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
y_pred_dt = model_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
roc_score_dt = roc_auc_score(y_test, y_pred_dt)
f1_score_dt = f1_score(y_test, y_pred_dt)
precision_score_dt = precision_score(y_test, y_pred_dt)
recall_score_dt = recall_score(y_test, y_pred_dt)
print(f'Accuracy Score: {accuracy_dt:0.2f}')
print(f'F1 Score: {f1_score_dt:0.2f}')
print(f'Precision Score: {precision_score_dt:0.2f}')
print(f'Recall Score: {recall_score_dt:0.2f}')
print('Elapsed Time:',time.time()-start)


# ## LogisticRegression

# In[20]:


start = time.time()
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
roc_score_lr = roc_auc_score(y_test, y_pred_lr)
f1_score_lr = f1_score(y_test, y_pred_lr)
precision_score_lr = precision_score(y_test, y_pred_lr)
recall_score_lr = recall_score(y_test, y_pred_lr)
print(f'Accuracy Score: {accuracy_lr:0.2f}')
print(f'F1 Score: {f1_score_lr:0.2f}')
print(f'Precision Score: {precision_score_lr:0.2f}')
print(f'Recall Score: {recall_score_lr:0.2f}')
print('Elapsed Time:',time.time()-start)


# ## RandomForestClassifier

# In[21]:


start = time.time()
model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_score_rf = roc_auc_score(y_test, y_pred_rf)
f1_score_rf = f1_score(y_test, y_pred_rf)
precision_score_rf = precision_score(y_test, y_pred_rf)
recall_score_rf = recall_score(y_test, y_pred_rf)
print(f'Accuracy Score: {accuracy_rf:0.2f}')
print(f'F1 Score: {f1_score_rf:0.2f}')
print(f'Precision Score: {precision_score_rf:0.2f}')
print(f'Recall Score: {recall_score_rf:0.2f}')
print('Elapsed Time:',time.time()-start)


# # Hyperparameter Tuning

# ## DecisionTreeClassifier

# In[22]:


start = time.time()
# setup parameter space
parameters = {'criterion':['gini','entropy'],
              'max_depth':np.arange(1,21).tolist()[0::2],
              'min_samples_split':np.arange(2,11).tolist()[0::2],
              'max_leaf_nodes':np.arange(3,26).tolist()[0::2]}

# create an instance of the grid search object
gs_dt = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_jobs=-1)

# conduct grid search over the parameter space
gs_dt.fit(X_train,y_train)

# show best parameter configuration found for classifier
params_dt = gs_dt.best_params_
print('The best parameters : ',params_dt)

# compute performance on test set
model = gs_dt.best_estimator_
y_pred = model.predict(X_test)
print('accuracy score: %.2f' % accuracy_score(y_test,y_pred))
print('precision score: %.2f' % precision_score(y_test,y_pred))
print('recall score: %.2f' % recall_score(y_test,y_pred))
print('f1 score: %.2f' % f1_score(y_test,y_pred))
print('Elapsed Time:',time.time()-start)


# ## LogisticRegression

# In[23]:


start = time.time()
parameters = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [10,100,1000,2500, 5000]
    }
]

gs_lr = GridSearchCV(LogisticRegression(), parameters, cv = 5, n_jobs=-1)

gs_lr.fit(X_train,y_train)

params_lr = gs_lr.best_params_
print('The best parameters : ',params_lr)

model_lr = gs_lr.best_estimator_
y_pred_lr = model_lr.predict(X_test)
print('accuracy score: %.2f' % accuracy_score(y_test,y_pred_lr))
print('precision score: %.2f' % precision_score(y_test,y_pred_lr))
print('recall score: %.2f' % recall_score(y_test,y_pred_lr))
print('f1 score: %.2f' % f1_score(y_test,y_pred_lr))
print('Elapsed Time:',time.time()-start)


# ## RandomForestClassifier

# In[24]:


start = time.time()
parameters_rf = {'n_estimators': [5,20,50,100],
'max_features': ['auto', 'sqrt'],
'max_depth': [10,20,30],
'min_samples_split': [2, 6, 10],
'min_samples_leaf': [1, 3, 4],
'bootstrap': [True, False]}

gs_rf = GridSearchCV(RandomForestClassifier(), parameters_rf, cv = 5, n_jobs=-1)

gs_rf.fit(X_train,y_train)

params_rf = gs_rf.best_params_
print('The best parameters : ',params_rf)

model_rf = gs_rf.best_estimator_
y_pred_rf = model_rf.predict(X_test)
print('accuracy score: %.2f' % accuracy_score(y_test,y_pred_rf))
print('precision score: %.2f' % precision_score(y_test,y_pred_rf))
print('recall score: %.2f' % recall_score(y_test,y_pred_rf))
print('f1 score: %.2f' % f1_score(y_test,y_pred_rf))
print('Elapsed Time:',time.time()-start)


# The LogisticRegression seems to be best model for this dataset which run in shorter elapsed time. And the best parameters are {'C': 0.03359818286283781, 'max_iter': 10, 'penalty': 'l2', 'solver': 'sag'}

# In[ ]:




