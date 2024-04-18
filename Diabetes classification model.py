#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Healthcare-Diabetes.csv')


# In[3]:


sns.set_theme(rc={'figure.dpi': 100, 'figure.figsize': (2,2)})
sns.countplot(x='Outcome',data=df)


# In[4]:


df.columns


# In[5]:


sns.set_theme(rc={'figure.dpi': 250, 'figure.figsize': (4,4)})
sns.pairplot(df[['Insulin', 'Age', 'Outcome']],hue='Outcome')


# In[6]:


sns.set_theme(rc={'figure.dpi': 250, 'figure.figsize': (4,4)})
sns.pairplot(df[['Pregnancies', 'DiabetesPedigreeFunction','Outcome']],hue='Outcome')


# In[7]:


sns.set_theme(rc={'figure.dpi': 250, 'figure.figsize': (4,4)})
sns.pairplot(df[['Insulin','Outcome']],hue='Outcome')


# In[8]:


sns.set_theme(rc={'figure.dpi': 250, 'figure.figsize': (4,4)})
sns.pairplot(df[['Age','BMI','Outcome']],hue='Outcome')


# In[9]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)


# In[10]:


X = df.drop(['Outcome','Id'],axis=1)
y = df['Outcome']


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[13]:


scaler = StandardScaler()


# In[14]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[15]:


from sklearn.linear_model import LogisticRegressionCV 


# In[16]:


log_model = LogisticRegressionCV()


# In[17]:


log_model.fit(scaled_X_train,y_train)


# In[18]:


log_model.C_


# In[19]:


log_model.get_params()


# In[20]:


log_model.coef_


# In[21]:


coefs = pd.Series(index=X.columns,data=log_model.coef_[0])


# In[22]:


coefs = coefs.sort_values()


# sns.set_theme(rc={'figure.dpi': 250, 'figure.figsize': (6,6)})
# sns.barplot(x=coefs.index,y=coefs.values);

# In[23]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix


# In[24]:


y_pred = log_model.predict(scaled_X_test)


# In[25]:


confusion_matrix(y_test,y_pred)


# In[33]:


plt.figure(figsize=(10,6))
sns.barplot(x=coefs.index,y=coefs.values);


# In[27]:


print(classification_report(y_test,y_pred))


# In[30]:


patient = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]


# In[31]:


log_model.predict(patient)


# In[32]:


log_model.predict_proba(patient)


# In[34]:


plot_confusion_matrix(log_model,scaled_X_test,y_test)


# In[ ]:




