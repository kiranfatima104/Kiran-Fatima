#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

file = pd.read_csv("customer_data.csv")


# In[2]:


file.head()


# In[3]:


file.describe()


# In[4]:


seasons = file['product_category'].unique()
average_purchase_season = file.groupby('product_category')['purchase_amount'].mean()
average_purchase_season.plot(kind = "bar", title = "Purchase Amount by category", ylabel = "Average Purchase_Amount", xlabel = "product category", color = ['blue','pink','red','purple', 'green', 'yellow', 'orange'])
#plt.savefig('bar_plot.png', dpi=300)  # Set the DPI value as needed


# In[5]:


hist = px.histogram(file, x='age', title='Distribution of Age')
hist.update_layout(
    title={'text': 'Distribution of age', 'font': {'size': 20}},  # Change the font size of the title
    xaxis={'title': 'age distribution', 'title_font': {'size': 30}},  # Change the font size of the x-axis label
    yaxis={'title': 'count', 'title_font': {'size': 20}}  # Change the font size of the y-axis label
)
hist.show()


# In[6]:


# Bar chart for 'Gender'
gender_counts = file['gender'].value_counts().reset_index()
gender_counts.columns = ['gender', 'Count']
barchart = px.bar(gender_counts, x='gender', 
             y='Count', 
             title='Gender Distribution')
barchart.show()


# In[7]:


#no. of purchases per region
linfo = file.groupby("gender")["purchase_amount"].agg(["mean"])


# In[8]:


linfo


# In[9]:


satisfaction_avg = file.groupby('promotion_usage')['satisfaction_score'].mean().reset_index()


# In[10]:


satisfaction_avg


# In[11]:


devices_grouped = file.groupby('loyalty_status')['purchase_frequency'].count().reset_index()
devices_grouped.columns = ['loyalty_status', 'purchase_frequency']
fig = px.bar(devices_grouped, x='loyalty_status', y='purchase_frequency', color= 'loyalty_status', color_discrete_map={'Gold': 'blue', 'Regular': 'green', 'Silver': 'red', 'D': 'blue'}, title='Number of visits by Loyalty status')
fig.update_layout(
    title={'text': 'Number of visits by loyalty status', 'font': {'size': 20}},  # Change the font size of the title
    xaxis={'title': 'loyalty status', 'title_font': {'size': 30}},  # Change the font size of the x-axis label
    yaxis={'title': 'purchase frequency', 'title_font': {'size': 20}}  # Change the font size of the y-axis label
)
fig.show()


# In[ ]:




