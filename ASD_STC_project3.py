#!/usr/bin/env python
# coding: utf-8

# # Statistical and Computational Lab
# 
# ### Collaborators
#     Siddharth Das     (2017120011)
#     Anushka Chintrate (2018220066)
#     Sonal Kamble      (2018220068)
# #### Mentor
#     Prof. Dayanand Ambawade

# ## START

# In[1]:


import os
os.getcwd()


# In[2]:


get_ipython().system(u'pip install docx2txt')


# In[3]:


import docx2txt
description = docx2txt.process('./Autism-Screening-Adult-Data Description.docx')
print(description)


# ### Load python modules 

# In[4]:


import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[5]:


import warnings
warnings.simplefilter('ignore')


# ### Load dataset

# In[6]:


# Do not use 'DATASET' directly; make a copy to avoid making changes to the original file.
DATASET = pd.read_csv('./Toddler Autism dataset July 2018.csv')


# In[7]:


# For an added layer of abstract protection
data = DATASET

# Use this dataframe
df = data


# ### EDA

# In[8]:


df.head()


# In[9]:


df.columns


# In[10]:


df.info()


# ### Label encode

# In[11]:


# Label encoding helps in handling the categorical variables elegantly

print('\nWho completed the test:- \n\t',
      df['Who completed the test'].unique(),
      '\nFamily_mem_with_ASD:- \n\t',
      df['Family_mem_with_ASD'].unique(),
      '\nJaundice:- \n\t',
      df['Jaundice'].unique(),
      '\nEthnicity:- \n\t',
      df['Ethnicity'].unique(),
      '\nSex:- \n\t',
      df['Sex'].unique(),
      '\nQchat-10-Score:- \n\t',
      df['Qchat-10-Score'].unique(),
      '\nAge_Mons:- \n\t',
      df['Age_Mons'].unique())


# In[12]:


# Import label encoder 
from sklearn import preprocessing 
 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'species'. 
print(df['Sex'].unique())
df['Sex']= label_encoder.fit_transform(df['Sex']) 
print(df['Sex'].unique(), '\n')

print(df['Ethnicity'].unique())
df['Ethnicity']= label_encoder.fit_transform(df['Ethnicity']) 
print(df['Ethnicity'].unique(), '\n')

print(df['Jaundice'].unique())
df['Jaundice']= label_encoder.fit_transform(df['Jaundice']) 
print(df['Jaundice'].unique(), '\n')

print(df['Family_mem_with_ASD'].unique())
df['Family_mem_with_ASD']= label_encoder.fit_transform(df['Family_mem_with_ASD']) 
print(df['Family_mem_with_ASD'].unique(), '\n')

print(df['Who completed the test'].unique())
df['Who completed the test']= label_encoder.fit_transform(df['Who completed the test']) 
print(df['Who completed the test'].unique(), '\n')


# In[13]:


df.sample(n=5)


# ### Logistic regression

# In[14]:


# split dataset in features and target variable
feature_cols = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','Age_Mons', 
                'Qchat-10-Score', 'Sex', 'Ethnicity','Jaundice','Family_mem_with_ASD','Who completed the test','Class/ASD Traits ']

# Features
X = df.iloc[:,1:17]

# Target variable
y = df.iloc[:,18]

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# In[15]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()


# In[16]:


# fit the model
logreg.fit(X_train,y_train)


# In[17]:


y_pred = logreg.predict(X_test)


# In[18]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# ### Random forest classifier

# In[19]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

X1=df[['Case_No', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
       'Age_Mons', 'Qchat-10-Score', 'Sex', 'Ethnicity', 'Jaundice',
       'Family_mem_with_ASD', 'Who completed the test']]  # Features
y1=df['Class/ASD Traits ']  # Labels

# Split dataset into training set and test set
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3) # 70% training and 30% test


# In[20]:


# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, criterion='entropy')

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X1_train,y1_train)

y1_pred=clf.predict(X1_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y1_test, y1_pred))

#predict
clf.predict([[522, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 19, 5, 1, 1, 0, 0, 1]])


# ### Feature extraction

# In[21]:


# Know the features' relative score with each other
feature_imp = pd.Series(clf.feature_importances_,index=feature_cols).sort_values(ascending=False)
feature_imp


# In[22]:


# Visualize using matplotlib and seaborn
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# Generating the model based on selected features

# In[23]:


# Import train_test_split function
from sklearn.cross_validation import train_test_split
# Split dataset into features and labels
X2=df[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
       'Age_Mons', 'Qchat-10-Score', 'Sex', 'Family_mem_with_ASD', 'Who completed the test']]
      # Removed feature "'Jaundice', 'Case_no', 'Ethnicity'"
y2=df['Class/ASD Traits '] # Target
# Split dataset into training set and test set
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.70, random_state=5) # 70% training and 30% test


# Compare with previous version of model

# In[24]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, criterion='gini')

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X2_train,y2_train)

# prediction on test set
y2_pred=clf.predict(X2_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y2_test, y2_pred))


# In[25]:


# import the metrics class
# from sklearn import metrics # done already
cnf_matrix2 = metrics.confusion_matrix(y2_test, y2_pred)
cnf_matrix2


# ### Result

# In[26]:


# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[27]:


# Here you will visualize confusion matrix using heat map
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Logistic Regression', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[28]:


# Here you will visualize confusion matrix using heat map
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# ### Conclusion
# 
# This simluation shows us the results of both the ML algorithms: 'Logistic Regression' and 'Random Forest'. 
# 
# Linear models are composed of one or multiple independent variables that describes a relationship to a dependent response variable. 
# Mapping qualitative or quantitative input features to a target variable that is attempted to being predicted such as financial, biological, or sociological data is known as supervised learning in machine learning terminology if the labels are known. 
# One of the most common utilized linear statistical models for discriminant analysis is logistic regression. 
# This algorithm has poorer boundary mapping capability than the two. 
# Logistic Regression gives up accuracy for model interpretability.
# 
# Random forest is an ensemble-based learning algorithm which is comprised of n collections of de-correlated decision trees. 
# It is built off the idea of bootstrap aggregation, which is a method for resampling with replacement in order to reduce variance. 
# Random Forest uses multiple trees to average (regression) or compute majority votes (classification) in the terminal leaf nodes when making a prediction. 
# Built off the idea of decision trees, random forest models have resulted in significant improvements in prediction accuracy as compared to a single tree by growing 'n' number of trees; each tree in the training set is sampled randomly without replacement. 
# Decision trees consist simply of a tree-like structure where the top node is considered the root of the tree that is recursively split at a series of decision nodes from the root until the terminal node or decision node is reached. 
# This algorithm has a very good boundary shaping capability.
# Random Forest sacrifices interpretability for accuracy of model.
# 
# So, applications where interpretation of the model is of importance, use Logistic Regerssion. While Random Forest is used for better model fitting on the data.
