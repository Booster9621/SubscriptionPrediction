#!/usr/bin/env python
# coding: utf-8

# ## Final Project : Classification
# 
# The project is about predicting weather the costumer wil take the subscription of term deposit or not based on the past trends. This is the final froject I submitted after completing course on Data Science in Internshala.
# 
# The dataset was provided by the tuter.
# 
# Let's start by importing the libraries that we will be using.

# In[1]:


# importing libraries
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt 
import seaborn as sn                   # For plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")


# In[2]:


# loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Let's check the features present in our data and then we will look at their data types.

# In[3]:


train.columns


# In[4]:


test.columns


# It can be infered that **subscribed** is the target variable as it is not present in the test dataset. Let's look at the shape of the dataset.

# In[5]:


train.shape, test.shape


# We have 17 independent variables and 1 target variable, i.e. subscribed in the train dataset. We have similar features in the test dataset as the train dataset except the subscribed. We will predict the subscribed with the help of model built using the train data.
# 
# Next, let's look at how many categorical and numerical variables are there in our dataset. We will look at their data types.

# In[6]:


# Print data types for each variable
train.dtypes


# We can see there are two format of data types:
# 
# 1. **object**: Object format means variables are categorical. Categorical variables in our dataset are: job, marital, education, default, housing, loan, contact, month, poutcome, subscribed
# 
# 2. **int64**: It represents the integer variables. Integer variables in our dataset are: ID, age, balance, day, duration, campaign, pdays, previous

# In[7]:


#printing first five rows of the dataset
train.head()


# ## Univariate Analysis

# Now Let's look at the distribution of our target variable, i.e. subscribed. As it is a categorical variable, let us look at its frequency table, percentage distribution and bar plot.

# In[8]:


train['subscribed'].value_counts()


# In[9]:


# Normalize can be set to True to print proportions instead of number 
train['subscribed'].value_counts(normalize=True)


# In[10]:


# plotting the bar plot of frequencies
train['subscribed'].value_counts().plot.bar()   #plot.bar() defined in pandas


# So, 3715 users out of total 31647 have subscribed which is around 12%. Let's now explore the variables to have a better understanding of the dataset. We will first explore the variables individually using univariate analysis, then we will look at the relation between various independent variables and the target variable. We will also look at the correlation plot to see which variables affects the target variable most.
# 
# Let's first look  at the distribution of age variable to see how many people belongs to a particular age group.

# In[11]:


sn.distplot(train["age"])


# We can infer that most of the clients fall in the age group between 20-60. Now let's look at what are the different types of jobs of the clients. As job is a categorical variable, we will look at its frequency table

# In[12]:


train['job'].value_counts().plot.bar()


# We see that most of the clients belongs to blue-collar job and the students are least in number as students generally do not take a term deposit. Let's also look at how many clients have default history.

# In[13]:


train['default'].value_counts().plot.bar()  


# More than 90% of the clients have no default history. Now we will explore these variables against the target variable using bivariate analysis. We will make use of scatter plots for continuous or numeric variables and crosstabs for the categorical variables. Let's start with job and subscribed variable.

# ## Bivariate Analysis

# In[14]:


print(pd.crosstab(train['job'],train['subscribed']))

job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')


# From the above graph we can infer that students and retired people have higher chances of subscribing to a term deposit, which is surprising as students generally do not subscribe to a term deposit. The possible reason is that the number of students in the dataset is less and comparatively to other job types, more students have subscribed to a term deposit.
# 
# Next, let's explore the default variable against the subscribed variable.

# In[15]:


print(pd.crosstab(train['default'],train['subscribed']))

default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')
plt.show()


# We can infer that clients having no previous default have slightly higher chances of subscribing to a term loan as compared to the clients who have previous default history.
# 
# Let's now look at how correlated our numerical variables are. We will see the correlation between each of these variables and the variable which have high negative or positive values are correlated. By this we can get an overview of the variables which might affect our target variable. We will convert our target variable into numeric values first.

# In[16]:


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[17]:


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# We can infer that duration of the call is highly correlated with the target variable. This can be verified as well. As the duration of the call is more, there are higher chances that the client is showing interest in the term deposit and hence there are higher chances that the client will subscribe to term deposit.
# 
# Next we will look for any missing values in the dataset.

# In[18]:


train.isnull().sum()


# There are no missing values in the train dataset. 
# 
# Next, we will start to build our predictive model to predict whether a client will subscribe to a term deposit or not.
# 
# As the sklearn models takes only numerical input, we will convert the categorical variables into numerical values using dummies. We will remove the ID variables as they are unique values and then apply dummies. We will also remove the target variable and keep it in a separate variable.

# ## Model Building

# In[19]:


target = train['subscribed']
train = train.drop('subscribed',1) 
#removing subscribed column from train dataset and save it to target


# In[20]:


# applying dummies on the train dataset
train = pd.get_dummies(train) #converting categorical variable into dumies of 0 and 1 etc.


# Now, its time to build our model. We will split the train data into training and validation set so that we will be able to validate the results of our model on the validation set. We will keep 20% data as validation set and rest as the training set. 

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


# splitting into train and validation with 20% data in validation set and 80% data in train set.
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)
#X_val and y_val are new variables where 20% of data will get saved


# Now our data is ready. Its time to build our model and check its performance. Logistic regression is used for classification problems and as it is a classification problem let's first build a Logistic Regression model.

# ## Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression


# In[24]:


# defining the logistic regression model
lreg = LogisticRegression()


# In[25]:


# fitting the model on  X_train and y_train
lreg.fit(X_train,y_train)


# In[26]:


# making prediction on the validation set
prediction = lreg.predict(X_val)


# Now we will evaluate how accurate our predictions are. As the evaluation metric for this problem is accuracy, let's calculate the accuracy on validation set.

# In[27]:


from sklearn.metrics import accuracy_score


# In[29]:


# calculating the accuracy score
accuracy_score(y_val, prediction) #number of values matching/total number of data points


# In[30]:


test = pd.get_dummies(test)
test_prediction = lreg.predict(test)


# We got an accuracy score of around 90% on the validation dataset. 
# 

# In[31]:


submission = pd.DataFrame()
submission['ID'] = test['ID']
submission['subscribed'] = test_prediction


# In[32]:


submission['ID'] = test['ID']
submission['subscribed'] = test_prediction


# Since the target variable is yes or no, we will convert 1 and 0 in the predictions to yes and no respectively.

# In[33]:


submission['subscribed'].replace(0,'no',inplace=True)
submission['subscribed'].replace(1,'yes',inplace=True)


# In[34]:


submission.to_csv('submission.csv', header=True, index=False)


# Now you have the submission file with you. Follow these steps to generate your score:
# 1. Open the submission.csv file.
# 2. Copy the values in the subscribed column and paste them in the subscribed column of solution_checker.xlsx file.
# 3. You will see the accuracy of the model on test dataset under Your accuracy score column.

# In[ ]:




