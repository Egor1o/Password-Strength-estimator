#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # import numpy package under shorthand "np" import pandas as pd # import pandas package under shorthand␣
import pandas as pd
import matplotlib.pyplot as plt
from nose.tools import assert_equal
from numpy.testing import assert_array_equal
#get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[2]:


data = pd.read_csv("dataset.csv")


# In[3]:


selected_features = data[["strength", "length", "entropy"]]
#correlation matrix for the selected features
correlation_matrix = selected_features.corr()
print(correlation_matrix)


# In[4]:


X_features = data.drop(['password','crack_time_sec','crack_time','class_strength','length'],axis=1)
print(X_features)
X_features = np.array(X_features)
y_label = np.array(data.drop(['password','crack_time','class_strength','entropy','strength','length'],axis=1))
y_label = np.log10(y_label)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_label,test_size=0.33, shuffle= True)
#standardization
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#polynomial features' defining
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_scaled)
X_poly_test = poly.transform(X_test_scaled)


# In[5]:


#stating data correlation
sns.set(style="ticks")
sns.pairplot(data, vars=["strength", "length", "entropy"], markers='o')


# In[6]:


fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# First subplot
axs[0].scatter(X_train_scaled[:, 0], y_train)
axs[0].set_xlabel('Strength')
axs[0].set_ylabel('Labels')
axs[0].set_title('Scatter Plot of Strength vs. Labels (Individual Points) Training set.')
axs[0].grid(True)

# Second subplot
axs[1].scatter(X_train_scaled[:, 1], y_train)
axs[1].set_xlabel('Entropy')
axs[1].set_ylabel('Labels')
axs[1].set_title('Scatter Plot of Entropy vs. Labels (Individual Points) Training set.')
axs[1].grid(True)

plt.tight_layout()

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# First subplot
axs[0].scatter(X_test_scaled[:, 0], y_test)
axs[0].set_xlabel('Strength')
axs[0].set_ylabel('Labels')
axs[0].set_title('Scatter Plot of Strength vs. Labels (Individual Points) Test set.')
axs[0].grid(True)

# Second subplot
axs[1].scatter(X_test_scaled[:, 1], y_test)
axs[1].set_xlabel('Entropy')
axs[1].set_ylabel('Labels')
axs[1].set_title('Scatter Plot of Entropy vs. Labels (Individual Points) Test set.')
axs[1].grid(True)

plt.tight_layout()

plt.show()


# In[7]:


#fiting polynomail model
lin_regr = LinearRegression(fit_intercept=False) 
lin_regr.fit(X_poly_train,y_train)
y_pred = lin_regr.predict(X_poly_test)
#error calculation
tr_error = mean_squared_error(y_test,y_pred)
#scaling via pca for visualiztion of multidimensional features.
n_components = 1 
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_poly_train)
X_test_pca = pca.transform(X_poly_test)
#checking training set's accuracy
y_pred_train = lin_regr.predict(X_poly_train)
tr_error_train = mean_squared_error(y_train,y_pred_train)
#erorrs to be obtained in the very end
tet = tr_error_train 
te = tr_error
print("validation set's error:  " + str(tr_error))
print("training set's error:  " + str(tr_error_train))


# In[8]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test_pca[:, 0], y_test, color='blue', label='Testing Features - Actual')
plt.scatter(X_test_pca[:, 0], y_pred, color='red', label='Testing Features - Predicted', alpha=0.5)
plt.xlabel('PCA Component 1') 
plt.ylabel('Log(crack_time_sec)')
plt.title('PCA vs. Log(crack_time_sec) - Testing Data') 
plt.legend()
plt.tight_layout()
plt.show()


# In[9]:


plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('Actual Log(crack_time_sec)')
plt.ylabel('Predicted Log(crack_time_sec)')
plt.title('Polynomial Regression: Actual vs. Predicted Values')
plt.show()


# In[10]:


#Method two - MLP Regressor
y_train = y_train.ravel()
y_test = y_test.ravel()


# In[11]:


## define a list of values for the number of hidden layers
num_layers = [1,2,3,4,5,6,8,10] # number of hidden layers
num_neurons = 3 # number of neurons in each layer


# In[12]:


# we will use this variable to store the resulting training errors corresponding to different hidden-layer numbers
mlp_tr_errors = []
mlp_test_errors = []
y_pred_3 = y_pred
for i, num in enumerate(num_layers):
    hidden_layer_sizes = tuple([num_neurons]*num) # size (num of neurons) ofeach layer stacked in a tuple
    mlp_regr = MLPRegressor(hidden_layer_sizes= hidden_layer_sizes,max_iter =1000, random_state =42)
    mlp_regr.fit(X_train_scaled, y_train)
    y_pred_train = mlp_regr.predict(X_train_scaled) # predict on the training set
    tr_error = mean_squared_error(y_train, y_pred_train) # calculate thetraining error
    y_pred_test = mlp_regr.predict(X_test_scaled) # predict values for the validationdata
    if num == 3:
        y_pred_3 = y_pred_test
    test_error = mean_squared_error(y_test, y_pred_test) # calculate the validation error
    assert mlp_regr.n_layers_ == num_layers[i]+2 # total layers = num of hidden layers + input layer + output layer
    mlp_tr_errors.append(tr_error)
    mlp_test_errors.append(test_error)
print(mlp_tr_errors)


# In[13]:


plt.figure(figsize=(8, 6))
plt.plot(num_layers, mlp_tr_errors, label = 'Train')
plt.plot(num_layers, mlp_test_errors,label = 'Test')
plt.xticks(num_layers)
plt.legend(loc = 'upper left')
plt.xlabel('Layers')
plt.ylabel('Loss')
plt.title('Train vs test loss')
plt.show()


# In[14]:


plt.scatter(y_test, y_pred_3, color='blue', alpha=0.5)
plt.xlabel('Actual Log(crack_time_sec)')
plt.ylabel('Predicted Log(crack_time_sec)')
plt.title('MLP Regression: Actual vs. Predicted Values')
plt.show()


# In[15]:


# creates a table to compare the training and validation errors for MLPs with different number of hidden layers
errors = {"num_hidden_layers":num_layers,
"mlp_train_errors":mlp_tr_errors,
"mlp_val_errors":mlp_test_errors,
}
pd.DataFrame(errors)


# In[16]:


# Data for the first subplot
errors1 = [mlp_tr_errors[2], mlp_test_errors[2]]
labels1 = ['Test Error', 'Train Error']
colors1 = ['orange', 'blue']

# Data for the second subplot
errors2 = [te, tet]
labels2 = ['Test Error', 'Train Error']
colors2 = ['orange', 'blue']
print(tr_error)
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first subplot
ax1.bar(labels1, errors1, color=colors1)
ax1.set_xlabel('Error Type')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Test and Train Errors (MLP Regression)')

# Plot the second subplot
ax2.bar(labels2, errors2, color=colors2)
ax2.set_xlabel('Error Type')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Test and Train Errors (Polynomial Regression)')

plt.tight_layout()

plt.show()

