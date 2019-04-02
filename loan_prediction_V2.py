# Loan Prediction

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_loan_prediction.csv')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

#Getting familiar with data
dataset.head()
dataset.describe()

dataset['Credit_History'].value_counts()
dataset['ApplicantIncome'].hist(bins=50)
dataset.boxplot(column='ApplicantIncome')
dataset.boxplot(column='ApplicantIncome', by = 'Education')
dataset['LoanAmount'].hist(bins=50)
dataset.boxplot(column='LoanAmount')

#Handling missing data
dataset.apply(lambda x: sum(x.isnull()),axis=0)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(), inplace=True)

dataset['Self_Employed'].fillna('No',inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
dataset.dtypes

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(n_splits=5)
  error = []
  for train, test in kf.split(data):
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])
  

# Fitting the Regression Model to the dataset
# Create your regressor here
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, dataset,predictor_var,outcome_var)

#We can try different combination of variables:
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount']
classification_model(model, dataset,predictor_var,outcome_var)

#Decision Tree
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, dataset,predictor_var,outcome_var)

#We can try different combination of variables:
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount']
classification_model(model, dataset,predictor_var,outcome_var)

#Random Forest
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount','ApplicantIncome']
classification_model(model, dataset,predictor_var,outcome_var)

#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['ApplicantIncome','LoanAmount','Credit_History','Dependents','Property_Area']
classification_model(model, dataset,predictor_var,outcome_var)
