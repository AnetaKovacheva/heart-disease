import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


st.title("Explore Personal Key Indicators of Heart Disease")
st.write("""This application explores and models the *Personal Key Indicators of Heart Disease* dataset,
 available in [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).
 Data come from the Centers for Disease Control and Prevention (CDC) and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), 
 which conducts annual telephone surveys to gather data on the health status of U.S. residents.
 The most recent dataset (as of February 15, 2022) includes data from 2020. 
 It consists of 401,958 rows and 279 columns; the latter were reduced to just about 20 variables.

 The task of this application is to display EDA outputs and to provide opportunity to model
 the data with different classification algorithms.

 """)

# Read data
st.subheader("1. Load data")
st.write("""The dataset is too large to model all samples on the run. 
Therefore, to speed up computations, you can load less rows. Please, select from the 
left-side menu the number of rows you want to load and use for modelling.

The dataset has 18 columns. *HeartDisease* ["Yes", "No"] is used as 
an explonatory variable. It should be noted that the classes are quite unbalanced.""")

st.sidebar.header('Number of samples to load')
samples_num = st.sidebar.selectbox('Number of samples', [100, 1000, 2000, 5000, 10000, 20000,
                                                    30000, 40000, 50000, 60000, 70000, 80000, 90000,
                                                    100000, 150000, 200000, 250000, 300000])

hd_data = pd.read_csv("https://raw.githubusercontent.com/AnetaKovacheva/streamlit/master/heart_2020_cleaned.csv", nrows = samples_num)
st.dataframe(hd_data.head(10))
st.write("""
Meaning of features and their values:
- HeartDisease: respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) 
- BMI: Body Mass Index
- Smoking: if the respondent smoked at least 100 cigarettes in his/her entire life? 
- AlcoholDrinking: heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
- Stroke: if the respondent ever had a stroke
- PhysicalHealth: number of days the respondent had a physical illness and injury (during the past 30 days)
- MentalHealth: number of days the respondent was not in good mental condition (during the past 30 days)
- DiffWalking: if the respondent had a serious difficulty walking or climbing stairs 
- Sex:  male or female 
- AgeCategory: fourteen-level age category
- Race: imputed race/ethnicity value
- Diabetic: if the respondent ever had diabetes
- PhysicalActivity: doing physical activity or exercise during the past 30 days other than regular job
- GenHealth: describe state of general health
- SleepTime: average hours of sleep in a 24-hour period
- Asthma: if the respondent ever had asthma
- KidneyDisease: if the respondent ever had kidney disease (excluding kidney stones, bladder infection or incontinence)
- SkinCancer: if the respondent ever had skin cancer
""")

st.subheader("2. Exploratory Data Analysis")
st.write(f"The sub-sampled dataset has {hd_data.shape[0]} rows and {hd_data.shape[1]} columns. Nine are booleans, 5 - strings and 4 - decimals.")

# BMI
st.write("**BMI** is the only feature with continuous values. Their distribution seems almost normal.")
fig = plt.figure(figsize=(4,2)) 
ax = plt.axes()
sns.histplot(x = hd_data["BMI"], data = hd_data)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

# Numeric columns
st.write("""The plots below show the distributions in **Physical Health**,
**Mental Health**, and **Speep time**. Most people didn't report bad days due to physical injury or
emotional unfit, and managed to sleep 7 or 8 hours.""")
numeric = hd_data[["PhysicalHealth", "MentalHealth", "SleepTime"]]
fig, ax = plt.subplots(1, 3, figsize=(25, 10))
for variable, subplot in zip(numeric, ax.flatten()):
    sns.countplot(x = hd_data[variable], data = hd_data, ax = subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

# Categorical cols
st.write("""Most features hold categorical data. Their values are displayed on the 
plots below. """)
categorical=[]
for column in hd_data:
    if is_string_dtype(hd_data[column]):
        categorical.append(column)


fig, ax = plt.subplots(7, 2, figsize=(15, 30))
# fig.delaxes(ax[4][2])
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(x = hd_data[variable], data = hd_data, ax=subplot)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

# Cross plot
st.write("""Next, the cross-bar chart can be used to explore
the **relationships between two categorical features**. Please, select the feature
 you want to see on *x* axis (Lead feature), then the one to cross-analyse it with (Hue feature).  
""")
st.sidebar.header("Cross-bar chart elements")
col_1_name = st.sidebar.selectbox("Lead feature", ("HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", 
                            "DiffWalking", "Sex", "AgeCategory", 
                            "Race", "Diabetic", "PhysicalActivity", "GenHealth", 
                            "Asthma", "KidneyDisease", "SkinCancer"))

col_2_name = st.sidebar.selectbox("Hue feature", ("Smoking", "AlcoholDrinking", "Stroke", 
                            "DiffWalking", "Sex", "AgeCategory", 
                            "Race", "Diabetic", "PhysicalActivity", "GenHealth", 
                            "Asthma", "KidneyDisease", "SkinCancer", "HeartDisease"))

st.write(f"Relationship between {col_1_name} and {col_2_name}")
fig = plt.figure(figsize=(4,2)) 
ax = plt.axes()
sns.countplot(data = hd_data, x = col_1_name, hue = col_2_name)
# plt.legend(fontsize=12, title_fontsize=12)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

# Encode categorical variables
hd_data_categorical = hd_data[categorical]
mappings = {
    "HeartDisease": {"No": 0, "Yes": 1},
    "Smoking": {"No": 0, "Yes": 1},
    "AlcoholDrinking": {"No": 0, "Yes": 1},
    "Stroke": {"No": 0, "Yes": 1},
    "DiffWalking": {"No": 0, "Yes": 1},
    "Sex": {"Female": 0, "Male": 1},
    "AgeCategory": {"18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4, "45-49": 5, '50-54': 6,
                   "55-59": 7, "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11, "80 or older": 12},
    "Race": {"White": 0, "Black": 1, "Asian": 2, "American Indian/Alaskan Native": 3, "Hispanic": 4, "Other": 5},
    "Diabetic": {"No": 0, "Yes": 1, "Yes (during pregnancy)": 2, "No, borderline diabetes": 3},
    "PhysicalActivity": {"No": 0, "Yes": 1},
    "GenHealth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
    "Asthma": {"No": 0, "Yes": 1},
    "KidneyDisease": {"No": 0, "Yes": 1},
    "SkinCancer": {"No": 0, "Yes": 1},
}
hd_data_encoded = hd_data_categorical.apply(lambda col: col.map(mappings[col.name]))

numerical_col=[]
for column in hd_data:
    if is_numeric_dtype(hd_data[column]):
        numerical_col.append(column)

hd_data_numerical = hd_data[numerical_col]

hd_data_all = pd.concat([hd_data_numerical, hd_data_encoded], axis = 1)

st.subheader("3. Data Preprocessing")
st.write("""All string values were replaced with numeric ones. Categorical 
features were encoded manually (i.e. instead by applying `pd.get_dummies()`). 

The **Correlation matrix** below shows that the coefficients vary between 
-0.5 and 0.5, which means that there is not a strong correlation between
features. """)
fig = plt.figure(figsize = (15, 15))
sns.heatmap(hd_data_all.corr(),
            vmin=-1.0, vmax=1.0,
            cmap="coolwarm", annot=True, fmt='.2f')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

st.write("""
The first count plot above confirmed that the respondents who didn't report 
coronary heart disease or myocardial infarction outnumber the ones who did, which makes
the dataset quite imbalanced. This problem was addressed by **undersampling** 
the persons who didn't experience heart issues (oversampling the other class increases
the total number of samples; hence computations and modelling are slowed down). Now,
the dataset holds equal number of entries for both classes, and their total number
slighly exceeds 4000.
""")
# balance the dataset
class_1, class_2 = hd_data_all["HeartDisease"].value_counts()
df_no = hd_data_all[hd_data_all["HeartDisease"] == 0]
df_yes = hd_data_all[hd_data_all["HeartDisease"] == 1]
df_no_under = df_no.sample(class_2, replace=True)
hd_data_all_balanced = pd.concat([df_yes, df_no_under], axis=0)

st.write("- Number of samples without ('No') heart disease after undersampling:", len(hd_data_all_balanced[hd_data_all_balanced["HeartDisease"] == 0]))
st.write("- Number of samples, with ('Yes') heart disease:", len(hd_data_all_balanced[hd_data_all_balanced["HeartDisease"] == 1]))
st.write("- Total number of samples in the dataset:", len(hd_data_all_balanced))

st.write("""
Furthermore, all values were converted into **float32** dtype (to optimize memory usage and speed up 
computations), and labels (column "HeartDisease") were **separated** from features. Thereafter, 
the dataset was **split** into *train* and *test* samples.
""")
# make all float 32
hd_data_all_balanced = hd_data_all_balanced.astype("float32")

# separate features and labels
features = hd_data_all_balanced.drop("HeartDisease", axis = 1)
label = hd_data_all_balanced["HeartDisease"]

# split train and test data
features_train, features_test, label_train, label_test = train_test_split(features, label,
                                        test_size=0.2, stratify=label, shuffle=True, random_state=42)

# Model data
st.subheader("4. Model the data")
st.write("""
This section provides opportunity to model the data with different Machine Learning algorithms. Model
performance could be adjusted by tuning some of its hyperparameters.

Please, select the modelling algorithm you would like to play with from the left-side menu.

For [**Logistic Regression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
please, select the regularization type (defined as *penality*) you want to apply. 

For [**Decision Tree**](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html),
please, select its maximum number of nodes (*max_depth*) by sliding the dot on the slide bar in the left-side menu.

For [**Random Forest**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
please, select the number of trees (*n_estimators*) and the maximum number of nodes (*max_depth*) per tree 
by sliding the dot on the slide bar in the left-side menu.

Model performance is evaluated on a test sample.
""")
st.sidebar.header("Modelling elements")
classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "Decision Tree", "Random Forest"))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Logistic Regression":
        penalty = st.sidebar.selectbox("penalty", ("none", "l2", "l1", "elasticnet"))
        params["penalty"] = penalty
    elif clf_name == "Decision Tree":
        max_depth = st.sidebar.slider("max_depth", 3, 10)
        params["max_depth"] = max_depth
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(penalty=params["penalty"], solver = "saga")
    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params["max_depth"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                                random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
clf.fit(features_train, label_train)
y_pred = clf.predict(features_test)

acc = accuracy_score(label_test, y_pred)
f1 = f1_score(label_test, y_pred)

st.write("""
Model evaluation metrics are listed below:
""")

st.write(f"- The data are classified with {classifier_name}.")
st.write(f"- Model's **accuracy** on the test data is: {acc:.2f}%.")
st.write(f"- Model's **f1 score** on the test data is: {f1:.2f}%.")
st.write(f"- **Confusion matrix** on the test data with properly and wrongly predicted labels.")

fig = plt.figure(figsize = (4, 3))
sns.heatmap(confusion_matrix(label_test, y_pred),
    annot = True, fmt = ".0f", cmap = "coolwarm", 
    linewidths = 2, linecolor = "red")
plt.title("Actual values")
plt.ylabel("Predicted values")
plt.tight_layout()

st.pyplot(fig)

st.write("""End of application""")