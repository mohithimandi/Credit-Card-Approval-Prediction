import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
app = pd.read_csv('application_record.csv')
credit = pd.read_csv('credit_record.csv')
print("number of people working status:")
print(app["OCCUPATION_TYPE"].value_counts())
sns.set(rc={'figure.figsize':(18,6)})
sns.countplot(x='OCCUPATION_TYPE', data=app, palette='Set2')
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(app.select_dtypes(include=np.number).corr(), annot=True)




app.drop_duplicates(subset=[

    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',

    'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',

    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',

    'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',

    'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'

], keep='first', inplace=True)


app.isnull().mean()


def data_cleaning(app, credit):


    app['CNT_FAM_MEMBERS'] = app['CNT_CHILDREN'] + 1


    app.drop([

        'FLAG_MOBIL',

        'FLAG_WORK_PHONE',

        'FLAG_PHONE',

        'FLAG_EMAIL'

    ], axis=1, inplace=True)




    app['DAYS_BIRTH'] = abs(app['DAYS_BIRTH'])

    app['DAYS_EMPLOYED'] = abs(app['DAYS_EMPLOYED'])




    app['NAME_HOUSING_TYPE'] = app['NAME_HOUSING_TYPE'].map({

        'House / apartment': 1,

        'Rented apartment': 2,

        'With parents': 3,

        'Municipal apartment': 4,

        'Office apartment': 5,

        'Co-op apartment': 6

    })


    app['NAME_INCOME_TYPE'] = app['NAME_INCOME_TYPE'].map({

        'Working': 1,

        'Commercial associate': 2,

        'Pensioner': 3,

        'State servant': 4,

        'Student': 5

    })


    app['NAME_EDUCATION_TYPE'] = app['NAME_EDUCATION_TYPE'].map({

        'Secondary / secondary special': 1,

        'Higher education': 2,

        'Incomplete higher': 3,

        'Lower secondary': 4,

        'Academic degree': 5

    })


    app['NAME_FAMILY_STATUS'] = app['NAME_FAMILY_STATUS'].map({

        'Married': 1,

        'Single / not married': 2,

        'Civil marriage': 3,

        'Separated': 4,

        'Widow': 5

    })




    credit_group = credit.groupby('ID').agg({

        'MONTHS_BALANCE': ['min','max'],

        'STATUS': 'count'

    })


    credit_group.columns = ['open_month','end_months','window']


    return app, credit_group




app_cleaned, credit_grouped = data_cleaning(app.copy(), credit.copy())

print(app_cleaned.head())


def to_binary_status(status_code):

    if status_code in ['0', 'X', 'C']:

        return 0

    else:

        return 1


credit_temp = credit.copy()

credit_temp['BAD_STATUS'] = credit_temp['STATUS'].apply(to_binary_status)


target_df = credit_temp.groupby('ID')['BAD_STATUS'].max().reset_index()

target_df.rename(columns={'BAD_STATUS': 'TARGET'}, inplace=True)


print("Target distribution:")

print(target_df['TARGET'].value_counts())


final_df = app_cleaned.merge(

    target_df,

    how='inner',

    on='ID'

)


final_df = final_df.merge(

    credit_grouped,

    how='left',

    on='ID'

)


print(f"Merged shape: {final_df.shape}")


print(final_df.head())


print("\nMissing values in each column:")

print(final_df.isnull().sum().sort_values(ascending=False))


from sklearn.preprocessing import LabelEncoder


cg = LabelEncoder()

oc = LabelEncoder()

own_r = LabelEncoder()






final_df['CODE_GENDER'] = cg.fit_transform(final_df['CODE_GENDER'])

final_df['FLAG_OWN_CAR'] = oc.fit_transform(final_df['FLAG_OWN_CAR'])

final_df['FLAG_OWN_REALTY'] = own_r.fit_transform(final_df['FLAG_OWN_REALTY'])




print(final_df.head())


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

import seaborn as sns


def logistic_reg(X_train, X_test, y_train, y_test):


    lr_model = LogisticRegression(random_state=42, max_iter=1000)


    lr_model.fit(X_train, y_train)


    y_pred = lr_model.predict(X_test)


    cm = confusion_matrix(y_test, y_pred)


    print("Confusion Matrix:")

    print(cm)


    print(classification_report(y_test, y_pred))


    plt.figure(figsize=(10,8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    plt.xlabel('Predicted Label')

    plt.ylabel('True Label')

    plt.title('Confusion Matrix - Logistic Regression')

    plt.show()


    return lr_model, y_pred












def random_forest(X_train, X_test, y_train, y_test):

    """

    Builds, trains, and tests a Random Forest classification model,

    returning performance metrics.

    """




    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


    print("\nTraining Random Forest model...")

    rf_model.fit(X_train, y_train)


    print("Generating predictions...")

    y_pred = rf_model.predict(X_test)


    print("\n" + "="*40)

    print("Random Forest Model Evaluation")

    print("="*40)


    print("\nClassification Report")

    print(classification_report(y_test, y_pred))




def d_tree(xtrain,xtest,ytrain,ytest):

    dt=DecisionTreeClassifier()

    dt.fit(xtrain,ytrain)

    ypred=dt.predict(xtest)

    print('***DecisionTreeClassifier***')

    print('Confusion matrix')

    print(confusion_matrix(ytest,ypred))

    print('Classification report')

    print(classification_report(ytest,ypred))
    
    # ==========================================================
# TRAIN MODEL AND GENERATE model.pkl
# ==========================================================

import joblib
from sklearn.preprocessing import LabelEncoder

# Prepare features and target
X = final_df.drop(['ID', 'TARGET'], axis=1)
y = final_df['TARGET']

# Encode remaining categorical columns
categorical_cols = X.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Fill missing values
X.fillna(0, inplace=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest Model
print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(rf_model, 'model.pkl')

print("\n✅ model.pkl generated successfully!")

