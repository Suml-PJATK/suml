import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from pycaret.regression import setup, compare_models, predict_model, tune_model, save_model, load_model
import pickle



def load_housing_data(housing):
    return housing


def explore_housing_data(housing):
    print(housing.head())
    print(housing.info())
    print(housing.ocean_proximity.value_counts())
    return housing


def prepare_data(data,  test_size=0.2, random_state=42):
    # Separate the target variable
    features = data.drop("median_house_value", axis=1)
    features = data.drop("ocean_proximity", axis=1)

    labels = data["median_house_value"].copy()

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    # features_with_extra_features = add_extra_features(features, add_bedrooms_per_room)

    # Define numerical and categorical columns
    num_attribs = ["longitude", "latitude", "housing_median_age",
                   "total_rooms", "total_bedrooms", "population",
                   "households", "median_income"]



# Define the custom transformer for additional attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# The main function for data preparation
def prepare_data(strat_train_set):
    # Separate features and labels
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # Fill missing values in 'total_bedrooms'
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median, inplace=True)

    # Numerical attributes pipeline
    num_attribs = list(housing.drop("ocean_proximity", axis=1))
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    # Full pipeline for both numerical and categorical attributes
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
    ])

    # Apply the full pipeline
    housing_prepared = full_pipeline.fit_transform(housing)


    # Apply transformations to training and test sets
    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(X_test)

    return X_train_prepared, y_train, X_test_prepared, y_test, full_pipeline



def train_model(features, labels):
    model = LinearRegression()
    model.fit(features, labels)
    return model


def predict(model, features):
    predictions = model.predict(features)
    return predictions

def predictions_to_dataframe(predictions):
    df = pd.DataFrame(predictions, columns=['Predicted_Value'])
    return df


def features_to_dataframe(features):
    df = pd.DataFrame(features)
    return df


def train_model(features, labels):
    model = LinearRegression()
    model.fit(features, labels)
    return model

