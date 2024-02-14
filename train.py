# train.py
'''Este script entrena el modelo con los datos de prep
'''
#Importa librerías
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer

def entrena_modelo(data_final):
    # Entrenamos el modelo 
    # Seleccionar características numéricas y eliminar 'SalePrice'
    numerical_cols = data_final.select_dtypes(include=['int64', 'float64']).drop('SalePrice', axis=1).columns
    numerical_transformer = StandardScaler()

    # Crear ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ])

    # Modelos
    model_list = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regression': RandomForestRegressor()
    }

    # División de la base de datos
    x = data_final.drop('SalePrice', axis=1)
    y = data_final['SalePrice']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # Crear pipelines
    pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('model', model)]) for name, model in model_list.items()}

    # Entrenamiento y evaluación de los modelos
    rmse_results = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_results[name] = rmse

    # Mostrar los RMSE resultantes
    rmse_results_sorted = dict(sorted(rmse_results.items(), key=lambda item: item[1]))

    # Se dividen las bases
    df_train, df_test = train_test_split(datafinal, test_size=0.2, random_state=42)

    x_train = df_train.drop(['SalePrice'], axis = 1)
    y_train = df_train['SalePrice']  

    # Se entrena con el XGBoost Regressor
    RFR_model = RandomForestRegressor()
    RFR_model.fit(x_train, y_train)

    # Se hace la predicción en la base test usando el modelo entrenado
    y_pred = RFR_model.predict(df_test.drop(['SalePrice'], axis = 1))

    joblib.dump(RFR_model, "./rfr_model.joblib")