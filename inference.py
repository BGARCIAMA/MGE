# inference.py
'''Este script realiza la predicción
'''
#Importa librerías
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer

def prediccion_precio():             
    # Definir las variables necesarias para la predicción
    variables = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 
                'LotFrontage','TotalBsmtSF', 'GrLivArea', 'GarageArea']
    locale.setlocale(locale.LC_ALL, '')

    # Solicitar al usuario ingresar las variables de entrada
    print("Ingrese los valores de las variables para predecir el precio de la casa:")
    ov = input("OverallQual - Calidad general de materiales y acabados (valor entre 1 y 10): ")
    yearB = input("YearBuilt - Año en que se construyó: ")
    yearRemo = input("YearRemoAdd - Año en que se remodeló (si no ha tenido entonces es igual al de construcción): ")
    LT = float(input("LotFrontage - Tamaño en pies cuadrados de la entrada a la calle: "))
    TotalB = float(input("TotalBsmtSF - Tamaño en pies cuadrados del sótano: "))
    GrL = float(input("GrLivArea - Tamaño en pies cuadrados de la superficie habitable: "))
    GarArea = float(input("GarageArea - Tamaño en pies cuadrados de la cochera: "))
                
    user_input = pd.DataFrame({
        'OverallQual': [ov],
        'YearBuilt': [yearB],
        'YearRemodAdd': [yearRemo],
        'LotFrontage': [LT],
        'TotalBsmtSF': [TotalB],
        'GrLivArea': [GrL],
        'GarageArea': [GarArea]
    })

    # Crear un df de la info a calcular
    input_data = pd.DataFrame(user_input, columns = variables)

    # Cargar el modelo previamente entrenado
    loaded_rfr = joblib.load("./rfr_model.joblib")

    # Resultado del modelo
    prediction = loaded_rfr.predict(input_data)
    prediction_formateada = locale.currency(prediction[0], grouping=True)

    # La predicción es:
    print(f'El precio estimado de la casa es: {prediction_formateada}')