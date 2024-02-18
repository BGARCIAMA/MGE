# train.py
'''Este script entrena el modelo con los datos de prep
    Las funciones dentro de este script son:
    - entrena_modelo
'''
#Importa las librerias
import pandas as pd
import joblib
from src.script import entrena_modelo

# Read the data
data_final = pd.read_csv("data/prep/data_final.csv")

# Entrena el modelo
modelo = entrena_modelo(data_final)
