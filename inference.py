# inference.py
'''Este script realiza la predicción
    Las funciones dentro de este script son:
    - prediccion_precio
'''
#Importa las librerias
import pandas as pd
import joblib
from src import script

# Realiza las predicciones
y_pred = script.prediccion_precio()