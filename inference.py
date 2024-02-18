# inference.py
'''Este script realiza la predicción
    Las funciones dentro de este script son:
    - prediccion_precio
'''
#Importa las librerias
import pandas as pd
import joblib
from src.script import prediccion_precio

# Realiza las predicciones
y_pred = prediccion_precio()