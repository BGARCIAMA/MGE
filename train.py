# train.py
'''Este script entrena el modelo con los datos de prep
    Las funciones dentro de este script son:
    - entrena_modelo
'''
#Importa las librerias
from src.script import entrena_modelo

PATH_PREP ="data/prep/data_prep.csv"
PATH_MODELS = "./models"

# Entrena el modelo
entrena_modelo(PATH_PREP, PATH_MODELS)
