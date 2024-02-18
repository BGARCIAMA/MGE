# prep.py
'''Este script prepara los datos para incorporarlos al modelo
    Las funciones dentro de este script son:
    - descargar_datos
    - impute_continuous_missing_data (dentro de preprocesar_datos) 
    - preprocesar_datos
'''
# Importa las librerias
from src.script import descargar_datos, preprocesar_datos

BASE_PATH_DATA = "./data/raw/"
PATH_DATA = "./data/raw/data_total.csv"
PATH_OUT = "./data/raw/"
BASE_PATH_OUT_PREP="data/prep/data_prep.csv"

#Descarga los datos
data_total = descargar_datos(BASE_PATH_DATA, PATH_OUT)

#Preprocesa los datos
data_final = preprocesar_datos(PATH_DATA, BASE_PATH_OUT_PREP)
