# prep.py

# Importa las librerias
# from src.script import descargar_datos, preprocesar_datos

# BASE_PATH_DATA = "./data/raw/"
# PATH_DATA = "./data/raw/data_total.csv"
# PATH_OUT = "./data/raw/"
# BASE_PATH_OUT_PREP = "data/prep/data_prep.csv"

# Descarga los datos
# data_total = descargar_datos(BASE_PATH_DATA, PATH_OUT)

# Preprocesa los datos
# data_final = preprocesar_datos(PATH_DATA, BASE_PATH_OUT_PREP)


# -------------
# prep.py
'''Este script prepara los datos para incorporarlos al modelo
    Las funciones dentro de este script son:
    - descargar_datos
    - impute_continuous_missing_data (dentro de preprocesar_datos)
    - preprocesar_datos
'''
import argparse
from src.script import descargar_datos, preprocesar_datos


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocesamiento de datos')
    parser.add_argument('--base_path_data', 
                        type=str, default="./data/raw/",
                        help='Ruta base de los datos crudos')
    parser.add_argument('--path_out', 
                        type=str, default="./data/raw/",
                        help='Ruta de salida para guardar los datos descargados')
    parser.add_argument('--path_data', 
                        type=str, default="./data/raw/data_total.csv",
                        help='Ruta del archivo CSV de entrada')
    parser.add_argument('--base_path_out_prep', 
                        type=str, default="data/prep/data_prep.csv",
                        help='Ruta de salida para guardar los datos preprocesados')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    # Descarga los datos
    data_total = descargar_datos(args.base_path_data, args.path_out)

    # Preprocesa los datos
    data_final = preprocesar_datos(args.path_data, args.base_path_out_prep)
