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
## prep.py
'''Este script prepara los datos para incorporarlos al modelo
    Las funciones dentro de este script son:
    - descargar_datos
    - impute_continuous_missing_data (dentro de preprocesar_datos)
    - preprocesar_datos
'''
import os
import argparse
import logging
from datetime import datetime
from src.script import descargar_datos, preprocesar_datos


def setup_logging():
    '''Configura el logging para el script de preprocesamiento
    '''
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    log_prep_file_name = f"logs/{date_time}_prep.log"
    logging.basicConfig(
        filename=log_prep_file_name,
        level=logging.DEBUG,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')


# Descarga los datos
def parse_arguments():
    '''Parsea los argumentos de la línea de comandos'''
    parser = argparse.ArgumentParser(description='Preprocesamiento de datos')
    parser.add_argument('--base_path_data',
                        type=str, default="./data/raw/",
                        help='Ruta base de los datos crudos')
    parser.add_argument('--path_out',
                        type=str, default="./data/raw/",
                        help='Ruta para guardar los datos descargados')
    parser.add_argument('--path_data',
                        type=str, default="./data/raw/data_total.csv",
                        help='Ruta del archivo CSV de entrada')
    parser.add_argument('--base_path_out_prep',
                        type=str, default="data/prep/data_prep.csv",
                        help='Ruta para guardar los datos preprocesados')
    parsed_args = parser.parse_args()
    return parsed_args


args = parse_arguments()
if not os.path.exists("logs/"):
    os.makedirs("logs/")

setup_logging()

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    try:
        logging.info("Inicio del script de preprocesamiento.")

        args = parse_arguments()

        data_total = descargar_datos(args.base_path_data, args.path_out)

        logging.debug("Número de filas en el dataset de entrada: %d", len(data_total))
        logging.debug("Path de entrada: %s", args.path_data)
        logging.info("Datos descargados.")

        # Preprocesa los datos
        data_final = preprocesar_datos(args.path_data, args.base_path_out_prep)

        # Log de variables
        logging.debug("Número de columnas en el dataset de salida: "
                      "%d", len(data_final.columns))
        logging.debug("Path de salida: %s", args.base_path_out_prep)

        logging.info("Fin del script de preprocesamiento.")

    except Exception as e:
        logging.error("Error en el script de preprocesamiento.")
        logging.error(e)
        raise
