# inference.py

# Importa las librerias
# from src.script import prediccion_precio

# Realiza las predicciones
# prediccion_precio()

# ----------------
# inference.py
'''Este script realiza la predicción
    Las funciones dentro de este script son:
    - prediccion_precio
'''
import os
import argparse
import logging
from datetime import datetime
import pandas as pd
from src.script import prediccion_precio


if not os.path.exists("logs/"):
    os.makedirs("logs/")
# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_infer_file_name = f"logs/{date_time}_inference.log"
logging.basicConfig(
    filename=log_infer_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def parse_arguments():
    '''Parsea los argumentos de la línea de comandos'''
    parser = argparse.ArgumentParser(description='Realiza de predicciones')
    parser.add_argument('--input_data',
                        type=str, default='./data/prep/data_prep.csv',
                        help='Ruta del CSV de entrada para predicción')
    parser.add_argument('--model_file',
                        type=str, default='./models/rfr_model.joblib',
                        help='Ruta del archivo del modelo entrenado')
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    try:
        logging.info("Inicio del script de predicción.")
        args = parse_arguments()

        # Carga los datos
        try:
            data = pd.read_csv(args.input_data)
            logging.debug("Número de filas en el dataset de entrada: "
                          "%d", len(data))
            logging.debug("Número de columnas en el dataset de entrada: "
                          "%d", len(data.columns))
            logging.debug("Path de entrada: %s", args.input_data)
        except pd.errors.EmptyDataError:
            logging.error("El archivo CSV está vacío.")
            raise
        except FileNotFoundError:
            logging.error("No se encontró el archivo CSV.")
            raise

        # Realiza las predicciones
        prediccion_precio(args.input_data)  # Pasa el argumento 'input_data'
        logging.info("Predicciones realizadas.")
    except Exception as e:
        logging.error("Error en el script de predicción.")
        logging.error(e, exc_info=True)
        raise
