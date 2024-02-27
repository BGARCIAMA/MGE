# train.py

# Importa las librerias
# from networkx import configuration_model
# from src.script import entrena_modelo, cargar_configuracion

# PATH_PREP = "data/prep/data_prep.csv"
# PATH_MODELS = "./models"
# configuracion = cargar_configuracion('config.yml')

# Entrena el modelo
# entrena_modelo(PATH_PREP, PATH_MODELS, configuracion)

# --------------------------------------
# train.py
'''Este script entrena el modelo con los datos de prep
    Las funciones dentro de este script son:
    - entrena_modelo
'''
import os
import argparse
import logging
from datetime import datetime
from src.script import entrena_modelo, cargar_configuracion


if not os.path.exists("logs/"):
    os.makedirs("logs/")
# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_train_file_name = f"logs/{date_time}_train.log"
logging.basicConfig(
    filename=log_train_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def parse_arguments():
    '''Parsea los argumentos de la línea de comandos'''
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo')
    parser.add_argument('--path_prep',
                        type=str, default="data/prep/data_prep.csv",
                        help='Ruta del CSV preprocesado para entrenamiento')
    parser.add_argument('--path_models', type=str, default="./models",
                        help='Ruta para guardar el modelo entrenado')
    parser.add_argument('--config_file', type=str, default='config.yml',
                        help='Ruta del YAML de configuración')
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    logging.info("Inicio del script de entrenamiento.")
    
    args = parse_arguments()
    configuracion = cargar_configuracion(args.config_file)

    # Entrena el modelo
    entrena_modelo(args.path_prep, args.path_models, configuracion)

    logging.info("Finalizo el script de entrenamiento.")