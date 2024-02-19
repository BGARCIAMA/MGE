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
import argparse
from src.script import prediccion_precio


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
    args = parse_arguments()
    # Realiza las predicciones
    prediccion_precio(args.input_data)  # Pasa el argumento 'input_data'
