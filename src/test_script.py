import pandas as pd
import numpy as np
import os
from script import preprocesar_datos

def test_preprocesar_datos():
    # Crear datos de prueba
    entrada_data = "test_data.csv"
    base_path_out_prep = "preprocessed_data.csv"
    data_total = pd.DataFrame({
        'MSSubClass': [20, 30, 40],
        'LotFrontage': [70, 80, 90],
        'LotArea': [5000, 6000, 7000],
        'OverallQual': [5, 6, 7],
        'OverallCond': [5, 6, 7],
        'YearBuilt': [2000, 2001, 2002],
        'YearRemodAdd': [2000, 2001, 2002],
        'MasVnrArea': [100, 200, 300],
        'BsmtFinSF1': [500, 600, 700],
        'BsmtFinSF2': [0, 0, 0],
        'BsmtUnfSF': [1000, 1100, 1200],
        'TotalBsmtSF': [1500, 1700, 1900],
        '1stFlrSF': [1500, 1700, 1900],
        '2ndFlrSF': [0, 0, 0],
        'LowQualFinSF': [0, 0, 0],
        'GrLivArea': [1500, 1700, 1900],
        'BsmtFullBath': [1, 2, 3],
        'BsmtHalfBath': [0, 0, 0],
        'FullBath': [2, 2, 3],
        'HalfBath': [0, 0, 1],
        'BedroomAbvGr': [3, 3, 4],
        'KitchenAbvGr': [1, 1, 1],
        'TotRmsAbvGrd': [6, 7, 8],
        'Fireplaces': [1, 1, 2],
        'GarageYrBlt': [2000, 2001, 2002],
        'GarageCars': [2, 2, 3],
        'GarageArea': [500, 600, 700],
        'WoodDeckSF': [0, 0, 0],
        'OpenPorchSF': [50, 60, 70],
        'EnclosedPorch': [0, 0, 0],
        '3SsnPorch': [0, 0, 0],
        'ScreenPorch': [0, 0, 0],
        'PoolArea': [0, 0, 0],
        'MiscVal': [0, 0, 0],
        'MoSold': [1, 2, 3],
        'YrSold': [2000, 2001, 2002],
        'SalePrice': [100000, 200000, 300000]
    })

    # Crear archivo CSV de prueba
    data_total.to_csv(entrada_data, index=False)

    # Ejecutar la función a probar
    data_final = preprocesar_datos(entrada_data, base_path_out_prep)

    # Verificar que el archivo CSV de salida existe
    assert os.path.exists(base_path_out_prep)

    # Verificar que el número de columnas sea correcto
    assert len(data_final.columns) == 8

    # Verificar que las columnas sean correctas
    expected_columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'LotFrontage',
                        'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'SalePrice']
    assert all(col in data_final.columns for col in expected_columns)

    # Verificar que el número de filas sea correcto
    assert len(data_final) == 3

    # Verificar que los valores sean correctos
    expected_values = np.array([
        [5, 2000, 2000, 70, 1500, 1500, 500, 100000],
        [6, 2001, 2001, 80, 1700, 1700, 600, 200000],
        [7, 2002, 2002, 90, 1900, 1900, 700, 300000]
    ])
    assert np.allclose(data_final[expected_columns].values, expected_values)

    # Eliminar archivos de prueba
    os.remove(entrada_data)
    os.remove(base_path_out_prep)
