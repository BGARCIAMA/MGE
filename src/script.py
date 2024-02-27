# script.py
'''Funciones que luego puedas importar a tu código principal
    (prep.py, train.py, inference.py)
'''
# Se importan las librerias necesarias
# pylint: disable = unused-import
import warnings
import yaml
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


def cargar_configuracion(ruta_config):
    '''De aqui se obtiene la configuracion del
    config.yaml
    '''
    with open(ruta_config, 'r', encoding='utf-8') as archivo:
        configuracion = yaml.safe_load(archivo)
    return configuracion


def descargar_datos(base_path_data, path_out):
    '''Descaga la info que vendrá de CSV del data raw
    Params:
        data_train: informacion del dataset de entrenamiento
        data_test:  informacion del dataset de test
    Returns:
        Un dataframe con las dos bases unidas
    '''
    # Lee los archivos csv con los datos
    data_train = pd.read_csv(base_path_data + "train.csv")
    data_test = pd.read_csv(base_path_data + "test.csv")

    # Quita las columnas de ID
    train_sin_id = data_train.drop('Id', axis=1)
    test_sin_id = data_test.drop('Id', axis=1)

    # Une las bases para la base total
    data_total = pd.concat([train_sin_id, test_sin_id], axis=0)

    data_total.to_csv(f"{path_out}/data_total.csv", index=False)
    print(f"Datos unidos guardados en {path_out}")

    return data_total


def impute_continuous_missing_data(data, missing_data_cols, passed_col):
    '''Completa las variables vacías usando
    un imputador (apoyándonos del código
    https://www.kaggle.com/code/muhammadibrahimqasmi/predicting-house-prices")
    '''
    df_null = data[data[passed_col].isnull()]
    df_not_null = data[data[passed_col].notnull()]

    x_var = df_not_null.drop(passed_col, axis=1)
    y_var = df_not_null[passed_col]

    # Transformación de variables categóricas
    label_encoder = LabelEncoder()
    x_var = x_var.apply(lambda col: label_encoder.fit_transform(col) if
                        col.dtype in ['object', 'category'] else col)

    # Imputación de valores faltantes
    it_im = IterativeImputer(estimator=RandomForestRegressor(random_state=123),
                             add_indicator=True)
    for col in [col for col in x_var.columns if col in missing_data_cols]:
        if x_var[col].isnull().sum() > 0:
            col_with_missing_values = x_var[col].values.reshape(-1, 1)
            imputed_values = it_im.fit_transform(
                col_with_missing_values)
            x_var[col] = imputed_values[:, 0]

    # División de datos y entrenamiento del modelo
    x_train, _, y_train, _ = train_test_split(x_var, y_var,
                                              test_size=0.2, random_state=123)
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x_train, y_train)

    # Imputación de valores faltantes en df_null
    x_var = df_null.drop(passed_col, axis=1)
    x_var = x_var.apply(lambda col: label_encoder.fit_transform(col)
                        if col.dtype in ['object', 'category'] else col)
    for col in [col for col in x_var.columns if col in missing_data_cols]:
        if x_var[col].isnull().sum() > 0:
            col_with_missing_values = x_var[col].values.reshape(-1, 1)
            imputed_values = it_im.fit_transform(col_with_missing_values)
            x_var[col] = imputed_values[:, 0]

    if not df_null.empty:
        df_null[passed_col] = rf_regressor.predict(x_var)

    # Concatenación de resultados
    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]


def preprocesar_datos(entrada_data, base_path_out_prep):
    '''Trabaja en la base total para preprocesar
    Params:
        data_total: base completa para prepocesar las variables para modelo
    Returns:
        Un dataframe con las variables finales a utilizar en el modelo
    '''
    numeric_cols = ['MSSubClass',
                    'LotFrontage',
                    'LotArea',
                    'OverallQual',
                    'OverallCond',
                    'YearBuilt',
                    'YearRemodAdd',
                    'MasVnrArea',
                    'BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    '1stFlrSF',
                    '2ndFlrSF',
                    'LowQualFinSF',
                    'GrLivArea',
                    'BsmtFullBath',
                    'BsmtHalfBath',
                    'FullBath',
                    'HalfBath',
                    'BedroomAbvGr',
                    'KitchenAbvGr',
                    'TotRmsAbvGrd',
                    'Fireplaces',
                    'GarageYrBlt',
                    'GarageCars',
                    'GarageArea',
                    'WoodDeckSF',
                    'OpenPorchSF',
                    'EnclosedPorch',
                    '3SsnPorch',
                    'ScreenPorch',
                    'PoolArea',
                    'MiscVal',
                    'MoSold',
                    'YrSold',
                    'SalePrice']

    # Cargar datos desde el archivo CSV
    data_total = pd.read_csv(entrada_data)

    # Quita variables con muchos missings
    var_sin_miss = data_total.drop(['PoolQC', 'MiscFeature', 'Alley',
                                    'Fence', 'FireplaceQu'], axis=1)

    # Se queda con variables numericas solamente
    var_modelo = var_sin_miss.select_dtypes(include=['float64', 'int64'])

    # Elimina filas y columas duplicadas
    var_modelo = var_modelo.drop_duplicates()
    var_modelo = var_modelo.reset_index(drop=True)

    warnings.filterwarnings('ignore')
    missing_data_cols = (var_modelo
                         .isnull()
                         .sum()
                         [var_modelo.isnull().sum() > 0]
                         .index
                         .tolist()
                         )

    # Imputamos informacioón en valores vacíos con nuestras funciones
    for col in missing_data_cols:
        if col in numeric_cols:
            var_modelo[col] = impute_continuous_missing_data(data_total,
                                                             missing_data_cols,
                                                             col)
        else:
            pass

    data_final = var_modelo[['OverallQual', 'YearBuilt',
                             'YearRemodAdd', 'LotFrontage',
                             'TotalBsmtSF', 'GrLivArea',
                             'GarageArea', 'SalePrice']]

    # Guardar el resultado en un nuevo archivo CSV
    data_final.to_csv(base_path_out_prep, index=False)
    print(f"La base preprocesada se guardo en {base_path_out_prep}")
    return data_final


def entrena_modelo(data_final, path_models, configuracion):
    '''Con la información ya preprocesada de data_final
    entrena el modelo
    '''

    # Lee csv
    data_final = pd.read_csv(data_final)

    # Seleccionar características numéricas y eliminar 'SalePrice'
    numerical_cols = (data_final
                      .select_dtypes(include=['int64', 'float64'])
                      .drop('SalePrice', axis=1)
                      .columns)
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ])

    # Obtener los parámetros del diccionario de configuración
    rf_params = configuracion['random_forest']

    # Crear la lista de modelos
    model_list = {
                  'Linear Regression': LinearRegression(),
                  'RFR': RandomForestRegressor(
                      n_estimators=rf_params['n_estimators'],)
    }

    x_model = data_final.drop('SalePrice', axis=1)
    y_model = data_final['SalePrice']
    x_train, x_test, y_train, y_test = train_test_split(x_model,
                                                        y_model, test_size=0.2,
                                                        random_state=123)

    pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', model)]) for name,
                 model in model_list.items()}

    rmse_results = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_results[name] = rmse

    rfr_model = RandomForestRegressor(
        n_estimators=configuracion['random_forest']['n_estimators'],
        max_depth=configuracion['random_forest']['max_depth'],
        min_samples_split=configuracion['random_forest']['min_samples_split'],
        random_state=configuracion['random_forest']['random_seed']
    )
    rfr_model.fit(x_train, y_train)

    joblib.dump(rfr_model, f"{path_models}/rfr_model.joblib")
    print(f"El modelo fue entrenado y guardado en {path_models}")


def prediccion_precio(input_data):
    '''Con los input que ingrese el usuario, se predice el
    precio de la casa con esas especificaciones
    '''
    # Definir las variables necesarias para la predicción
    variables = ['OverallQual', 'YearBuilt', 'YearRemodAdd',
                 'LotFrontage', 'TotalBsmtSF', 'GrLivArea', 'GarageArea']

    # Solicitar al usuario ingresar las variables de entrada
    print("Ingrese lo siguiente para predecir el precio de la casa:")
    ov_qual = input("Calidad de materiales y acabados (entre 1 y 10): ")
    year_built = input("Año en que se construyó: ")
    year_remo = input("Año de remodelación (si no, es = al de construcción): ")
    lot_front = float(input("Tamaño en pies cuadrados de entrada principal: "))
    total_bsmtsf = float(input("Tamaño en pies cuadrados del sótano: "))
    liv_area = float(input("Tamaño en pies cuadrados de la sup. habitable: "))
    gar_area = float(input("Tamaño en pies cuadrados de la cochera: "))

    user_input = pd.DataFrame({
        'OverallQual': [ov_qual],
        'YearBuilt': [year_built],
        'YearRemodAdd': [year_remo],
        'LotFrontage': [lot_front],
        'TotalBsmtSF': [total_bsmtsf],
        'GrLivArea': [liv_area],
        'GarageArea': [gar_area]
    })

    # Crear un df de la info a calcular
    input_data = pd.DataFrame(user_input, columns=variables)

    # Cargar el modelo previamente entrenado
    loaded_rfr = joblib.load("./models/rfr_model.joblib")

    # Resultado del modelo
    prediction = loaded_rfr.predict(input_data)

    # La predicción es:
    print(f'El precio estimado de la casa es: {prediction}')


# Utiliza la configuración en las funciones
descargar_datos("./data/raw/", "./data/raw/")
