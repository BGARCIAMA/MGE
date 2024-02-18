#script.py
'''Funciones que luego puedas importar a tu código principal
    (prep.py, train.py, inference.py)
'''
# Importa librerías
import locale
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# Path del repo
BASE_PATH_DATA = "./data/raw"
BASE_PATH_OUT = "./data/prep"

def descargar_datos(data_train,data_test):
    '''Descaga la info que vendrá de CSV del data raw
    Params:
        data_train: informacion del dataset de entrenamiento
        data_test:  informacion del dataset de test
    Returns:
        Un dataframe con las dos bases unidas
    '''
    #Lee los archivos csv con los datos
    data_train = pd.read_csv(BASE_PATH_DATA + "train.csv")
    data_test = pd.read_csv(BASE_PATH_DATA + "test.csv")

    #Quita las columnas de ID
    train_sin_id = data_train.drop('Id', axis=1)
    test_sin_id = data_test.drop('Id', axis=1)

    #Une las bases para la base total
    data_total = pd.concat([train_sin_id, test_sin_id], axis = 0)

    data_total.to_csv(f"{BASE_PATH_OUT}/data_total.csv", index=False)
    print(f"Datos preprocesados guardados en {BASE_PATH_OUT}/total_data.csv")

def impute_continuous_missing_data(data, missing_data_cols, passed_col):
    '''Completa las variables vacías usando un imputador (apoyándonos del código
    https://www.kaggle.com/code/muhammadibrahimqasmi/predicting-house-prices")
    '''
    bool_cols = []
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

    df_null = data[data[passed_col].isnull()]
    df_not_null = data[data[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
    iterative_imputer = IterativeImputer(estimator = RandomForestRegressor(random_state = 123),
                                         add_indicator=True)
    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123)
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass

    if len(df_null) > 0:
        df_null[passed_col] = rf_regressor.predict(X)
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

def preprocesar_datos(data_total):
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

    #Quita variables con muchos missings
    var_sin_miss = data_total.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'],
                                   axis = 1)

    #Se queda con variables numericas solamente
    var_modelo = var_sin_miss.select_dtypes(include = ['float64', 'int64'])


    #Elimina filas y columas duplicadas
    var_modelo = var_modelo.drop_duplicates()
    var_modelo = var_modelo.reset_index(drop = True)

    import warnings
    warnings.filterwarnings('ignore')
    missing_data_cols = var_modelo.isnull().sum()[var_modelo.isnull().sum() > 0].index.tolist()

    # Imputamos informacioón en valores vacíos con nuestras funciones
    for col in missing_data_cols :
        print("Missing Values", col, ":", str(round((
            var_modelo[col].isnull().sum()/len(var_modelo))*100, 2))+"%")
        if col in numeric_cols:
            var_modelo[col] = impute_continuous_missing_data(col)
        else:
            pass

    data_final = var_modelo[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'LotFrontage',
                            'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'SalePrice']]

def entrena_modelo(data_final):

    # Entrenamos el modelo
    # Seleccionar características numéricas y eliminar 'SalePrice'
    numerical_cols = data_final.select_dtypes(include =
                                              ['int64', 'float64']).drop('SalePrice',
                                                                         axis=1).columns
    numerical_transformer = StandardScaler()

    # Crear ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ])

    # Modelos
    model_list = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regression': RandomForestRegressor()
    }

    # División de la base de datos
    x = data_final.drop('SalePrice', axis=1)
    y = data_final['SalePrice']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # Crear pipelines
    pipelines = {name: Pipeline(steps = [('preprocessor', preprocessor),
                                         ('model', model)]) for name, model in model_list.items()}

    # Entrenamiento y evaluación de los modelos
    rmse_results = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_results[name] = rmse

    # Mostrar los RMSE resultantes
    rmse_results_sorted = dict(sorted(rmse_results.items(), key=lambda item: item[1]))

    # Se dividen las bases
    df_train, df_test = train_test_split(data_final, test_size=0.2, random_state=42)

    x_train = df_train.drop(['SalePrice'], axis = 1)
    y_train = df_train['SalePrice']

    # Se entrena con el XGBoost Regressor
    RFR_model = RandomForestRegressor()
    RFR_model.fit(x_train, y_train)

    # Se hace la predicción en la base test usando el modelo entrenado
    y_pred = RFR_model.predict(df_test.drop(['SalePrice'], axis = 1))

    joblib.dump(RFR_model, "./rfr_model.joblib")

def prediccion_precio():
    # Definir las variables necesarias para la predicción
    variables = ['OverallQual', 'YearBuilt', 'YearRemodAdd',
                'LotFrontage','TotalBsmtSF', 'GrLivArea', 'GarageArea']
    locale.setlocale(locale.LC_ALL, '')

    # Solicitar al usuario ingresar las variables de entrada
    print("Ingrese los valores de las variables para predecir el precio de la casa:")
    ov = input("OverallQual - Calidad general de materiales y acabados (valor entre 1 y 10): ")
    yearB = input("YearBuilt - Año en que se construyó: ")
    yearRemo = input("YearRemoAdd - Año en que se remodeló (si no, es igual al de construcción): ")
    LT = float(input("LotFrontage - Tamaño en pies cuadrados de la entrada a la calle: "))
    TotalB = float(input("TotalBsmtSF - Tamaño en pies cuadrados del sótano: "))
    GrL = float(input("GrLivArea - Tamaño en pies cuadrados de la superficie habitable: "))
    GarArea = float(input("GarageArea - Tamaño en pies cuadrados de la cochera: "))

    user_input = pd.DataFrame({
        'OverallQual': [ov],
        'YearBuilt': [yearB],
        'YearRemodAdd': [yearRemo],
        'LotFrontage': [LT],
        'TotalBsmtSF': [TotalB],
        'GrLivArea': [GrL],
        'GarageArea': [GarArea]
    })

    # Crear un df de la info a calcular
    input_data = pd.DataFrame(user_input, columns = variables)

    # Cargar el modelo previamente entrenado
    loaded_rfr = joblib.load("./rfr_model.joblib")

    # Resultado del modelo
    prediction = loaded_rfr.predict(input_data)
    prediction_formateada = locale.currency(prediction[0], grouping=True)

    # La predicción es:
    print(f'El precio estimado de la casa es: {prediction_formateada}')
