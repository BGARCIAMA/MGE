# prep.py
'''Este script prepara los datos para incorporarlos al modelo
    Las funciones dentro de este script son:
    - descargar_datos
    - impute_continuous_missing_data
    - preprocesar_datos
'''
#path del repo
base_path_data = "./data/raw"

#Importa librerías
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder 


def descargar_datos(data_train,data_test):
    '''Descaga la info que vendrá de CSV del data raw
    Params:
        data_train: informacion del dataset de entrenamiento
        data_test:  informacion del dataset de test
    Returns:
        Un dataframe con las dos bases unidas
    '''
    #Lee los archivos csv con los datos
    data_train = pd.read_csv(base_path_data + "train.csv")
    data_test = pd.read_csv(base_path_data + "test.csv")

    #Quita las columnas de ID
    train_sin_ID = data_train.drop('Id', axis=1)
    test_sin_ID = data_test.drop('Id', axis=1)

    #Une las bases para la base total
    data_total = pd.concat([train_sin_ID, test_sin_ID], axis = 0)

    data_total.to_csv(data_total, index=False)
    print(f"Datos preprocesados guardados en {data_total}")

def impute_continuous_missing_data(passed_col):
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

    df_null = var_modelo[var_modelo[passed_col].isnull()]
    df_not_null = var_modelo[var_modelo[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missingdata if col != passed_col]
    
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
    
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state = 123), add_indicator=True)

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
    #Quita variables con muchos missings
    var_sin_miss = data_total.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)

    #Se queda con variables numericas solamente
    var_modelo = var_sin_miss.select_dtypes(include = ['float64', 'int64'])


    #Elimina filas y columas duplicadas
    var_modelo = var_modelo.drop_duplicates()
    var_modelo = var_modelo.reset_index(drop = True)
    
    import warnings
    warnings.filterwarnings('ignore')

    # Imputamos informacioón en valores vacíos con nuestras funciones
    for col in missingdata:
        print("Missing Values", col, ":", str(round((var_modelo[col].isnull().sum() / len(var_modelo)) * 100, 2))+"%")
        if col in numeric_cols:
            var_modelo[col] = impute_continuous_missing_data(col)
        else:
            pass
   
    data_final = var_modelo[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'LotFrontage', 
                            'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'SalePrice']]

    #duda si ponerlo así: 
    data_final.to_csv(base_path_data, index=False)
