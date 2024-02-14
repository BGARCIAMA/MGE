# Tarea de Métodos de Gran Escala 
# (MCD ITAM Primavera 2024)         

## Autor

| Nombre                        |  CU    | Correo Electrónico             | Usuario Github |
|-------------------------------|--------|--------------------------------|----------------|
| Blanca Estela García Manjarrez | 118886 | bgarci11@itam.mx               | BGARCIAMA      |

# Introducción  🧠
Este es un repositorio que contiene a detalle los scripts del modelo que permita estimar el precio de una casa dadas algunas características que el usuario deberá proporcionar a través de un front al momento de la inferencia.

# Contenido del repositorio  🎯
Aqui se incluye el árbol con la estructura de tu repositorio:
PENDIENTEEEE

# Base de datos  ✍
Usamos el [conjunto de precios de compra-venta de casas de la ciudad Ames, Iowa en Estados Unidos](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

La base contiene 80 variables descritas a continuación:
  \begin{itemize}
  \item ID: Identifier
  \itemSalePrice: the property's sale price in dollars. This is the target variable to predict
  \itemMSSubClass: The building class
  \itemMSZoning: The general zoning classification
  \itemLotFrontage: Linear feet of street connected to property
  \itemLotArea: Lot size in square feet
  \itemStreet: Type of road access
  \itemAlley: Type of alley access
  \itemLotShape: General shape of property
  \itemLandContour: Flatness of the property
  \itemUtilities: Type of utilities available
  \itemLotConfig: Lot configuration
  \itemLandSlope: Slope of property
  \itemNeighborhood: Physical locations within Ames city limits
  \itemCondition1: Proximity to main road or railroad
  \itemCondition2: Proximity to main road or railroad (if a second is present)
  \itemBldgType: Type of dwelling
  \itemHouseStyle: Style of dwelling
  \itemOverallQual: Overall material and finish quality
  \itemOverallCond: Overall condition rating
  \itemYearBuilt: Original construction date
  \itemYearRemodAdd: Remodel date
  \itemRoofStyle: Type of roof
  \itemRoofMatl: Roof material
  \itemExterior1st: Exterior covering on house
  \itemExterior2nd: Exterior covering on house (if more than one material)
  \itemMasVnrType: Masonry veneer type
  \itemMasVnrArea: Masonry veneer area in square feet
  \itemExterQual: Exterior material quality
  \itemExterCond: Present condition of the material on the exterior
  \itemFoundation: Type of foundation
  BsmtQual: Height of the basement
  BsmtCond: General condition of the basement
  BsmtExposure: Walkout or garden level basement walls
  BsmtFinType1: Quality of basement finished area
  BsmtFinSF1: Type 1 finished square feet
  BsmtFinType2: Quality of second finished area (if present)
  BsmtFinSF2: Type 2 finished square feet
  BsmtUnfSF: Unfinished square feet of basement area
  TotalBsmtSF: Total square feet of basement area
  Heating: Type of heating
  HeatingQC: Heating quality and condition
  CentralAir: Central air conditioning
  Electrical: Electrical system
  1stFlrSF: First Floor square feet
  2ndFlrSF: Second floor square feet
  \itemLowQualFinSF: Low quality finished square feet (all floors)
  \itemGrLivArea: Above grade (ground) living area square feet
  \itemBsmtFullBath: Basement full bathrooms
  \itemBsmtHalfBath: Basement half bathrooms
  \itemFullBath: Full bathrooms above grade
  \itemHalfBath: Half baths above grade
  \itemBedroom: Number of bedrooms above basement level
  \itemKitchen: Number of kitchens
  \itemKitchenQual: Kitchen quality
  \itemTotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
  \itemFunctional: Home functionality rating
  \itemFireplaces: Number of fireplaces
  \itemFireplaceQu: Fireplace quality
  \itemGarageType: Garage location
  \itemGarageYrBlt: Year garage was built
  \itemGarageFinish: Interior finish of the garage
  \itemGarageCars: Size of garage in car capacity
  \itemGarageArea: Size of garage in square feet
  \itemGarageQual: Garage quality
  \itemGarageCond: Garage condition
  \itemPavedDrive: Paved driveway
  \itemWoodDeckSF: Wood deck area in square feet
  \itemOpenPorchSF: Open porch area in square feet
  \itemEnclosedPorch: Enclosed porch area in square feet
  \item3SsnPorch: Three season porch area in square feet
  \itemScreenPorch: Screen porch area in square feet
  \itemPoolArea: Pool area in square feet
  \itemPoolQC: Pool quality
  \itemFence: Fence quality
  \itemMiscFeature: Miscellaneous feature not covered in other categories
  \itemMiscVal: Value of miscellaneous feature
  \itemMoSold: Month Sold
  \itemYrSold: Year Sold
  \itemSaleType: Type of sale
  \itemSaleCondition: Condition of sale
  \end{itemize}
