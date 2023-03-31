import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_path = "D:\Machine_learning\Data_sets\DataWithoutNaN_columns.csv"
well_data = pd.read_csv(data_path, sep = ",")
# well_data.info()
# well_data = well_data.dropna(axis = 1)
print(well_data.columns)
feauters_start = ['AF90', 'AF30', 'AF60', 'GR', 'HCAL', 'HTEM', 'PEFZ', 'RHOZ', 'RLA3',
       'RLA4', 'SHALE']
# # well_data.to_excel("DataWithoutNaN_columns.xlsx")


def create_data_set(data, feauters):
    from sklearn.model_selection import train_test_split
    y = data.poro
    X = data[feauters]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 1, test_size=0.2)
    return X_train, X_test, y_train, y_test


def scaling_data(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    y_train = np.array(y_train)
    return X_train_scaled, X_test_scaled, y_train
# def visuali_importenses(feature_imp):
#     sns.barplot(x=feature_imp, y=feature_imp.index)
#     plt.show()


def choose_parametrs(model, index_arrey):
    feature_imp = pd.Series(model.feature_importances_, index=index_arrey)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.show()
    feature_dict = dict(zip(feature_imp.index.tolist(), feature_imp.values.tolist()))
    new_feature = []
    for feature in feature_dict:
        importenses = feature_dict[feature]
        if importenses >= 0.01:
            # print("Добавлен: ", feature)
            new_feature.append(feature)
    print(new_feature)
    return new_feature


def lin_regr(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    lin_regres_model = LinearRegression()
    lin_regres_model.fit(X_test,y_test)
    lin_regres_prediction = lin_regres_model.predict(X_test)
    #ЛАССО
    from sklearn.linear_model import Lasso
    lasso_model = Lasso(alpha=1)
    lasso_model.fit(X_test,y_test)
    lasso_prediction = lasso_model.predict(X_test)
    return lin_regres_prediction, lasso_prediction

def RFR(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    rfr_model = RandomForestRegressor(criterion= 'squared_error', random_state=1)
    rfr_model.fit(X_train, y_train)
    rfr_predictions = rfr_model.predict(X_test)
    return rfr_model, rfr_predictions

def Gr_boost(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import GradientBoostingRegressor
    grb_model = GradientBoostingRegressor(random_state=1)
    grb_model.fit(X_train,y_train)
    grb_prediction = grb_model.predict(X_test)
    return grb_model, grb_prediction


def RFC(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    rfc_model = RandomForestClassifier()
    rfc_model.fit(X_train, y_train)
    GHE_pred = rfc_model.predict(X_test)
    return rfc_model, GHE_pred


def quality_of_classification_model (method, y_test, y_pred):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    A_s = accuracy_score(y_test, y_pred)
    print('accuracy_score', method, ': ', A_s)


def quality_of_regression_model(method, prediction, y_test):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    arrey_to_list = prediction.tolist()
    count = 0
    for i in arrey_to_list:
        # print(arrey_to_list[count])
        count += 1
    mae = mean_absolute_error(y_test, prediction)
    determination = r2_score(y_test, prediction)
    MSE = mean_squared_error(y_test, prediction)
    MAPE = mean_absolute_percentage_error(y_test, prediction)
    print('MAE for', method,': ',  mae)
    print('MAPE', method, ': ', MAPE)
    print('R2', method,': ',  determination)
    print('MSE', method, ': ', MSE)
    return



# Классификация - на потом.
# X_train, X_test, y_train, y_test = create_data_set(well_data,feauters_start)
#
# rfc_model, GHE_pred = RFC(X_train, X_test, y_train, y_test)
# new_feature = choose_parametrs(rfc_model, feauters_start)
# X_train, X_test, y_train, y_test = create_data_set(well_data, new_feature)
# rfc_model, GHE_pred = RFC(X_train, X_test, y_train, y_test)
# quality_of_classification_model("RandomForestClassifier", y_test, GHE_pred)

def Line_regression_models():
    X_train, X_test, y_train, y_test = create_data_set(well_data, feauters_start)
    lin_regres_prediction, lasso_prediction =lin_regr(X_train, X_test, y_train, y_test)
    quality_of_regression_model("LinearRegression", lin_regres_prediction, y_test)
    quality_of_regression_model("LassoRegression", lasso_prediction, y_test)
    return


def RandomForrest():
    X_train, X_test, y_train, y_test = create_data_set(well_data, feauters_start)
    rf_model, rf_fzi = RFR(X_train, X_test, y_train, y_test)
    new_feature = choose_parametrs(rf_model, feauters_start)
    X_train, X_test, y_train, y_test = create_data_set(well_data, new_feature)
    rf_model, rf_fzi = RFR(X_train, X_test, y_train, y_test)
    quality_of_regression_model("RandomForestRegression", rf_fzi, y_test)
    return


def GradientBoost():
    X_train, X_test, y_train, y_test = create_data_set(well_data, feauters_start)
    grb_model, grb_prediction = Gr_boost(X_train, X_test, y_train, y_test)
    new_feature = choose_parametrs(grb_model, feauters_start)
    X_train, X_test, y_train, y_test = create_data_set(well_data, new_feature)
    grb_model, grb_prediction = Gr_boost(X_train, X_test, y_train, y_test)
    quality_of_regression_model("GradientBoostingRegressor", grb_prediction, y_test)
    return

Line_regression_models()
RandomForrest()
GradientBoost()