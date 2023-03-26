import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_path = "D:\Machine_learning\Data_sets\dataSet_Well5_GHE.csv"
well_data = pd.read_csv(data_path, sep = ",")
# well_data.info()
well_data = well_data.dropna()
# well_data.info()
feauters_start = ['AFRT', 'AF90', 'AF30', 'AF60', 'AF20', 'AF10', 'AMF', 'BIT', 'CFTC',
       'CNTC', 'GR', 'HCAL', 'HDRA', 'HMIN', 'HMNO', 'HTEM', 'PEFZ', 'RHOZ',
       'RLA1', 'RLA2', 'RLA3', 'RLA4', 'RLA5', 'RT_HRLT', 'RXOZ', 'SP', 'KPA',
       'DOLM', 'KPob', 'KPD', 'KPkv', 'KPmz', 'KPN', 'KSDR', 'KTIM', 'LIME',
       'PORN', 'PORW', 'SHALE', 'TCMR', 'CMFF', 'SWE']



def create_data_set(data, feauters):
    from sklearn.model_selection import train_test_split
    y = data.GHE
    X = data[feauters]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 1, test_size=0.25)
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
        if importenses >= 0.02:
            print("Добавлен: ", feature)
            new_feature.append(feature)
    print(new_feature)
    return new_feature



def RFR(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(criterion= 'squared_error', random_state=1)
    rf_model.fit(X_train, y_train)
    rf_fzi_predictions = rf_model.predict(X_test)
    return rfr_model, rfr_fzi_predictions


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
    print("y_test: ", "\n", y_test)
    print(type(prediction))
    print("prediction: \n",)
    arrey_to_list = prediction.tolist()
    count = 0
    for i in arrey_to_list:
        print(arrey_to_list[count])
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

X_train, X_test, y_train, y_test = create_data_set(well_data,feauters_start)

rfc_model, GHE_pred = RFC(X_train, X_test, y_train, y_test)
new_feature = choose_parametrs(rfc_model, feauters_start)
X_train, X_test, y_train, y_test = create_data_set(well_data, new_feature)
rfc_model, GHE_pred = RFC(X_train, X_test, y_train, y_test)
quality_of_classification_model("RandomForestClassifier", y_test, GHE_pred)

# rf_model, rf_fzi = RFR(X_train, X_test, y_train, y_test)
# new_feature = choose_parametrs(rf_model, feauters_start)
# X_train, X_test, y_train, y_test = create_data_set(well_data, new_feature)
# rf_model, rf_fzi = RFR(X_train, X_test, y_train, y_test)



