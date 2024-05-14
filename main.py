import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_path = "Data_sets/ready/db_variation.xlsx"
well_data = pd.read_excel(data_path)
well_data.info()
well_data = well_data.dropna(axis='columns', thresh=(889*0.85))
well_data = well_data.dropna(axis='index')
print(well_data.columns)

feauters_start = ['A04M01N', 'A1M01N', 'A2M05N', 'BK', 'IK', 'NKTB', 'W', 'GK']
target = 'Пористость в атм, %'
well_data = well_data[[target, 'A04M01N', 'A1M01N', 'A2M05N', 'BK', 'IK', 'NKTB', 'W', 'GK']]
#well_data = well_data.drop(['Месторождение', 'Скважина', 'Пласт', 'Глубина(привзяка). м','DEPT'], axis=1)
correlation_matrix = well_data.corr(numeric_only=True)
plt.figure(figsize= (20, 15))
sns.heatmap(correlation_matrix, annot = True, annot_kws={"size":7}, vmin=-1, vmax=1, center=0, cmap ='RdYlGn')
plt.show()
gis_data = pd.read_csv('gis_data/v_terr.csv', sep='*')
gis_data.info()
gis_data.set_index(['DEPT', 'NSKV'], inplace=True)
gis_test = gis_data[['A04M01N', 'A1M01N', 'BK', 'IK', 'NKTB', 'W', 'GK']]
gis_test = gis_test.dropna(axis='index')


def choose_feauters_by_corelation(features, target):
    correlation_with_target = well_data.corr(numeric_only=True)[target].drop(target)
    threshold = 0.05
    high_correlation_features = correlation_with_target[abs(correlation_with_target) > threshold].index.tolist()
    return high_correlation_features

feauters_start = choose_feauters_by_corelation(feauters_start, target)
#well_data.to_excel("Bd_0.xlsx")



def create_data_set(data, feauters):
    from sklearn.model_selection import train_test_split
    y = data[target]
    X = data[feauters]
    X = scaling_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    np.savetxt('y_test', y_test, fmt='%.2f', delimiter=' ')
    return X_train, X_test, y_train, y_test


def scaling_data(x):
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    x_scaled = ss.fit_transform(x.values)
    x_new = pd.DataFrame(x_scaled, columns=x.columns)
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
    # ax1.set_title('Before Scaling')
    # sns.kdeplot(x['MPZ'], ax=ax1)
    # sns.kdeplot(x['GK'], ax=ax1)
    # sns.kdeplot(x['NKTB'], ax=ax1)
    # ax2.set_title('After Standard Scaler')
    # sns.kdeplot(x_new['MPZ'], ax=ax2)
    # sns.kdeplot(x_new['GK'], ax=ax2)
    # sns.kdeplot(x_new['NKTB'], ax=ax2)
    # plt.show()
    return x_new
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
    # np.savetxt('lin_regres_prediction', lin_regres_prediction, fmt='%.18e', delimiter=' ')
    # np.savetxt('lasso_prediction', lasso_prediction, fmt='%.18e', delimiter=' ')
    return lin_regres_prediction, lasso_prediction

def RFR(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GridSearchCV
    rfr_model = RandomForestRegressor()
    rfr_model.fit(X_train, y_train)
    rfr_predictions = rfr_model.predict(X_test)
    parametrs = {   'n_estimators': [50, 100, 150],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                 }
    rs = GridSearchCV(rfr_model,
                      parametrs,
                      cv=5,
                      scoring='neg_mean_squared_error')
    rs.fit(X_train, y_train)
    print(rs.best_params_)
    np.savetxt('rfr_predictions', rfr_predictions, fmt='%.2f', delimiter=' ')
    return rfr_model, rfr_predictions

def Gr_boost(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import GradientBoostingRegressor
    grb_model = GradientBoostingRegressor(random_state=20)
    grb_model.fit(X_train,y_train)
    grb_prediction = grb_model.predict(X_test)
    np.savetxt('grb_prediction', grb_prediction, fmt='%.2f', delimiter=' ')
    return grb_model, grb_prediction


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
    x = well_data[new_feature]
    x = scaling_data(x)
    rf_model.fit(x, well_data[target])
    gis_d = gis_test[new_feature]
    gis_scaled = scaling_data(gis_d)
    predictet_poro = rf_model.predict(gis_scaled)
    gis_out = gis_d
    gis_out['poro'] = predictet_poro

    hist, bins = np.histogram(predictet_poro, bins=10, density=True)
    percentages = (hist / np.sum(hist)) * 100
    plt.bar(bins[:-1], percentages, width=np.diff(bins))
    plt.xlabel('Value')
    plt.ylabel('Percentage')
    plt.title('Percentage Histogram')
    plt.show()
    gis_out.to_csv('gis_data/v_terr_predicted_2.prn', sep='*')
    return


def GradientBoost():
    X_train, X_test, y_train, y_test = create_data_set(well_data, feauters_start)
    grb_model, grb_prediction = Gr_boost(X_train, X_test, y_train, y_test)
    new_feature = choose_parametrs(grb_model, feauters_start)
    X_train, X_test, y_train, y_test = create_data_set(well_data, new_feature)
    grb_model, grb_prediction = Gr_boost(X_train, X_test, y_train, y_test)
    quality_of_regression_model("GradientBoostingRegressor", grb_prediction, y_test)
    return

# Line_regression_models()
RandomForrest()
#GradientBoost()
