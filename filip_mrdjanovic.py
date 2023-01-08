import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


part = 3

pd.set_option('display.float_format', lambda x: '%.2f' % x) # ======== Formatting numbers to 2 decimals

# ======== Load csv file
csv_data = pd.read_csv('GuangzhouPM20100101_20151231.csv')

# ======== Drop PM rows except PM_US Post
csv_data = csv_data.drop(['PM_City Station', 'PM_5th Middle School'], axis=1)

# ======== Replace NaN values with the string 'Unknown'
csv_data['cbwd'] = csv_data['cbwd'].fillna('Unknown')

# ======== Get the index labels of the rows that contain the string 'Unknown' in the 'cbwd' column
to_drop = csv_data[csv_data['cbwd'].str.contains('Unknown')].index

# ======== Drop the rows with the index labels obtained above
csv_data = csv_data.drop(to_drop)
csv_data = csv_data.drop(['No'], axis=1)

# ======== Convert the column to a string data type
csv_data['cbwd'] = csv_data['cbwd'].astype(str)

# ======== Drop rows with null values
csv_data = csv_data.dropna()

# ======== Get unique values from the given column
csv_data['HUMI'] = csv_data['HUMI'].clip(lower=0)

if part == 1:
    # ======== Get unique values from the given column
    unique_values = csv_data['HUMI'].unique()

    # # ======== Get column types of 'csv_data' dataset
    types = csv_data.dtypes

    # # ======== Get info of 'csv_data' dataset
    info = csv_data.info()

    # # ======== Describe data
    description = csv_data['PM_US Post'].describe()

    # # ======== Preview data - first 5 (default) rows  of the DataFrame
    csv_data.head()

    # # ======== Select a column to focus on
    column = 'PM_US Post'

    # # ======== Plot the dependence of 'sepal_width' on the other columns
    sb.pairplot(csv_data, x_vars=[col for col in csv_data.columns if col != column], y_vars=[column])
    plt.show()

    # Compute the correlations
    corr_mat = csv_data.corr()

    # Create a heatmap
    sb.heatmap(corr_mat, annot=True)
    plt.show()

    # ======== Group the data by year
    groups = csv_data.groupby('year')

    # ======== Set up a figure with multiple subplots
    fig, axs = plt.subplots(nrows=len(groups), ncols=1, figsize=(8, 12))

    # ======== Iterate through the groups
    for ax, (name, group) in zip(axs, groups):
        # ======== Plot the data for each group
        ax.plot(group['month'], group['PM_US Post'])
        ax.set_title(name)
    fig.tight_layout()
    plt.show()

def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) # np.mean((y_test-y_predicted)**2)
    mae = mean_absolute_error(y_test, y_predicted) # np.mean(np.abs(y_test-y_predicted))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))

# LINEARNA REGRESIJA
if part == 2:

    opt = 0
    csv_data_regression = csv_data
    csv_data_dummy = pd.get_dummies(csv_data_regression['cbwd'])
    csv_data_regression = pd.concat([csv_data_regression, csv_data_dummy], axis=1)
    csv_data_regression.drop(['cbwd'], axis=1, inplace=True)

    X = csv_data_regression.drop(columns=['PM_US Post'], axis=1).copy()
    y = csv_data_regression['PM_US Post'].copy()

    # Split the data into a training set, a validation set, and a test set, if test_size=0.3 then train_size is 0.7 => splitting 0.3 in half results 2 datasets of 15% data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Calculate the percentage size of each set
    train_size = len(X_train) / len(csv_data_regression) * 100
    val_size = len(X_val) / len(csv_data_regression) * 100
    test_size = len(X_test) / len(csv_data_regression) * 100

    # Print the percentage size of each set
    print(f'Training set size: {train_size:.2f}%')
    print(f'Validation set size: {val_size:.2f}%')
    print(f'Test set size: {test_size:.2f}%')

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    if opt == 1:
        # Evaluacija
        model_evaluation(y_test, y_pred, X_train.shape[0], X_train.shape[1])

        # Ilustracija koeficijenata
        plt.figure(figsize=(10,5))
        plt.bar(range(len(model.coef_)),model.coef_)
        plt.show()
        print("koeficijenti: ", model.coef_)

    if opt == 2:
        X_sm = sm.add_constant(X_train)

        model = sm.OLS(y_train, X_sm.astype('float')).fit()
        model.summary()
        print(model.summary())

    if opt == 3:
        numeric_feats = [item for item in X.columns if 'cbwd' not in item]
        print(numeric_feats)
        dummy_feats = [item for item in X.columns if 'cbwd' in item]
        print(dummy_feats)

        scaler = StandardScaler()
        scaler.fit(X_train[numeric_feats])

        x_train_std = pd.DataFrame(scaler.transform(X_train[numeric_feats]), columns = numeric_feats)
        x_test_std = pd.DataFrame(scaler.transform(X_test[numeric_feats]), columns = numeric_feats)

        x_train_std = pd.concat([x_train_std, X_train[dummy_feats].reset_index(drop=True)], axis=1)
        x_test_std = pd.concat([x_test_std, X_test[dummy_feats].reset_index(drop=True)], axis=1)

        x_train_std.head()
        regression_model_std = LinearRegression()

        # Obuka modela
        regression_model_std.fit(x_train_std, y_train)

        # Testiranje
        y_predicted = regression_model_std.predict(x_test_std)

        # Evaluacija
        model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

        # Ilustracija koeficijenata
        plt.figure(figsize=(10,5))
        plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
        plt.show()
        print("koeficijenti: ", regression_model_std.coef_)
    
        corr_mat = X_train[numeric_feats].corr()

        plt.figure(figsize=(12, 9))
        sb.heatmap(corr_mat, annot=True)
        plt.show()
    
    if opt == 4:
        test_score = model.score(X_test, y_test)
        val_score = model.score(X_val, y_val)

        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        X_val_poly = poly.transform(X_val)

        model = Lasso(alpha=0.1)
        model.fit(X_train_poly, y_train)

        y_pred = model.predict(X_val_poly)

        model_evaluation(y_test, y_pred, X_train_poly.shape[0], X_train_poly.shape[1])

        test_score = model.score(X_test_poly, y_test)
        val_score = model.score(X_val_poly, y_val)

        plt.scatter(y_val, model.predict(X_val_poly))
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.show()

    # if opt == 4:
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    x_inter_train = poly.fit_transform(X_train)
    x_inter_test = poly.transform(X_test)

    # print(poly.get_feature_names())

    # Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

    # Inicijalizacija
    regression_model_inter = LinearRegression()

    # Obuka modela
    regression_model_inter.fit(x_inter_train, y_train)

    # Testiranje
    y_predicted = regression_model_inter.predict(x_inter_test)

    # Evaluacija
    model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

    # Ilustracija koeficijenata
    # plt.figure(figsize=(10,5))
    # plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
    # plt.show()
    # print("koeficijenti: ", regression_model_inter.coef_)

    # RIDGE
    # if opt == 5:
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    x_inter_train = poly.fit_transform(X_train)
    x_inter_test = poly.transform(X_test)

    # Inicijalizacija
    ridge_model = Ridge(alpha=5)

    # Obuka modela
    ridge_model.fit(x_inter_train, y_train)

    # Testiranje
    y_predicted = ridge_model.predict(x_inter_test)

    # Evaluacija
    model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


    # Ilustracija koeficijenata
    # plt.figure(figsize=(10,5))
    # plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
    # plt.show()
    # print("koeficijenti: ", ridge_model.coef_)

    # LASSO
    # if opt == 6:

    # Model initialization
    lasso_model = Lasso(alpha=0.01)

    # Fit the data(train the model)
    lasso_model.fit(x_inter_train, y_train)

    # Predict
    y_predicted = lasso_model.predict(x_inter_test)

    # Evaluation
    model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

    #ilustracija koeficijenata
    # plt.figure(figsize=(10,5))
    # plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
    # plt.show()
    # print("koeficijenti: ", lasso_model.coef_)

        
    plt.figure(figsize=(10,5))
    #plt.plot(regression_model_degree.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'linear',zorder=7) # zorder for ordering the markers
    plt.plot(ridge_model.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge') # alpha here is for transparency
    plt.plot(lasso_model.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Lasso')
    plt.xlabel('Coefficient Index',fontsize=16)
    plt.ylabel('Coefficient Magnitude',fontsize=16)
    plt.legend(fontsize=13,loc='best')
    plt.show()

# KNN KLASIFIKATOR
if part == 3:

    csv_data_regression = csv_data
    
    csv_data_dummy = pd.get_dummies(csv_data_regression['cbwd'])
    csv_data_regression = pd.concat([csv_data_regression, csv_data_dummy], axis=1)
    csv_data_regression.drop(['cbwd'], axis=1, inplace=True)

    csv_data_regression['bezbedno'] = 0
    csv_data_regression['nebezbedno'] = 0
    csv_data_regression['opasno'] = 0

    csv_data_regression['bezbedno'] = csv_data_regression['PM_US Post'].apply(lambda x: 1 if x < 55.4 else 0)
    csv_data_regression['nebezbedno'] = csv_data_regression['PM_US Post'].apply(lambda x: 1 if x >= 55.5 and x < 150.4 else 0)
    csv_data_regression['opasno'] = csv_data['PM_US Post'].apply(lambda x: 1 if x >= 150.5 else 0)

    X = csv_data_regression.iloc[:, 1:]
    y = csv_data_regression.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    X = csv_data_regression.drop(columns=['PM_US Post'])
    y = csv_data_regression['PM_US Post']


    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # parameters = {'metric':['minkowski', 'chebyshev', 'euclidean', 'manhattan', "hamming"], 'n_neighbors':range(1, 10)}
    # clf = GridSearchCV(estimator=knn, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
    # clf.fit(X_train, y_train)
    # print(f'Best score: {clf.best_score_:.2f}')
    # print(f'Best parameters: {clf.best_params_}')
    # parameters = {'metric':'manhattan', 'n_neighbors':1} # Output of knn

    knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test) 
    conf_mat = confusion_matrix(y_test, y_pred) # TN, FP
    print(conf_mat)  

    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F_score = 2*precision*sensitivity/(precision+sensitivity)
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)
