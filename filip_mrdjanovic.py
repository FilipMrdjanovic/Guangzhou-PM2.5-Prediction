import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

part = 2

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

if part == 2:
    # Assume that the independent variable is stored in the "x" column and the dependent variable is stored in the "y" column
    X = csv_data["year", "month", "day", "hour", "season", "DEWP", "HUMI", "PRES", "TEMP", "cbwd", "Iws", "precipitation", "Iprec"]
    y = csv_data["PM_US Post"]

    # Create the linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)
    # Plot the data
    sb.scatterplot(x="x", y="PM_US Post", data=csv_data)

    # Get the predicted values for the model
    y_pred = model.predict(x)

    # Plot the model
    sb.lineplot(x="x", y=y_pred, color="red")

    # Show the plot
    plt.show()