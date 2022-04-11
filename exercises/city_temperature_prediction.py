import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    # check validity
    data = data[data['Year'] >= 0]
    data = data[data['Temp'] >= -30]

    data = data[(data['Month'] >= 1) & (data['Month'] <= 12)]
    data = data[(data['Day'] >= 1) & (data['Day'] <= 31)]

    # add day of year feature
    data['DayOfYear'] = data['Date'].dt.day_of_year
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('/Users/talbrauner/Desktop/Tal/YEAR2/Sem_B/IML/IML.HUJI/datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    data_israel = data[data['Country'] == 'Israel']
    data_israel = data_israel.astype({'Year':str})

    fig21 = px.scatter(data_israel, x='DayOfYear', y='Temp', color='Year',title="Temperture as a function of day "
                                                                                "of year colored by year")
    fig21.show()
    data_israel_month = data_israel.groupby(['Month'])['Temp'].std()
    fig22 = px.bar(data_israel_month, y='Temp',title="Temperatures std by Month")
    fig22.show()
    # Question 3 - Exploring differences between countries
    avg_temp_by_country_month = data.groupby(['Country', 'Month'])['Temp'].mean().reset_index()
    std_temp_by_country_month = data.groupby(['Country', 'Month'])['Temp'].std().reset_index()
    fig3 = px.line(avg_temp_by_country_month, x='Month', y='Temp', line_group='Country', color='Country',
                   error_y=std_temp_by_country_month['Temp'],title="Avarage and std of the temp for each country")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    temp = data_israel['Temp']
    data_israel.drop('Temp',axis=1,inplace=True)
    trainX,trainY,testX,testY = split_train_test(data_israel,temp,train_proportion=0.75)


    res = dict()
    for k in range(1,11):
        model = PolynomialFitting(k)
        model.fit(trainX['DayOfYear'].to_numpy(),trainY.to_numpy())
        res[k] = round(model.loss(testX['DayOfYear'],testY),2)
    res = pd.DataFrame(res.items(),columns=['Degree','Loss'])
    print(res)
    fig4 = px.bar(res,x='Degree',y="Loss",
           title="MSE Loss over different k degrees of polynomial fitting model")

    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5)
    model.fit(data_israel['DayOfYear'].to_numpy(),temp.to_numpy())
    res = dict()
    for country in data['Country'].unique():
        if country != 'Israel':
            data_by_country = data[data['Country'] == country]
            res[country] = model.loss(data_by_country['DayOfYear'], data_by_country['Temp'])
    fig5 = px.bar(pd.DataFrame(res.items(),columns=['Country','Loss']),x='Country',y='Loss'
                  ,title="Loss of fitted model over Israel dataset over other countries",color='Country').show()