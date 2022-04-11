from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna().drop_duplicates()

    # drop unindicative values
    data.drop("id", axis=1, inplace=True)
    data.drop("date", axis=1, inplace=True)
    data.drop("lat", axis=1, inplace=True)
    data.drop("long", axis=1, inplace=True)
    data.drop('condition', axis=1, inplace=True)

    # remove samples with values that dont make sense
    for feat in ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'yr_built', 'sqft_living15',
                 'yr_renovated','sqft_basement', 'sqft_above', 'sqft_lot']:
        data = data[data[feat] >= 0]
    data = data[data['yr_built'] > 0]

    data = data[data['zipcode'] > 0]
    data = data[data['grade'].isin(range(0, 14))]
    data = data[data['view'].isin(range(0, 5))]
    data.zipcode = data.zipcode.astype(int)

    # add new features
    max_year = np.max(data["yr_built"])
    data["recently_built"] = data["yr_built"] >= (max_year - 20)
    data['yr_built'] = data['yr_built']/10
    #remove outliers
    data['is_renovated'] = data["yr_renovated"] >= 1960
    data = data[data['bedrooms'] < 8]

    data['sqft_lot'] = data['sqft_lot'].apply(lambda x:np.sqrt(x))


    # get a dict of zipcode:avg_price in zipcode
    d = data.groupby('zipcode')['price'].mean().round().to_dict()
    set_avg = lambda x: d[x]
    data['avg_price_by_zipcode'] = data['zipcode'].apply(set_avg)
    data.drop('zipcode', axis=1, inplace=True)


    return data


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sigma_y = np.std(y)
    for x in X:
        cor = np.cov(X[x],y.T)[0,1]/(sigma_y*np.std(X[x]))
        fig = go.Figure([go.Scatter(x=X[x], y=y, mode='markers',marker= dict(color = 'maroon'))],
                  layout=go.Layout(title="Price As Function of the " + x + "\n Correlation: "+ str(round(cor,4)),
                                   xaxis_title=x,
                                   yaxis_title="Price",
                                   height=500))

        fig.write_image(output_path+x+".jpeg")
        # print(x,np.cov(X[x], y.T)[0, 1] / (sigma_y * np.std(X[x])))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    d = load_data("/Users/talbrauner/Desktop/Tal/YEAR2/Sem_B/IML/IML.HUJI/datasets/house_prices.csv")
    d = d.astype('float64')
    price = d['price']
    d.drop("price", axis=1, inplace=True)
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(d, price, '/Users/talbrauner/Desktop/Tal/YEAR2/Sem_B/IML/EX/ex2/plots2')
    # Question 3 - Split samples into training- and testing sets.

    X_train, y_train, X_test, y_test = split_train_test(d, price)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    model = LinearRegression()
    mean_loss = []
    var_loss = []
    train_all = pd.concat([X_train, y_train], axis=1)
    for p in range(10, 101):
        loss_p = []
        for i in range(10):
            train_p= train_all.sample(frac=(p / 100))
            model.fit(train_p.iloc[:, :-1], train_p.iloc[:, -1:])
            loss_p.append(model.loss(X_test.to_numpy(), y_test.to_numpy().reshape(-1,1)))

        mean_loss.append((np.mean(loss_p)))
        var_loss.append(np.std(loss_p))


    mean_loss = np.array(mean_loss)
    var_loss = np.array(var_loss)
    a = np.arange(10,101)
    fig4 = go.Figure([go.Scatter(x=a, y=mean_loss, mode='markers+lines', name="Mean Loss"),
                      go.Scatter(x=a, y=(mean_loss-2*var_loss), fill='tonexty', mode="lines",
                                 line=dict(color="lightgrey"), showlegend=False),
                      go.Scatter(x=a, y=mean_loss + 2 * var_loss, fill='tonexty', mode="lines",
                                 line=dict(color="lightgrey"), showlegend=False)]
                     ,layout=go.Layout(title="Mean and Variance of the loss as a function of proportions of train set",
                                       xaxis_title = "Prencetage of train set",
                                       yaxis_title = "MSE of predicted price"))
    fig4.show()