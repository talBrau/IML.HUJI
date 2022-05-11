import numpy as np
from typing import Tuple


from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_loss = []
    test_loss = []
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)
    learners_list = np.arange(1, n_learners)
    for i in learners_list:
        train_loss.append(adaBoost.partial_loss(train_X, train_y, i))
        test_loss.append(adaBoost.partial_loss(test_X, test_y, i))
    fig1 = go.Figure([
        go.Scatter(x=np.arange(1, n_learners), y=train_loss, mode='lines', name=r'Train Loss'),
        go.Scatter(x=np.arange(1, n_learners), y=test_loss, mode='lines', name=r'Test Loss')])
    fig1.update_layout(title="Train and Test loss as function of #Base estimators", height=500,
                       xaxis=dict(title="Number Classifiers"), yaxis=dict(title="Loss"))
    fig1.show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    model_names = [str(t) + 'classifiers' for t in T]
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                         horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        f = lambda data: adaBoost.partial_predict(data, t)
        fig2.add_traces([decision_surface(f, lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(title="Fitted ensemble predictions up to number of iterations",
                       margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    min_error_t = np.argmin(np.array(test_loss)) + 1
    acc = np.sum(adaBoost.partial_predict(test_X,min_error_t) == test_y) / test_y.shape[0]
    fig3 = go.Figure()
    fig3.add_traces(
        [decision_surface(lambda data: adaBoost.partial_predict(data, min_error_t), lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
    fig3.update_layout(title="Best ensemble size predict " + str(min_error_t) + " Accuracy " + str(acc),
                       margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig3.show()
    # Question 4: Decision surface with weighted samples
    adaBoost.D_ = adaBoost.D_ / np.max(adaBoost.D_) * 7
    fig4 = go.Figure().add_traces(
        [decision_surface(lambda data: adaBoost.predict(data), lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=train_y, size=adaBoost.D_, colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
    fig4.update_layout(title="Train set with size according to weights", margin=dict(t=100),
                       xaxis_range=[np.floor(np.min(train_X[:, 0])), np.round(np.max(train_X[:, 0]))],
                       yaxis_range=[np.floor(np.min(train_X[:, 1])), np.round(np.max(train_X[:, 1]))])

    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
