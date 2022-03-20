from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    N = 1000
    mu, sigma = 10, 1
    S = np.random.normal(mu, sigma, size=N)
    uniGaus = UnivariateGaussian()
    uniGaus = uniGaus.fit(S)
    print('(' + str(uniGaus.mu_) + ', ' + str(uniGaus.var_) + ')')

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, N, 100).astype(int)
    estimated_mean = []
    for m in ms:
        m_samples = S[:m]
        X = UnivariateGaussian()
        X.fit(m_samples)
        estimated_mean.append(np.abs(X.mu_ - mu))

    fig2 = go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')]
                     , layout=go.Layout(title=r"$\text{ Estimation of absolute distance from"
                                              r"expectation as function of number of samples}$",
                                        xaxis_title="$m\\text{ - Number of samples}$",
                                        yaxis_title="Absolute distance from expectation",
                                        height=500))
    fig2.update_layout(
        font={'family': 'Arial', 'size': 18, 'color': "RebeccaPurple"}
    )
    fig2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfAtSamples = uniGaus.pdf(S)

    fig3 = go.Figure([go.Scatter(x=S, y=pdfAtSamples, mode='markers', name='Empirical Pdf Values')]
                     , layout=go.Layout(title=r"$\text{ Empirical Pdf Function Under fitted Uni variate model}$",
                                        xaxis_title="$\\text{Samples}$",
                                        yaxis_title="$\\text{Pdf}$",
                                        showlegend=True,
                                        height=500))
    fig3.update_layout(
        font={'family': 'Arial', 'size': 18, 'color': "RebeccaPurple"}
    )
    fig3.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    N = 1000
    means = np.array([0, 0, 4, 0])
    covMat = np.array(
        [[1, 0.2, 0, 0.5],
         [0.2, 2, 0, 0],
         [0, 0, 1, 0],
         [0.5, 0, 0, 1]])

    S = np.random.multivariate_normal(means, covMat, N)
    X = MultivariateGaussian().fit(S)
    print("The estimated expectation is:\n",X.mu_)
    print("The estimated covariance matrix is:\n",X.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    likelyHoods = [(MultivariateGaussian.log_likelihood(np.array([i, 0, j, 0]), covMat, S)) for i in f1 for j in f3]
    likelyHoods = np.reshape(likelyHoods,(200,200))
    fig5 = go.Figure(data=go.Heatmap(x=f1, y=f3, z=likelyHoods.tolist()),
                    layout=go.Layout(title='Heatmap Represtentation of Log Liklyhood'))
    fig5.show()

    # Question 6 - Maximum likelihood
    j_max, i_max = np.unravel_index(np.argmax(likelyHoods), np.shape(likelyHoods))
    val1, val3 = f1[i_max], f3[j_max]
    print('The model achieved the maximum log liklyhood was given by (val1,val3): ')
    print(val1, val3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
