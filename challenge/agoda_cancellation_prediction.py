from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from datetime import datetime as dt
import plotly.graph_objects as go

import numpy as np
import pandas as pd

SATURDAY = 5


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    full_data = full_data.drop(
        full_data[full_data["cancellation_policy_code"] == "UNKNOWN"].index)
    full_data["cancellation_datetime"].fillna(0, inplace=True)
    return full_data.dropna()


def str_to_time(x):
    return dt.strptime(x, r"%Y-%m-%d %H:%M:%S")


def preprocess(full_data: pd.DataFrame):
    full_data.dropna(inplace=True)
    features = full_data[["h_booking_id",
                          "hotel_star_rating",
                          "guest_is_not_the_customer",
                          "no_of_adults",
                          "no_of_children",
                          "no_of_extra_bed",
                          "no_of_room",
                          "original_selling_amount",
                          "is_user_logged_in",
                          "is_first_booking",
                          "request_nonesmoke",
                          "request_latecheckin",
                          "request_highfloor",
                          "request_largebed",
                          "request_twinbeds",
                          "request_airport",
                          "request_earlycheckin"]].copy()

    to_days = lambda x: x.days
    booking_date = full_data["booking_datetime"].apply(str_to_time)
    checkin_date = full_data["checkin_date"].apply(str_to_time)
    checkout_date = full_data["checkout_date"].apply(str_to_time)
    features["hotel_live_time"] = (pd.Timestamp.now() - pd.to_datetime(
        full_data.hotel_live_date)).dt.days
    features["booking_checkin_difference"] = (
            checkin_date - booking_date).apply(to_days)
    features["length_of_stay"] = (checkout_date - checkin_date).apply(to_days)
    arrival_day = checkin_date.apply(lambda x: x.weekday())
    features["stay_over_weekend"] = (features["length_of_stay"] > 6) | (
            (arrival_day <= SATURDAY) & (SATURDAY <= (arrival_day + features[
        "length_of_stay"])))
    features = pd.concat([features,
                          pd.get_dummies(full_data.accommadation_type_name,
                                         drop_first=True),
                          pd.get_dummies(full_data.charge_option,
                                         drop_first=True)], axis=1)
    features["has_cancellation_history"] = df["h_customer_id"].apply(
        number_of_times_cancelled)
    def current_policy(days_from_checkin, length_of_stay, penalty_code):
        penalties = []
        for penalty in penalty_code.split("_"):
            if "D" not in penalty:
                continue
            penalty_days, penalty_calculation = penalty.split("D")
            if penalty_calculation[-1] == "N":
                percentage = int(penalty_calculation[:-1]) / length_of_stay
            else:
                percentage = float(penalty_calculation[:-1])
            penalties.append((float(penalty_days), percentage))
        penalties.sort(key=lambda x: x[0], reverse=True)
        current_penalty = 0
        for days, penalty in penalties:
            if days < days_from_checkin:
                break
            current_penalty = penalty
        return current_penalty

    features["cancellation_policy_at_time_of_order"] = pd.concat(
        [features[["booking_checkin_difference", "length_of_stay"]],
         full_data["cancellation_policy_code"]], axis=1).apply(
        lambda x: current_policy(x["booking_checkin_difference"],
                                 x["length_of_stay"],
                                 x["cancellation_policy_code"]), axis=1)
    cancellation_window_start_diff = features.booking_checkin_difference - 7
    cancellation_window_start_diff.name = "cancellation_window_start"
    features[
        "cancellation_policy_at_start_of_cancellation_window"] = pd.concat(
        [cancellation_window_start_diff, features["length_of_stay"],
         full_data["cancellation_policy_code"]], axis=1).apply(
        lambda x: current_policy(x["cancellation_window_start"],
                                 x["length_of_stay"],
                                 x["cancellation_policy_code"]), axis=1)
    cancellation_window_end_diff = features.booking_checkin_difference - 35
    cancellation_window_end_diff.name = "cancellation_window_end"
    features[
        "cancellation_policy_at_end_of_cancellation_window"] = pd.concat(
        [cancellation_window_end_diff, features["length_of_stay"],
         full_data["cancellation_policy_code"]], axis=1).apply(
        lambda x: current_policy(x["cancellation_window_end"],
                                 x["length_of_stay"],
                                 x["cancellation_policy_code"]), axis=1)
    features["cancellation_ploicy_change_during_window"] = \
        features.cancellation_policy_at_end_of_cancellation_window - \
        features.cancellation_policy_at_start_of_cancellation_window
    return features.dropna()


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(
        filename, index=False)


def preprocess_labels(cancellation_date: pd.Series,
                      booking_datetime: pd.Series):
    def str_to_date(x):
        return dt.strptime(x, r"%Y-%m-%d")

    cancellation = cancellation_date.apply(
        lambda x: dt.now() if x == 0 else str_to_date(x))
    booking = booking_datetime.apply(str_to_time)
    diff = (pd.to_datetime(cancellation, unit="s") - pd.to_datetime(booking,
                                                                    unit="s")).dt.days
    return (diff >= 7) & (diff < 35)

number_of_times_customer_canceled = dict()

def number_of_times_cancelled(id):
    if id in number_of_times_customer_canceled:
        return number_of_times_customer_canceled[id]
    return 0


if __name__ == '__main__':
    np.random.seed(0)

    # Load data

    df = load_data(
        "../datasets/agoda_cancellation_train.csv")

    design_matrix = preprocess(df)


    for id, cancellation in df[
        ["h_customer_id", "cancellation_datetime"]].itertuples(index=False):
        if cancellation == 0:
            if id in number_of_times_customer_canceled:
                number_of_times_customer_canceled[id] += 1
            else:
                number_of_times_customer_canceled[id] = 1

    cancellation_labels = preprocess_labels(df.cancellation_datetime,
                                            df.booking_datetime)
    # Fit model over data
    model = AgodaCancellationEstimator().fit(design_matrix, cancellation_labels)
    print(pd.read_csv("test_set_week_1.csv").shape)
    test_set = preprocess(pd.read_csv("test_set_week_1.csv"))

    # add categories with 0 - not showibg in train
    missing_cols = set(design_matrix.columns) - set(test_set.columns)
    for c in missing_cols:
        test_set[c] = 0
    test_set = test_set[design_matrix.columns]
    print(test_set.shape)
    #TODO: 1 - model gives output between [0,1] insteaed T/F
    #TODO: 2 - test set holds only 35 samples after preproccessing
    # Store model predictions over test set
    evaluate_and_export(model, test_set, "319091385_314618794_318839610.csv")
    print("finished")
