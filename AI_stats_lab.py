import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    X, y, true_coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    return X, y, true_coef


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    X_out = X.copy()
    y_out = y.copy()
    random_normal_values = rng.randn(n_outliers)
    X_out[:n_outliers] = (10 + 0.75 * random_normal_values).reshape(-1, 1)
    y_out[:n_outliers] = -15 + 20 * random_normal_values
    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    fig, ax = plt.subplots()
    ax.scatter(X[n_outliers:], y[n_outliers:], label="Normal Data", alpha=0.6)
    ax.scatter(X[:n_outliers], y[:n_outliers], label="Artificial Outliers", marker="x", c="red")
    ax.set_title("Dataset with Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_huber_regression(X, y):
    model = HuberRegressor()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    model = TheilSenRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    return {name: abs(coef - true_coef) for name, coef in coef_dict.items()}


def best_robust_model(errors):
    robust_keys = ["huber_regression", "ransac_regression", "theilsen_regression"]
    robust_errors = {k: errors[k] for k in robust_keys}
    return min(robust_errors, key=robust_errors.get)


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    outlier_mask = ~model.inlier_mask_
    total_outliers_detected = int(np.sum(outlier_mask))
    added_outliers_detected = int(np.sum(outlier_mask[:n_outliers]))
    return total_outliers_detected, added_outliers_detected


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5, label="Data")

    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    lr_coef = fit_linear_regression(X, y)
    ax.plot(x_line, lr_coef * x_line, label="Linear Regression")

    huber_coef = fit_huber_regression(X, y)
    ax.plot(x_line, huber_coef * x_line, label="Huber Regression")

    ransac_coef = fit_ransac_regression(X, y, random_state=random_state)
    ransac = RANSACRegressor(random_state=random_state)
    ransac.fit(X, y)
    ransac_intercept = float(ransac.estimator_.intercept_)
    ax.plot(x_line, ransac_coef * x_line + ransac_intercept, label="RANSAC Regression")

    theilsen_coef = fit_theilsen_regression(X, y, random_state=random_state)
    ax.plot(x_line, theilsen_coef * x_line, label="Theil-Sen Regression")

    ax.set_title("Regression Fits Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots()
    ax.scatter(X[inlier_mask], y[inlier_mask], label="Inliers", alpha=0.6)
    ax.scatter(X[outlier_mask], y[outlier_mask], label="Outliers", marker="x", c="red")
    ax.set_title("RANSAC Inliers vs Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    return fig
