import numpy as np
from sympy import symbols, diff, simplify
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def run_symbolic_probe(time, orbital_radius, angular_momentum):
    # Polynomial regression for orbital_radius vs time
    degree = 3
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(time.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, orbital_radius)
    coefs = model.coef_
    intercept = model.intercept_
    t = symbols('t')
    r_expr = intercept
    for i in range(1, len(coefs)):
        r_expr += coefs[i] * t**i

    # Polynomial regression for angular_momentum vs time
    X_poly_L = poly.fit_transform(time.reshape(-1, 1))
    model_L = LinearRegression()
    model_L.fit(X_poly_L, angular_momentum)
    coefs_L = model_L.coef_
    intercept_L = model_L.intercept_
    L_expr = intercept_L
    for i in range(1, len(coefs_L)):
        L_expr += coefs_L[i] * t**i

    dr_dt = simplify(diff(r_expr, t))
    dL_dt = simplify(diff(L_expr, t))
    predicted = model.predict(X_poly)
    residuals = list(orbital_radius - predicted)
    time_scaled = (time - np.mean(time)) / np.std(time)
    time_scaled = [[v] for v in time_scaled]

    return {
        "r_expr": str(simplify(r_expr)),
        "L_expr": str(simplify(L_expr)),
        "dr_dt": str(dr_dt),
        "dL_dt": str(dL_dt),
        "transition_times": [],
        "residuals": residuals,
        "time_scaled": time_scaled,
        "time": list(time)
    }

# --- PySR symbolic regression (no pow operator, fixed param name) ---
def run_symbolic_regression(X, y):
    try:
        from pysr import PySRRegressor
    except ImportError:
        return "PySR not installed"

    model = PySRRegressor(
        model_selection="best",
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        maxsize=15,
        # loss_function="auto",  # Optional, you can omit for default squared loss
        verbosity=0
    )
    model.fit(np.array(X), np.array(y))
    return str(model.get_best())

