from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def train_models(data):
    # Classification
    X_class = data.drop(columns=["Approved", "Credit_Limit"])
    y_class = data["Approved"]
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_class, y_train_class)
    y_pred_class = clf.predict(X_test_class)
    classification_accuracy = accuracy_score(y_test_class, y_pred_class)

    # Regression
    approved_data = data[data["Approved"] == 1]
    X_reg = approved_data.drop(columns=["Approved", "Credit_Limit"])
    y_reg = approved_data["Credit_Limit"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg)
    regression_rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)

    metrics = {
        "classification_accuracy": classification_accuracy,
        "regression_rmse": regression_rmse
    }

    return clf, reg, X_class, X_reg, metrics

def predict_credit_limit(clf, reg, input_data):
    approved = clf.predict(input_data)[0]
    credit_limit = reg.predict(input_data)[0] if approved else None
    return approved, credit_limit

