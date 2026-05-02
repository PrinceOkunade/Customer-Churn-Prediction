"""Smoke tests for the inference core. These run in CI before deploy."""
import pytest

from inference import build_feature_vector, load_artifacts, predict


HIGH_RISK_PROFILE = {
    "gender": "Female", "SeniorCitizen": "Yes", "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
}

LOW_RISK_PROFILE = {
    "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 45.0, "TotalCharges": 2700.0,
}


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_load_artifacts_returns_four_objects(artifacts):
    model, scaler, feature_names, explainer = artifacts
    assert model is not None
    assert scaler is not None
    assert isinstance(feature_names, list) and len(feature_names) > 0
    assert explainer is not None


def test_feature_vector_shape_matches_training(artifacts):
    _, scaler, feature_names, _ = artifacts
    X = build_feature_vector(HIGH_RISK_PROFILE, scaler, feature_names)
    assert list(X.columns) == feature_names
    assert X.shape == (1, len(feature_names))


def test_predict_returns_expected_shape(artifacts):
    model, scaler, feature_names, explainer = artifacts
    result = predict(HIGH_RISK_PROFILE, model, scaler, feature_names, explainer)
    assert set(result.keys()) == {
        "churn_probability", "churn_prediction", "risk_level", "shap_top_features"
    }
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_prediction"] in (0, 1)
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert len(result["shap_top_features"]) == 5


def test_high_risk_profile_classifies_as_high(artifacts):
    model, scaler, feature_names, explainer = artifacts
    result = predict(HIGH_RISK_PROFILE, model, scaler, feature_names, explainer)
    assert result["risk_level"] == "HIGH", f"got {result}"
    assert result["churn_prediction"] == 1


def test_low_risk_profile_classifies_as_low(artifacts):
    model, scaler, feature_names, explainer = artifacts
    result = predict(LOW_RISK_PROFILE, model, scaler, feature_names, explainer)
    assert result["risk_level"] == "LOW", f"got {result}"
    assert result["churn_prediction"] == 0


def test_risk_levels_are_consistent_with_probability(artifacts):
    model, scaler, feature_names, explainer = artifacts
    for profile in (HIGH_RISK_PROFILE, LOW_RISK_PROFILE):
        r = predict(profile, model, scaler, feature_names, explainer)
        p = r["churn_probability"]
        if p < 0.30:
            assert r["risk_level"] == "LOW"
        elif p < 0.60:
            assert r["risk_level"] == "MEDIUM"
        else:
            assert r["risk_level"] == "HIGH"
