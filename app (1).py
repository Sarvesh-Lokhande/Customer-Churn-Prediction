# coding: utf-8
"""
ChurnPredict
Customer churn prediction, SHAP explanations, customer editing,
dataset analysis, and retention recommendations.
"""

import os
import pickle
import warnings
import traceback
import io

import numpy as np
import pandas as pd

from flask import Flask, request, render_template, redirect, url_for, flash, send_file

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MERGED_CSV_PATH = os.path.join(BASE_DIR, "first_telc_merged.csv")
CSV_PATH = MERGED_CSV_PATH if os.path.exists(MERGED_CSV_PATH) else os.path.join(BASE_DIR, "first_telc.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.sav")

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-churn-prediction-demo-key"
)

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

CUSTOMER_FIELDS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["tenure"]

CATEGORY_FALLBACKS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

# ============================================================
# 2. LOAD DATASET AND MODEL
# ============================================================

df = None
model = None
MODEL_FEATURES = None
load_errors = []
latest_batch_csv = None  # Local-demo export buffer; latest upload only.

try:
    df = pd.read_csv(CSV_PATH)
    df.columns = [str(column).strip() for column in df.columns]
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    print(f"[INFO] Dataset loaded: {df.shape}")
except Exception as exc:
    load_errors.append(f"Could not load first_telc.csv: {exc}")
    print("[ERROR] Dataset loading failed:", exc)

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    MODEL_FEATURES = list(getattr(model, "feature_names_in_", []))

    if not MODEL_FEATURES:
        MODEL_FEATURES = None
        load_errors.append(
            "The model does not expose feature_names_in_. "
            "The expected training feature order must be recovered."
        )

    print("[INFO] Model loaded:", type(model).__name__)
    if MODEL_FEATURES is not None:
        print("[INFO] Expected model features:", len(MODEL_FEATURES))
except Exception as exc:
    load_errors.append(f"Could not load model.sav: {exc}")
    print("[ERROR] Model loading failed:", exc)


# ============================================================
# 3. DATASET HELPERS
# ============================================================

def target_column():
    """Find the column containing the actual churn labels."""
    if df is None:
        return None
    for column in ["Churn", "churn", "Exited", "CustomerStatus"]:
        if column in df.columns:
            return column
    return None


def customer_id_column():
    """Find a customer ID column, if the dataset has one."""
    if df is None:
        return None
    for column in ["customerID", "CustomerID", "customer_id", "ID", "id"]:
        if column in df.columns:
            return column
    return None


def dataset_stats():
    """Summarize the loaded dataset without treating missing labels as retained."""
    if df is None:
        return {"rows": 0, "columns": 0, "churn_count": None,
                "retained_count": None, "target": None, "churn_rate": None,
                "contract_breakdown": []}

    target = target_column()
    churn_count = retained_count = churn_rate = None
    contract_breakdown = []
    if target:
        labels = df[target].astype(str).str.strip().str.lower()
        valid = df[target].notna()
        churn_labels = ["yes", "1", "true", "churn", "churned", "exited"]
        nonchurn_labels = ["no", "0", "false", "retained", "active"]
        churn_count = int((valid & labels.isin(churn_labels)).sum())
        retained_count = int((valid & labels.isin(nonchurn_labels)).sum())
        labeled_count = churn_count + retained_count
        churn_rate = round(100 * churn_count / labeled_count, 1) if labeled_count else None
        if "Contract" in df.columns:
            temp = df.loc[valid, ["Contract", target]].copy()
            temp["_is_churn"] = temp[target].astype(str).str.strip().str.lower().isin(churn_labels)
            for contract, group in temp.groupby("Contract", dropna=False):
                total = len(group)
                rate = round(100 * int(group["_is_churn"].sum()) / total, 1) if total else 0
                contract_breakdown.append({"contract": str(contract), "count": total, "rate": rate})
            contract_breakdown.sort(key=lambda x: x["rate"], reverse=True)

    return {"rows": len(df), "columns": len(df.columns), "churn_count": churn_count,
            "retained_count": retained_count, "target": target, "churn_rate": churn_rate,
            "contract_breakdown": contract_breakdown}


def allowed_values(column, fallback=None):
    """Get category options from the CSV."""
    fallback = fallback or CATEGORY_FALLBACKS.get(column, [])
    if df is not None and column in df.columns:
        values = df[column].dropna().astype(str).unique().tolist()
        if values:
            return sorted(values)
    return fallback


def all_category_options():
    """Return all category lists required by the HTML template."""
    return {
        column: allowed_values(column, fallback)
        for column, fallback in CATEGORY_FALLBACKS.items()
    }


def dataset_dropdown():
    """Prepare the customer selector shown in the dashboard."""
    if df is None:
        return []

    id_column = customer_id_column()
    customers = []

    for index in range(min(len(df), 2000)):
        label = str(df.iloc[index][id_column]) if id_column else f"Dataset row {index + 1}"
        customers.append({"index": index, "label": label})

    return customers


# ============================================================
# 4. DEFAULT CUSTOMER AND FORM VALUES
# ============================================================

def default_customer():
    """Build default values, preferring representative CSV values."""
    defaults = {
        "SeniorCitizen": 0,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 12
    }

    if df is None:
        return defaults

    for column in CUSTOMER_FIELDS:
        if column not in df.columns:
            continue

        series = df[column].dropna()
        if series.empty:
            continue

        if column in NUMERIC_FEATURES or column == "tenure":
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric_series.empty:
                median_value = float(numeric_series.median())
                defaults[column] = (
                    int(median_value)
                    if column in ["SeniorCitizen", "tenure"]
                    else median_value
                )
        else:
            modes = series.astype(str).mode()
            if not modes.empty:
                defaults[column] = str(modes.iloc[0])

    defaults["tenure"] = max(1, min(72, int(defaults["tenure"])))
    return defaults


def form_values(customer):
    """Convert customer fields into the query1...query19 form format."""
    field_order = [
        "SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender",
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "tenure"
    ]
    return {
        f"query{index}": customer.get(column, "")
        for index, column in enumerate(field_order, start=1)
    }


def customer_from_form(form):
    """Read and validate values submitted through the customer form."""
    field_order = [
        "SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender",
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "tenure"
    ]

    customer = {}
    for index, column in enumerate(field_order, start=1):
        value = form.get(f"query{index}", "").strip()
        if value == "":
            raise ValueError(f"Please provide {column}.")
        customer[column] = value

    return normalize_customer(customer)


def normalize_customer(data):
    """Convert values to the correct types and validate the customer."""
    customer = {}

    for column in NUMERIC_FEATURES:
        try:
            customer[column] = float(data[column])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"{column} must be a valid number.")

    customer["SeniorCitizen"] = int(customer["SeniorCitizen"])

    try:
        customer["tenure"] = int(float(data["tenure"]))
    except (TypeError, ValueError, KeyError):
        raise ValueError("Tenure must be a whole number of months.")

    for column in CATEGORICAL_FEATURES:
        if column not in data:
            raise ValueError(f"Missing customer field: {column}")
        customer[column] = str(data[column]).strip()

    if customer["SeniorCitizen"] not in [0, 1]:
        raise ValueError("Senior citizen must be 0 (No) or 1 (Yes).")

    for column in ["MonthlyCharges", "TotalCharges"]:
        value = customer[column]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{column} must be a non-negative finite number.")

    if not 1 <= customer["tenure"] <= 72:
        raise ValueError("Tenure must be between 1 and 72 months.")

    if df is not None:
        for column in CATEGORICAL_FEATURES:
            if column not in df.columns:
                continue
            options = df[column].dropna().astype(str).unique().tolist()
            if options and customer[column] not in options:
                raise ValueError(
                    f"Invalid value for {column}: {customer[column]!r}. "
                    f"Available values: {options}"
                )

    return customer


# ============================================================
# 5. TENURE ENCODING
# ============================================================

def tenure_group(value):
    """Map tenure to the six groups used by the trained model."""
    value = int(float(value))
    if not 1 <= value <= 72:
        raise ValueError("Tenure must be between 1 and 72 months.")
    upper = min(((value - 1) // 12 + 1) * 12, 72)
    lower = upper - 11
    return f"{lower} - {upper}"


# ============================================================
# 6. PREPARE THE MODEL INPUT
# ============================================================

def prepare_model_input(customer):
    """Build a DataFrame with the exact feature names and order in model.sav."""
    if model is None:
        raise RuntimeError("The trained model could not be loaded.")
    if MODEL_FEATURES is None:
        raise RuntimeError(
            "The model does not contain feature_names_in_. "
            "Cannot safely construct the expected model input."
        )

    expected_count = getattr(model, "n_features_in_", len(MODEL_FEATURES))
    if len(MODEL_FEATURES) != expected_count:
        raise RuntimeError(
            f"Model feature mismatch: model expects {expected_count}, "
            f"but feature_names_in_ contains {len(MODEL_FEATURES)}."
        )

    X = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)

    for column in NUMERIC_FEATURES:
        if column not in X.columns:
            raise ValueError(f"Required numeric model feature is missing: {column}")
        X.loc[0, column] = float(customer[column])

    for column in CATEGORICAL_FEATURES:
        encoded_column = f"{column}_{customer[column]}"
        if encoded_column not in X.columns:
            available = [
                feature for feature in MODEL_FEATURES
                if feature.startswith(column + "_")
            ]
            raise ValueError(
                f"Category {customer[column]!r} for {column} is not supported "
                f"by this model. Available encoded features: {available}"
            )
        X.loc[0, encoded_column] = 1.0

    group_name = tenure_group(customer["tenure"])
    tenure_feature = f"tenure_group_{group_name}"

    if tenure_feature not in X.columns:
        available = [
            feature for feature in MODEL_FEATURES
            if feature.startswith("tenure_group_")
        ]
        raise ValueError(
            f"Tenure group {tenure_feature!r} is not available. "
            f"Available tenure features: {available}"
        )

    X.loc[0, tenure_feature] = 1.0
    X = X[MODEL_FEATURES]

    if X.shape[1] != expected_count:
        raise ValueError(
            f"Model expects {expected_count} features, but received {X.shape[1]}."
        )
    if X.isnull().any().any():
        raise ValueError("Prepared model input contains missing values.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Prepared model input contains invalid numeric values.")

    return X


# ============================================================
# 7. PREDICTION HELPERS
# ============================================================

def is_churn_label(label):
    """Recognize common positive churn labels."""
    return str(label).strip().lower() in [
        "1", "yes", "true", "churn", "churned"
    ]


def churn_class_index():
    """Find the probability column corresponding to the churn class."""
    classes = list(getattr(model, "classes_", []))
    for index, label in enumerate(classes):
        if is_churn_label(label):
            return index
    if len(classes) == 2:
        return 1
    return 0


def churn_probability(X):
    """Return the estimated probability of the positive churn class."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)[0]
    class_index = churn_class_index()
    if class_index >= len(probabilities):
        return None

    return float(probabilities[class_index]) * 100.0


# ============================================================
# 8. SHAP EXPLANATION WITH GROUPED CATEGORICAL FEATURES
# ============================================================

def human_feature(name):
    """Make encoded model feature names easier to read."""
    if name.startswith("tenure_group_"):
        group = name.replace("tenure_group_", "")
        return f"Tenure: {group} months"

    for column in CATEGORICAL_FEATURES:
        prefix = column + "_"
        if name.startswith(prefix):
            category = name[len(prefix):]
            return f"{column} = {category}"

    return name.replace("_", " ").strip().title()


def shap_explanation(X, customer=None):
    """
    Calculate SHAP contributions for the churn class and group
    one-hot-encoded columns belonging to the same categorical field.
    """
    try:
        import shap

        if model is None:
            raise RuntimeError("The trained model is unavailable.")

        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(X)
        class_index = churn_class_index()

        if isinstance(raw_values, list):
            if not raw_values:
                raise ValueError("SHAP returned no class explanations.")
            selected_index = min(class_index, len(raw_values) - 1)
            contributions = np.asarray(raw_values[selected_index]).reshape(-1)
        else:
            values_array = np.asarray(raw_values)

            if values_array.ndim == 3:
                selected_index = min(class_index, values_array.shape[2] - 1)
                contributions = values_array[0, :, selected_index]
            elif values_array.ndim == 2:
                contributions = values_array[0, :]
            elif values_array.ndim == 1:
                contributions = values_array
            else:
                raise ValueError(
                    f"Unsupported SHAP output shape: {values_array.shape}"
                )

        contributions = np.asarray(contributions, dtype=float).reshape(-1)

        if len(contributions) != len(X.columns):
            raise ValueError(
                "SHAP contribution count does not match the model input feature count."
            )
        if not np.isfinite(contributions).all():
            raise ValueError("SHAP returned non-finite contribution values.")

        grouped_values = {}
        grouped_labels = {}

        for feature, contribution in zip(X.columns, contributions):
            contribution = float(contribution)
            group_name = feature
            display_label = human_feature(feature)

            for column in CATEGORICAL_FEATURES:
                if feature.startswith(column + "_"):
                    group_name = column
                    if customer is not None:
                        selected_category = str(customer.get(column, ""))
                        display_label = (
                            f"{column} (selected: {selected_category})"
                        )
                    else:
                        display_label = column
                    break

            grouped_values[group_name] = (
                grouped_values.get(group_name, 0.0) + contribution
            )
            grouped_labels[group_name] = display_label

        sorted_groups = sorted(
            grouped_values.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        explanation = []
        for group_name, contribution in sorted_groups[:8]:
            contribution = float(contribution)
            explanation.append({
                "feature": grouped_labels[group_name],
                "raw_feature": group_name,
                "value": contribution,
                "direction": "increases" if contribution > 0 else "decreases",
                "sign": "+" if contribution > 0 else "−"
            })

        note = (
            "Related one-hot-encoded categorical features are grouped into "
            "one explanation. For example, the Contract columns are combined "
            "into a single Contract contribution. Positive values move the "
            "explained model output toward churn; negative values move it away "
            "from churn. Values are model-output contributions, not percentages, "
            "and do not establish causation."
        )

        return explanation, note

    except ImportError:
        return (
            [],
            "SHAP is not installed in the active Python environment. "
            "Activate .venv and install a compatible SHAP version."
        )
    except Exception as exc:
        print("[SHAP ERROR]", repr(exc))
        traceback.print_exc()
        return (
            [],
            f"SHAP could not explain this prediction ({exc}). "
            "The churn prediction can still be used."
        )


# ============================================================
# 9. CUSTOMER RETENTION RECOMMENDATIONS
# ============================================================

def recommendations(customer, is_churn, probability, contributions):
    """Generate practical customer-care suggestions."""
    suggestions = []

    positive_features = [
        item for item in contributions if item["value"] > 0
    ]

    for item in positive_features:
        feature = item["raw_feature"]

        if feature == "Contract":
            if customer["Contract"] == "Month-to-month":
                suggestions.append(
                    "Explain available contract options, including pricing, "
                    "flexibility, and any early-exit terms."
                )
            else:
                suggestions.append(
                    "Review whether the customer's current contract continues "
                    "to meet their needs."
                )

        elif feature in ["TechSupport", "OnlineSecurity"]:
            if customer[feature] == "No":
                suggestions.append(
                    f"Ask whether the customer needs {feature} services or "
                    "assistance with existing service issues."
                )

        elif feature == "PaymentMethod":
            if customer["PaymentMethod"] == "Electronic check":
                suggestions.append(
                    "Offer to review payment methods and explain available "
                    "convenient billing options."
                )

        elif feature == "MonthlyCharges":
            suggestions.append(
                "Review the monthly bill and discuss lower-cost plans if they "
                "better match the customer's needs."
            )

        elif feature.startswith("tenure_group_"):
            suggestions.append(
                "Consider a customer check-in appropriate to their stage in "
                "the relationship with the service."
            )

    if customer["Contract"] == "Month-to-month":
        suggestions.append(
            "Discuss longer-term plans only if they suit the customer's needs, "
            "and explain the terms transparently."
        )

    if customer["tenure"] <= 12:
        suggestions.append(
            "Arrange an onboarding check-in to identify setup, billing, or "
            "service issues."
        )

    if customer["MonthlyCharges"] >= 80:
        suggestions.append(
            "Review the monthly bill for unexpected charges and discuss "
            "suitable plan alternatives."
        )

    if (
        customer["TechSupport"] == "No"
        and customer["InternetService"] != "No"
    ):
        suggestions.append(
            "Ask whether the customer has experienced technical issues and "
            "offer support where needed."
        )

    if not suggestions:
        suggestions.append(
            "Contact the customer to understand their experience before "
            "deciding whether a retention offer is appropriate."
        )

    return list(dict.fromkeys(suggestions))[:5]


# ============================================================
# 10. ANALYZE A CUSTOMER
# ============================================================

def analyze_customer(customer):
    """Validate, encode, predict, explain, and recommend."""
    customer = normalize_customer(customer)
    X = prepare_model_input(customer)

    prediction = model.predict(X)[0]
    probability = churn_probability(X)
    churn = is_churn_label(prediction)

    # Pass customer details so grouped SHAP labels show the selected category.
    contributions, shap_note = shap_explanation(X, customer=customer)

    recs = recommendations(customer, churn, probability, contributions)

    return {
        "customer": customer,
        "prediction": "Higher churn risk" if churn else "Lower churn risk",
        "is_churn": churn,
        "probability": probability,
        "contributions": contributions,
        "shap_note": shap_note,
        "recommendations": recs,
        "feature_count": X.shape[1]
    }


# ============================================================
# 11. SHARED TEMPLATE CONTEXT
# ============================================================

def render_dashboard(
    values=None,
    result=None,
    selected_dataset="",
    actual_label=None,
    dataset_results=None,
    analyzed_count=None,
    batch_results=None,
    batch_summary=None,
    evaluation=None
):
    """Supply all variables expected by home.html."""
    if values is None:
        values = form_values(default_customer())

    context = {
        "values": values,
        "stats": dataset_stats(),
        "category_options": all_category_options(),
        "dataset_customers": dataset_dropdown(),
        "load_errors": load_errors,
        "result": result,
        "selected_dataset": selected_dataset,
        "actual_label": actual_label
    }

    if dataset_results is not None:
        context["dataset_results"] = dataset_results
    if analyzed_count is not None:
        context["analyzed_count"] = analyzed_count
    if batch_results is not None:
        context["batch_results"] = batch_results
    if batch_summary is not None:
        context["batch_summary"] = batch_summary
    if evaluation is not None:
        context["evaluation"] = evaluation

    return render_template("home.html", **context)


# ============================================================
# 12. HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_dashboard()


# ============================================================
# 13. INDIVIDUAL CUSTOMER PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if df is None:
            raise RuntimeError("Dataset unavailable. Check first_telc.csv.")
        if model is None:
            raise RuntimeError("Model unavailable. Check model.sav.")

        customer = customer_from_form(request.form)
        result = analyze_customer(customer)
        selected_dataset = request.form.get("dataset_index", "")

        flash("Customer analysis completed.", "success")

        return render_dashboard(
            values=form_values(customer),
            result=result,
            selected_dataset=selected_dataset
        )

    except Exception as exc:
        print("[PREDICTION ERROR]", repr(exc))
        traceback.print_exc()
        flash(f"Prediction could not be completed: {exc}", "error")

        try:
            values = {
                f"query{i}": request.form.get(f"query{i}", "")
                for i in range(1, 20)
            }
            return render_dashboard(
                values=values,
                selected_dataset=request.form.get("dataset_index", "")
            )
        except Exception:
            return redirect(url_for("home"))


# ============================================================
# 14. LOAD CUSTOMER FROM CSV
# ============================================================

@app.route("/load-customer", methods=["POST"])
def load_customer():
    try:
        if df is None:
            raise RuntimeError("Dataset is not available.")

        index_value = request.form.get("dataset_index", "").strip()
        if not index_value:
            raise ValueError("Please select a customer from the CSV.")

        index = int(index_value)
        if index < 0 or index >= len(df):
            raise ValueError("The selected dataset row is invalid.")

        missing_columns = [
            column for column in CUSTOMER_FIELDS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(missing_columns)
            )

        row = df.iloc[index]
        customer = {column: row[column] for column in CUSTOMER_FIELDS}
        customer = normalize_customer(customer)

        actual_label = None
        target = target_column()
        if target:
            actual_label = str(row[target])

        return render_dashboard(
            values=form_values(customer),
            selected_dataset=str(index),
            actual_label=actual_label
        )

    except Exception as exc:
        print("[LOAD CUSTOMER ERROR]", repr(exc))
        flash(f"Could not load customer: {exc}", "error")
        return redirect(url_for("home"))


# ============================================================
# 15. BULK DATASET ANALYSIS
# ============================================================

@app.route("/analyze-dataset", methods=["POST"])
def analyze_dataset():
    try:
        if df is None:
            raise RuntimeError("Dataset is not available.")
        if model is None:
            raise RuntimeError("Trained model is not available.")

        try:
            limit = int(request.form.get("limit", "100"))
        except ValueError:
            limit = 100

        limit = max(1, min(limit, 1000))
        count = min(len(df), limit)

        id_column = customer_id_column()
        target = target_column()
        records = []

        missing_columns = [
            column for column in CUSTOMER_FIELDS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(missing_columns)
            )

        for index in range(count):
            row = df.iloc[index]
            customer = {column: row[column] for column in CUSTOMER_FIELDS}

            customer_id = (
                str(row[id_column]) if id_column else f"Row {index + 1}"
            )
            actual = str(row[target]) if target else "Not available"

            try:
                result = analyze_customer(customer)
                records.append({
                    "index": index,
                    "id": customer_id,
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "actual": actual
                })
            except Exception as row_exc:
                records.append({
                    "index": index,
                    "id": customer_id,
                    "prediction": "Could not analyze",
                    "probability": None,
                    "actual": str(row_exc)
                })

        flash(f"Finished analyzing {count} dataset rows.", "success")

        return render_dashboard(
            values=form_values(default_customer()),
            dataset_results=records,
            analyzed_count=count
        )

    except Exception as exc:
        print("[BULK ANALYSIS ERROR]", repr(exc))
        traceback.print_exc()
        flash(f"Dataset analysis failed: {exc}", "error")
        return redirect(url_for("home"))


def predict_customer_fast(customer):
    """Predict without SHAP; bulk jobs should not calculate explanations per row."""
    customer = normalize_customer(customer)
    X = prepare_model_input(customer)
    prediction = model.predict(X)[0]
    probability = churn_probability(X)
    is_churn = is_churn_label(prediction)
    return {
        "prediction": "Higher churn risk" if is_churn else "Lower churn risk",
        "is_churn": is_churn,
        "probability": probability,
        "customer": customer
    }


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Predict churn for a user-supplied CSV and prepare a downloadable report."""
    global latest_batch_csv
    try:
        if model is None:
            raise RuntimeError("Model unavailable. Keep model.sav beside this file.")
        upload = request.files.get("customer_file")
        if upload is None or not upload.filename:
            raise ValueError("Choose a CSV file first.")
        if not upload.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a .csv file.")

        uploaded_df = pd.read_csv(upload, nrows=1001)
        uploaded_df.columns = [str(c).strip() for c in uploaded_df.columns]
        if uploaded_df.empty:
            raise ValueError("The uploaded CSV has no customer rows.")
        if len(uploaded_df) > 1000:
            raise ValueError("The upload contains more than 1,000 rows. Split it into smaller files.")
        missing = [c for c in CUSTOMER_FIELDS if c not in uploaded_df.columns]
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))

        id_col = next((c for c in ["customerID", "CustomerID", "customer_id", "ID", "id"] if c in uploaded_df.columns), None)
        target = next((c for c in ["Churn", "churn", "Exited", "CustomerStatus"] if c in uploaded_df.columns), None)
        records, export_rows = [], []
        exposure = 0.0
        high_count = medium_count = low_count = failed_count = 0

        for row_idx, row in uploaded_df.iterrows():
            customer_id = str(row[id_col]) if id_col and pd.notna(row[id_col]) else f"Upload row {row_idx + 1}"
            actual = str(row[target]) if target and pd.notna(row[target]) else "Not supplied"
            try:
                customer = normalize_customer({field: row[field] for field in CUSTOMER_FIELDS})
                result = predict_customer_fast(customer)
                probability = result["probability"]
                weighted_exposure = (probability / 100.0) * customer["MonthlyCharges"] if probability is not None else None
                if weighted_exposure is not None:
                    exposure += weighted_exposure
                if probability is not None and probability >= 60:
                    risk = "High"
                    high_count += 1
                elif probability is not None and probability >= 30:
                    risk = "Medium"
                    medium_count += 1
                elif probability is not None:
                    risk = "Low"
                    low_count += 1
                else:
                    risk = "Unavailable"
                record = {
                    "id": customer_id, "prediction": result["prediction"], "probability": probability,
                    "risk": risk, "monthly_charges": customer["MonthlyCharges"],
                    "revenue_exposure": weighted_exposure, "actual": actual, "error": ""
                }
                records.append(record)
                export_rows.append({
                    "CustomerID": customer_id, "PredictedChurnClass": result["prediction"],
                    "ChurnProbabilityPercent": round(probability, 2) if probability is not None else "",
                    "RiskCategory": risk, "MonthlyCharges": customer["MonthlyCharges"],
                    "ProbabilityWeightedMonthlyRevenueExposure": round(weighted_exposure, 2) if weighted_exposure is not None else "",
                    "ActualChurnLabel": actual
                })
            except Exception as row_exc:
                failed_count += 1
                records.append({"id": customer_id, "prediction": "Could not analyze", "probability": None,
                    "risk": "Unavailable", "monthly_charges": None, "revenue_exposure": None,
                    "actual": actual, "error": str(row_exc)})
                export_rows.append({"CustomerID": customer_id, "PredictedChurnClass": "Could not analyze",
                    "ChurnProbabilityPercent": "", "RiskCategory": "Unavailable",
                    "MonthlyCharges": row.get("MonthlyCharges", ""),
                    "ProbabilityWeightedMonthlyRevenueExposure": "", "ActualChurnLabel": actual,
                    "Error": str(row_exc)})

        latest_batch_csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig")
        summary = {"total": len(uploaded_df), "high": high_count, "medium": medium_count,
                   "low": low_count, "failed": failed_count, "exposure": exposure}
        flash(f"Batch prediction completed for {len(uploaded_df)} rows; {failed_count} row(s) could not be analyzed.", "success")
        return render_dashboard(values=form_values(default_customer()), batch_results=records, batch_summary=summary)
    except Exception as exc:
        print("[BATCH UPLOAD ERROR]", repr(exc))
        traceback.print_exc()
        flash(f"Batch prediction failed: {exc}", "error")
        return redirect(url_for("home"))


@app.route("/download-batch", methods=["GET"])
def download_batch():
    if not latest_batch_csv:
        flash("Run a batch prediction before downloading results.", "error")
        return redirect(url_for("home"))
    return send_file(io.BytesIO(latest_batch_csv), mimetype="text/csv", as_attachment=True,
                     download_name="churn_predictions.csv")


@app.route("/evaluate-model", methods=["POST"])
def evaluate_model():
    """Evaluate labeled rows from first_telc.csv as a diagnostic."""
    try:
        if df is None or model is None:
            raise RuntimeError("Both first_telc.csv and model.sav must be available.")
        target = target_column()
        if not target:
            raise ValueError("No recognized churn target column was found in first_telc.csv.")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        y_true, y_pred = [], []
        skipped = 0
        for _, row in df.iterrows():
            if pd.isna(row[target]):
                skipped += 1
                continue
            try:
                customer = normalize_customer({field: row[field] for field in CUSTOMER_FIELDS})
                result = predict_customer_fast(customer)
                y_true.append(1 if is_churn_label(row[target]) else 0)
                y_pred.append(1 if result["is_churn"] else 0)
            except Exception:
                skipped += 1
        if not y_true:
            raise ValueError("No rows with usable customer fields and known churn labels could be evaluated.")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics = {"count": len(y_true), "skipped": skipped,
            "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
        flash("Evaluation completed. Treat these as a fair test score only if this CSV was not used to train the model.", "success")
        return render_dashboard(values=form_values(default_customer()), evaluation=metrics)
    except Exception as exc:
        print("[EVALUATION ERROR]", repr(exc))
        traceback.print_exc()
        flash(f"Model evaluation failed: {exc}", "error")
        return redirect(url_for("home"))


# ============================================================
# 16. START FLASK
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ChurnPredict is starting")
    print("Dashboard: http://127.0.0.1:6060")
    print("Dataset:", CSV_PATH)
    print("Model:", MODEL_PATH)

    if df is not None:
        print("Dataset shape:", df.shape)
    else:
        print("WARNING: Dataset not loaded.")

    if model is not None:
        print("Model type:", type(model).__name__)
        print("Model feature count:", getattr(model, "n_features_in_", "unknown"))
    else:
        print("WARNING: Model not loaded.")

    for message in load_errors:
        print("[STARTUP WARNING]", message)

    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=6060,
        debug=False,
        use_reloader=False
    )