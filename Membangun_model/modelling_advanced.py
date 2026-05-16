import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dagshub.init(
    repo_owner="jekiwantaufik",
    repo_name="Eksperimen_SML_Jekiwantaufik",
    mlflow=True
)

data = pd.read_csv("preprocessing/telco-customer-churn_preprocessing.csv")

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with mlflow.start_run():

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    plt.close()

    importance = np.abs(model.coef_[0])
    feature_names = X.columns

    plt.figure(figsize=(10,6))

    sns.barplot(
        x=importance,
        y=feature_names
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("feature_importance.png")

    mlflow.log_artifact("feature_importance.png")

    plt.close()

    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

print("Training selesai dan tercatat di MLflow + DagsHub")