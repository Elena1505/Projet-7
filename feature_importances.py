import shap
import mlflow.pyfunc
import pandas as pd
import pickle


# Import my model with mlflow
model_name = "KNeighborsClassifier"
model_version = 18
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

df = pd.read_csv("data.csv")
df = df.drop(["Unnamed: 0", "TARGET"], axis=1)

train_x = pd.read_csv("train_x.csv")
train_x = train_x.drop(["Unnamed: 0"], axis=1)

test_x = pd.read_csv("test_x.csv")
test_x = test_x.drop(["Unnamed: 0"], axis=1)


# Feature importances
explainer = shap.KernelExplainer(model.predict_proba, train_x)
shap_values = explainer.shap_values(test_x)

with open("shap_val.pickle", "wb") as f:
    pickle.dump(shap_values, f)

