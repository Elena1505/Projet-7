import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import shap
import matplotlib.pyplot as plt
import plotly.tools as tls

df = pd.read_csv("data.csv")
df = df.drop(["Unnamed: 0"], axis=1)


# Pie chart
def pie_chart():
    percent_sup = 100 * (df['TARGET']).sum() / df.shape[0]
    percent_inf = 100 - percent_sup
    d = {'col1': [percent_sup, percent_inf], 'col2': ['% Non-creditworthy customer', '% Creditworthy customer', ]}
    d = pd.DataFrame(data=d)
    fig = px.pie(d, values='col1', names='col2')
    st.plotly_chart(fig)


# Gender pie chart
def gender_pie_chart():
    fig = px.pie(df, names=df["CODE_GENDER"], title="Gender repartition")
    st.plotly_chart(fig)


# Age histogram
def age_histogram(id, data):
    fig = px.histogram(df, x=df["DAYS_BIRTH"] / -365, title="Customer age repartition", nbins=5, labels={'x': 'Age'})
    fig.update_layout(bargap=0.1)
    marker = str(round(data["DAYS_BIRTH"] / -365).values[0])
    fig.add_vline(x=marker)
    fig.add_trace(go.Scatter(x=[marker], y=[5], mode="lines", marker=dict(color="black"),
                             name="Age of the customer " + id))
    st.plotly_chart(fig)


# Years worked histogram
def years_worked(id, data):
    fig = px.histogram(df, x=df["DAYS_EMPLOYED"] / -365, title="Years worked repartition", nbins=10,
                       labels={'x': 'Years'})
    fig.update_layout(bargap=0.1)
    marker = str(round(data["DAYS_EMPLOYED"] / -365).values[0])
    fig.add_vline(x=marker)
    fig.add_trace(go.Scatter(x=[marker], y=[5], mode="lines", marker=dict(color="black"),
                             name="Years worked by the customer " + id))
    st.plotly_chart(fig)


# Income histogram
def income(id, data):
    fig = px.histogram(df, x=df["AMT_INCOME_TOTAL"], title="Income repartition", nbins=50, labels={'x': 'Income'})
    marker = str(round(data["AMT_INCOME_TOTAL"]).values[0])
    fig.add_vline(x=marker)
    fig.add_trace(go.Scatter(x=[marker], y=[5], mode="lines", marker=dict(color="black"),
                             name="Income of the customer " + id))
    st.plotly_chart(fig)


# Children histogram
def children_pie_chart(id, data):
    fig = px.pie(df, names=df["CNT_CHILDREN"], title="Children repartition")
    st.plotly_chart(fig)


# Feature importances for a specific customer
def feature_importances_customer(index):
    model_name = "KNeighborsClassifier"
    model_version = 18
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
    df = pd.read_csv("data.csv", nrows=10)
    df = df.drop(["Unnamed: 0", "TARGET"], axis=1)
    test_x = pd.read_csv("test_x.csv", nrows=2)
    test_x = test_x.drop(["Unnamed: 0"], axis=1)
    train_x = pd.read_csv("train_x.csv", nrows=8)
    train_x = train_x.drop(["Unnamed: 0"], axis=1)
    explainer = shap.KernelExplainer(model.predict, train_x)
    data = test_x.iloc[index, :]
    shap_values = explainer.shap_values(data)
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values, feature_names=df.columns, matplotlib=True))


def main():

    st.title('Interactive dashboard')
    st.markdown("Welcome to this Interactive Dashboard! In a first part, you will find general information"
                " about customers and features importances. In a second part you will find more specific"
                " informations about specific customer. ")

    st.markdown('**Percentage of customer creditworthiness**')
    pie_chart()

    st.subheader("Choose your customer and an action:")
    id = st.text_input('Choose a customer id among : 107751, 100121, 108290, 100060, 103775 or 111126')

    info_btn = st.button('Customer informations')
    predict_btn = st.button('Creditworthiness prediction')

    if info_btn:
        data = df[df['SK_ID_CURR'] == int(id)]
        st.subheader("Main informations about the customer " + id + ":")
        st.text("Gender: " + str(data["CODE_GENDER"].values[0]))
        st.text("Age: " + str(round(data["DAYS_BIRTH"] / -365).values[0]))
        st.text("Years worked: " + str(round(data["DAYS_EMPLOYED"] / -365).values[0]))
        st.text("Income: " + str(round(data['AMT_INCOME_TOTAL'].values[0])))
        st.text("Number of child/children: " + str(round(data['CNT_CHILDREN'].values[0])))

        st.subheader("All details about the customer " + id + ":")
        st.table(data)

        st.subheader("Comparison with other customers: ")

        gender_pie_chart()
        age_histogram(id, data)
        years_worked(id, data)
        income(id, data)
        children_pie_chart(id, data)

    if predict_btn:

        st.subheader("Feature importances for the customer " + id + ":")
        global index
        if int(id) == 107751:
            index = 0
        elif int(id) == 100121:
            index = 1
        elif int(id) == 108290:
            index = 2
        elif int(id) == 100060:
            index = 3
        elif int(id) == 103775:
            index = 4
        elif int(id) == 111126:
            index = 5
        feature_importances_customer(index)


if __name__ == '__main__':
    main()
