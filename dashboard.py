import streamlit as st
import pandas as pd
import plotly.express as px


df = pd.read_csv("data.csv")
df = df.drop(["Unnamed: 0"], axis=1)


# Pie chart
def pie_chart():
    percent_sup = 100 * (df['TARGET']).sum() / df.shape[0]
    percent_inf = 100 - percent_sup
    d = {'col1': [percent_sup, percent_inf], 'col2': ['% Non-creditworthy customer', '% Creditworthy customer', ]}
    d = pd.DataFrame(data=d)
    fig = px.pie(d, values='col1', names='col2', title='Percentage of customer creditworthiness')
    st.plotly_chart(fig)


def main():
    st.title('Interactive dashboard')
    st.markdown("Welcome to this Interactive Dashboard! In a first part, you will find general information"
                " about customers and features importances. In a second part you will find more specific"
                " informations about specific customer. ")

    pie_chart()


if __name__ == '__main__':
    main()