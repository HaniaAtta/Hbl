import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv("data/Enhanced_Dummy_HBL_Data - Sheet1.csv")

# Set Streamlit page configuration
st.set_page_config(page_title="HBL Data Analysis", layout="wide")
st.title("HBL Data Analysis Dashboard")

# Dataset Overview
st.header("Dataset Overview")
st.write(data)
st.write(f"Dataset size: {data.size}")

# Define custom colors
colors = {
    'dark_blue': '#0c4160',
    'slate_blue': '#38495a',
    'light_beige': '#e8d8c4',
    'dark_slate': '#1B2735'
}

# Task 1: Account Type Distribution
st.subheader("Task 1: Distribution of Account Types")
account_type_counts = data['Account Type'].value_counts()
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(account_type_counts, labels=account_type_counts.index, autopct='%1.1f%%', startangle=140, 
         colors=[colors['dark_blue'], colors['slate_blue'], colors['light_beige'], colors['dark_slate']])
ax1.set_title('Distribution of Account Types')
st.pyplot(fig1)
st.write("**Explanation:** This pie chart illustrates the distribution of different account types in the dataset. "
         "It shows the proportion of each account type, helping to identify which types are most common. "
         "For instance, if one account type dominates, it may indicate a specific customer preference or business focus.")

# Task 2: Transaction Flow by Beneficiary Bank
st.subheader("Task 2: Top 5 Beneficiary Banks with Highest Credit Transactions by Region")
top_banks = data.groupby(['Region', 'Transaction To'])['Credit'].sum().reset_index()
top_banks = top_banks.sort_values(by='Credit', ascending=False).groupby('Region').head(5)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_banks, x='Transaction To', y='Credit', hue='Region', ax=ax2, 
            palette=[colors['dark_blue'], colors['slate_blue'], colors['light_beige'], colors['dark_slate']])
ax2.set_title('Top 5 Beneficiary Banks with Highest Credit Transactions by Region')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
st.pyplot(fig2)
st.write("**Explanation:** This bar chart displays the top 5 beneficiary banks with the highest credit transactions for each region. "
"It provides insights into regional banking preferences and highlights which banks are most frequently used for credit transactions.")

# Task 3: Geographic Heatmap of Transactions
st.subheader("Task 3: Transaction Intensity by Region")
transaction_intensity = data.groupby('Region')[['Credit', 'Debit']].sum().reset_index()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(transaction_intensity.set_index('Region'), annot=True, cmap='YlGnBu', fmt='.0f', ax=ax3)
ax3.set_title('Transaction Intensity by Region')
st.pyplot(fig3)
st.write("**Explanation:** This heatmap visualizes the intensity of credit and debit transactions by region. "
"The annotations provide exact transaction amounts, allowing for quick identification of regions with high transaction volumes. This can help in understanding regional economic activity.")

# Task 4: Anomalies in Transactions
st.subheader("Task 4: Anomalies in Credit Transactions")
data['Credit_Z'] = (data['Credit'] - data['Credit'].mean()) / data['Credit'].std()
data['Debit_Z'] = (data['Debit'] - data['Debit'].mean()) / data['Debit'].std()
outliers_credit = data[data['Credit_Z'].abs() > 3]
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.scatter(data.index
