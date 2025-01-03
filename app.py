import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv("/mnt/data/Enhanced_Dummy_HBL_Data - Sheet1.csv")

st.set_page_config(page_title="HBL Data Analysis", layout="wide")
st.title("HBL Data Analysis Dashboard")

st.header("Dataset Overview")
st.write(data)
st.write(f"Dataset size: {data.size}")

# Task 1: Account Type Distribution
st.subheader("Task 1: Distribution of Account Types")
account_type_counts = data['Account Type'].value_counts()
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(account_type_counts, labels=account_type_counts.index, autopct='%1.1f%%', startangle=140)
ax1.set_title('Distribution of Account Types')
st.pyplot(fig1)

# Task 2: Transaction Flow by Beneficiary Bank
st.subheader("Task 2: Top 5 Beneficiary Banks with Highest Credit Transactions by Region")
top_banks = data.groupby(['Region', 'Transaction To'])['Credit'].sum().reset_index()
top_banks = top_banks.sort_values(by='Credit', ascending=False).groupby('Region').head(5)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_banks, x='Transaction To', y='Credit', hue='Region', ax=ax2)
ax2.set_title('Top 5 Beneficiary Banks with Highest Credit Transactions by Region')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
st.pyplot(fig2)

# Task 3: Geographic Heatmap of Transactions
st.subheader("Task 3: Transaction Intensity by Region")
transaction_intensity = data.groupby('Region')[['Credit', 'Debit']].sum().reset_index()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(transaction_intensity.set_index('Region'), annot=True, cmap='YlGnBu', fmt='.0f', ax=ax3)
ax3.set_title('Transaction Intensity by Region')
st.pyplot(fig3)

# Task 4: Anomalies in Transactions
st.subheader("Task 4: Anomalies in Credit Transactions")
data['Credit_Z'] = (data['Credit'] - data['Credit'].mean()) / data['Credit'].std()
data['Debit_Z'] = (data['Debit'] - data['Debit'].mean()) / data['Debit'].std()
outliers_credit = data[data['Credit_Z'].abs() > 3]
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.scatter(data.index, data['Credit'], label='Credit', alpha=0.5)
ax4.scatter(outliers_credit.index, outliers_credit['Credit'], color='red', label='Outliers (Credit)', alpha=0.7)
ax4.set_title('Anomalies in Credit Transactions')
ax4.set_xlabel('Index')
ax4.set_ylabel('Credit Amount')
ax4.legend()
st.pyplot(fig4)

# Task 5: Comparative Analysis of Transaction Types
st.subheader("Task 5: Comparative Analysis of Credit and Debit Transactions by Account Type")
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=data.melt(id_vars='Account Type', value_vars=['Credit', 'Debit']),
    x='Account Type', y='value', hue='variable', ax=ax5
)
ax5.set_title('Comparative Analysis of Credit and Debit Transactions by Account Type')
ax5.set_xlabel('Account Type')
ax5.set_ylabel('Transaction Amount')
ax5.legend(title='Transaction Type')
st.pyplot(fig5)

# Task 6: Transaction Trends Over Time
if 'Time' in data.columns:
    st.subheader("Task 6: Transaction Trends Over Time")
    data['Time'] = pd.to_datetime(data['Time'])
    data = data.dropna(subset=['Time'])
    if not data.empty:
        data.set_index('Time', inplace=True)
        time_series = data.resample('D')[['Credit', 'Debit']].sum().reset_index()
        if not time_series.empty:
            fig6, ax6 = plt.subplots(figsize=(12, 6))
            ax6.plot(time_series['Time'], time_series['Credit'], label='Credit', color='blue')
            ax6.plot(time_series['Time'], time_series['Debit'], label='Debit', color='red')
            ax6.set_title("Transaction Trends Over Time")
            ax6.set_xlabel("Time")
            ax6.set_ylabel("Transaction Amount")
            ax6.legend()
            st.pyplot(fig6)
        else:
            st.write("Time series data is empty after processing.")
    else:
        st.write("No valid time data found in the dataset.")
else:
    st.write("The 'Time' column is not available in the dataset.")

# Task 7: Total Credit and Debit Amounts by Account Type
st.subheader("Task 7: Total Credit and Debit Amounts by Account Type")
customer_transactions = data.groupby('Account Type')[['Credit', 'Debit']].sum().reset_index()
fig7, ax7 = plt.subplots(figsize=(10, 6))
customer_transactions.set_index('Account Type').plot(kind='bar', stacked=True, ax=ax7)
ax7.set_title('Total Credit and Debit Amounts by Account Type')
ax7.set_xlabel('Account Type')
ax7.set_ylabel('Transaction Amount')
ax7.legend(title='Transaction Type')
st.pyplot(fig7)
