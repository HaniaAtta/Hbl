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

# Set the background color to a milk-like color and custom styles
st.markdown(
     """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    .reportview-container {
        background: #e8d8c4;  /* Milk-like color */
        font-family: 'Roboto', sans-serif;  /* Custom font */
    }
    .sidebar .sidebar-content {
        background: #e8d8c4;  /* Milk-like color for sidebar */
    }
    h1 {
        text-align: center;  /* Center headers */
        margin: 20px 0;  /* Add spacing */
        font-size: 34px;  /* Adjust font size */
    }
    h2 {
        text-align: center;  /* Center subheaders */
        margin: 20px 0;  /* Add spacing */
       
    }
    h3 {
        text-align: center;  /* Center sub-subheaders */
        margin: 20px 0;  /* Add spacing */
       
    }
    .centered-table {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
    }
    .plot-container {
        margin: 20px 0;  /* Add spacing around plots */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dataset Overview
st.header("Dataset Overview")

# Center the dataset table
st.markdown('<div class="centered-table">', unsafe_allow_html=True)
st.write(data)
st.markdown('</div>', unsafe_allow_html=True)

st.write(f"Dataset size: {data.size}")

# Define custom colors
colors = {
    'dark_blue': '#B85042',
    'slate_blue': '#A7BEAE',
    'light_beige': '#E7E8D1',
    'dark_slate': '#A7BEAE'
}

# Sliders for plot dimensions
width = st.sidebar.slider("Plot Width", 1, 22, 4)
height = st.sidebar.slider("Plot Height", 1, 22, 3)

# Sliders for font sizes
title_font_size = st.sidebar.slider("Title Font Size", 2, 30, 8)
label_font_size = st.sidebar.slider("Label Font Size", 2, 20, 6)
legend_font_size = st.sidebar.slider("Legend Font Size", 2, 20, 6)

# Task 1: Account Type Distribution
st.markdown("<h2 style='font-size: 40px;'>Task 1: Distribution of Account Types</h2>", unsafe_allow_html=True)
account_type_counts = data['Account Type'].value_counts()
fig1, ax1 = plt.subplots(figsize=(width, height))  # Smaller plot size
ax1.pie(account_type_counts, labels=account_type_counts.index, autopct=lambda p: f'{p:.1f}%', startangle=150,
         colors=[colors['dark_blue'], colors['slate_blue'], colors['light_beige'], colors['dark_slate']],
         textprops={'fontsize': label_font_size})  # Adjust label font size here
ax1.set_title('Distribution of Account Types', fontsize=title_font_size)  # Adjust title font size

# Display the pie chart
st.pyplot(fig1)

st.markdown("<p style='font-size: 21px;'>Explanation: This pie chart illustrates the distribution of different account types in the dataset. "
            "It shows the proportion of each account type, helping to identify which types are most common. "
            "For instance, if one account type dominates, it may indicate a specific customer preference or business focus.</p>", unsafe_allow_html=True)
# Task 2: Transaction Flow by Beneficiary Bank
st.markdown("<h2 style='font-size: 40px;'>Task 2: Top 5 Beneficiary Banks with Highest Credit Transactions by Region</h2>", unsafe_allow_html=True)

top_banks = data.groupby(['Region', 'Transaction To'])['Credit'].sum().reset_index()
top_banks = top_banks.sort_values(by='Credit', ascending=False).groupby('Region').head(5)
fig2, ax2 = plt.subplots(figsize=(width, height))  # Smaller plot size
sns.barplot(data=top_banks, x='Transaction To', y='Credit', hue='Region', ax=ax2, 
            palette=[colors['dark_blue'], colors['slate_blue'], colors['light_beige'], colors['dark_slate']])
ax2.set_title('Top 5 Beneficiary Banks with Highest Credit Transactions by Region', fontsize=title_font_size)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=label_font_size)

st.pyplot(fig2)
st.write("<p style='font-size: 21px;'>Explanation: This bar chart displays the top 5 beneficiary banks with the highest credit transactions for each region. "
"It provides insights into regional banking preferences and highlights which banks are most frequently used for credit transactions.</p>", unsafe_allow_html=True)

# Task 3: Geographic Heatmap of Transactions
st.markdown("<h2 style='font-size: 40px;'>Task 3: Transaction Intensity by Region</h2>", unsafe_allow_html=True)


transaction_intensity = data.groupby('Region')[['Credit', 'Debit']].sum().reset_index()

# Create a custom colormap using the defined colors
custom_cmap = sns.color_palette([colors['dark_blue'], colors['slate_blue'], colors['light_beige'], colors['dark_slate']])

fig3, ax3 = plt.subplots(figsize=(width, height))  # Smaller plot size
sns.heatmap(transaction_intensity.set_index('Region'), annot=True, cmap=custom_cmap, fmt='.0f', ax=ax3)
ax3.set_title('Transaction Intensity by Region', fontsize=title_font_size)

st.pyplot(fig3)
st.write("<p style='font-size: 21px;'>Explanation: This heatmap visualizes the intensity of credit and debit transactions by region. The annotations provide exact transaction amounts, allowing for quick identification of regions with high transaction volumes. This can help in understanding regional economic activity.</p>", unsafe_allow_html=True)

# Task 4: Anomalies in Transactions
st.markdown("<h2 style='font-size: 40px;'>Task 4: Anomalies in Credit Transactions</h2>", unsafe_allow_html=True)

data['Credit_Z'] = (data['Credit'] - data['Credit'].mean()) / data['Credit'].std()
data['Debit_Z'] = (data['Debit'] - data['Debit'].mean()) / data['Debit'].std()
outliers_credit = data[data['Credit_Z'].abs() > 3]
fig4, ax4 = plt.subplots(figsize=(width, height))  # Smaller plot size
ax4.scatter(data.index, data['Credit'], label='Credit', alpha=0.5, color=colors['dark_blue'])
ax4.scatter(outliers_credit.index, outliers_credit['Credit'], color='red', label='Outliers (Credit)', alpha=0.7)
ax4.set_title('Anomalies in Credit Transactions', fontsize=title_font_size)
ax4.set_xlabel('Index', fontsize=label_font_size)
ax4.set_ylabel('Credit Amount', fontsize=label_font_size)
ax4.legend(fontsize=legend_font_size)

st.pyplot(fig4)
st.write("<p style='font-size: 21px;'>Explanation: This scatter plot identifies anomalies in credit transactions by highlighting outliers in red. Outliers can indicate unusual transaction behavior, which may warrant further investigation for fraud detection or error correction.</p>", unsafe_allow_html=True)

# Task 5: Comparative Analysis of Transaction Types
st.markdown("<h2 style='font-size: 40px;'>Task 5: Comparative Analysis of Credit and Debit Transactions by Account Type</h2>", unsafe_allow_html=True)

fig5, ax5 = plt.subplots(figsize=(width, height))  # Smaller plot size
sns.boxplot(
    data=data.melt(id_vars='Account Type', value_vars=['Credit', 'Debit']),
    x='Account Type', y='value', hue='variable', ax=ax5, palette=[colors['dark_blue'], colors['slate_blue']]
)
ax5.set_title('Comparative Analysis of Credit and Debit Transactions by Account Type', fontsize=title_font_size)
ax5.set_xlabel('Account Type', fontsize=label_font_size)
ax5.set_ylabel('Transaction Amount', fontsize=label_font_size)
ax5.legend(title='Transaction Type', fontsize=legend_font_size)

st.pyplot(fig5)
st.write("<p style='font-size: 21px;'>Explanation: This box plot compares the distribution of credit and debit transactions across different account types. It highlights the median, quartiles, and potential outliers, providing insights into the transaction behavior of various account types.</p>", unsafe_allow_html=True)

# Task 6: Transaction Trends Over Time
st.markdown("<h2 style='font-size: 40px;'>Task 6: Time-Based Analysis (if applicable)</h2>", unsafe_allow_html=True)

if 'Time' in data.columns:
    st.subheader("Task 6: Transaction Trends Over Time")
    data['Time'] = pd.to_datetime(data['Time'])
    data = data.dropna(subset=['Time'])
    if not data.empty:
 
        data.set_index('Time', inplace=True)
        time_series = data.resample('D')[['Credit', 'Debit']].sum().reset_index()
        if not time_series.empty:
            fig6, ax6 = plt.subplots(figsize=(width, height))  # Smaller plot size
            ax6.plot(time_series['Time'], time_series['Credit'], label='Credit', color=colors['dark_blue'])
            ax6.plot(time_series['Time'], time_series['Debit'], label='Debit', color=colors['slate_blue'])
            ax6.set_title("Transaction Trends Over Time", fontsize=title_font_size)
            ax6.set_xlabel("Time", fontsize=label_font_size)
            ax6.set_ylabel("Transaction Amount", fontsize=label_font_size)
            ax6.legend(fontsize=legend_font_size)

            st.pyplot(fig6)
        else:
            st.write("Time series data is empty after processing.")
    else:
        st.write("No valid time data found in the dataset.")
else:
    st.write("The 'Time' column is not available in the dataset.")

st.write("<p style='font-size: 21px;'>Explanation: This line plot is intended to illustrate the trends of credit and debit transactions over time, allowing for the identification of patterns, seasonal effects, or anomalies in transaction behavior. However, since the dataset does not contain a 'Time' column, the analysis could not be performed, and thus no time-based trends are displayed. This absence of time data limits the ability to forecast future transactions based on historical data.</p>", unsafe_allow_html=True)

# Task 7: Total Credit and Debit Amounts by Account Type
st.markdown("<h2 style='font-size: 40px;'>Task 7: Total Credit and Debit Amounts by Account Type</h2>", unsafe_allow_html=True)

customer_transactions = data.groupby('Account Type')[['Credit', 'Debit']].sum().reset_index()
fig7, ax7 = plt.subplots(figsize=(width, height))  # Smaller plot size
customer_transactions.set_index('Account Type').plot(kind='bar', stacked=True, ax=ax7, color=[colors['dark_blue'], colors['slate_blue']])
ax7.set_title('Total Credit and Debit Amounts by Account Type', fontsize=title_font_size)
ax7.set_xlabel('Account Type', fontsize=label_font_size)
ax7.set_ylabel('Transaction Amount', fontsize=label_font_size)
ax7.legend(title='Transaction Type', fontsize=legend_font_size)

st.pyplot(fig7)
st.write("<p style='font-size: 21px;'>Explanation: This stacked bar chart visualizes the total credit and debit amounts for each account type. "
         "It provides a clear comparison of how different account types contribute to overall transaction volumes. "
         "These insights can guide strategic decisions, such as tailoring services to high-transaction account types or addressing gaps in others.</p>", unsafe_allow_html=True)


# Task: Correlation Matrix (if applicable)
st.markdown("<h2 style='font-size: 40px;'>Correlation Matrix</h2>", unsafe_allow_html=True)

# Ensure only numeric columns are included
numeric_data = filtered_data[['Credit', 'Debit']].dropna()

if not numeric_data.empty:
    corr_matrix = numeric_data.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(width, height))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title('Correlation Matrix', fontsize=title_font_size)
    st.pyplot(fig_corr)
else:
    st.write("No numeric data available for correlation analysis.")


