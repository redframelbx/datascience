#import libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#set streamlit page config
st.set_page_config(page_title='Credit Card Customer Churn Prediction', layout='wide', initial_sidebar_state='auto')
st.image('/mount/src/datascience/churn/banner.png')
st.header('Customer Churn Prediction')
st.write('This app predicts if a customer is likely to churn or not. Please input all the rquired customer profile details.')
st.write('The results of the prediction will be displayed on the bottom of the page.')
#create sidebar
st.sidebar.header('Select Page to View')
st.write('---')

import pickle

# Load the saved model from a file
with open('mlp_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    # Load the saved model from a file
with open('svm_model.pkl', 'rb') as file:
    loaded_model2 = pickle.load(file)
    # Load the saved model from a file
with open('svm_model2.pkl', 'rb') as file:
    loaded_model3 = pickle.load(file)
with open('rf_model.pkl', 'rb') as file:
    loaded_model3a = pickle.load(file)

#user inpur for prediction on streamlit

# Assuming you have your DataFrame named 'data'
# Replace this with your actual data loading code
# Read the data
data1=pd.read_csv('SetA-CreditCardCustomers.csv')
data2=pd.read_csv('SetB-CreditCardCustomers.csv')

# Add the 'attrition' column with labels to df1 and df2
data1 = data1.assign(attrition='existing cust')
data2 = data2.assign(attrition='attrited cust')

# Concatenate df1 and df2
data=pd.concat([data1,data2],ignore_index=True)
data.drop(columns="Unnamed: 20", inplace=True)

#create dropdown button for user to select the model
# model = st.selectbox(
#     'Select Model',
#     ('Model 1', 'Model 2', 'Model 3')
# )
#create dropdown button for user to select values for prediction based on columns in the dataframe data


# Assuming 'data' is your DataFrame
# Replace 'data' with your actual DataFrame

# Create dropdowns for each column with unique values
col1, col2, col3 = st.columns(3)

with col1:
    # customer_age = st.selectbox("Select Customer Age:", data['Customer_Age'].unique())
    customer_age = st.number_input("Input Customer Age:", step=1)
    gender = st.selectbox("Select Gender:", data['Gender'].unique())
    dependent_count = st.selectbox("Select Dependent Count:", data['Dependent_count'].unique())
    education_level = st.selectbox("Select Education Level:", data['Education_Level'].unique())
    marital_status = st.selectbox("Select Marital Status:", data['Marital_Status'].unique())
    income_category = st.selectbox("Select Income Category:", data['Income_Category'].unique())
    card_category = st.selectbox("Select Card Category:", data['Card_Category'].unique())
with col2:
    # months_on_book = st.selectbox("Select Months on Book:", data['Months_on_book'].unique())
    months_on_book = st.number_input("Input Months on Book:",step=1)
    total_relationship_count = st.selectbox("Select Total Product Relationship Count:", data['Total_Relationship_Count'].unique())
    contacts_count_12_mon = st.selectbox("Select Contacts Count 12 Months:", data['Contacts_Count_12_mon'].unique())
    # credit_limit = st.selectbox("Select Credit Limit:", data['Credit_Limit'].unique())
    credit_limit = st.number_input("Input Credit Limit:")
    # total_revolving_bal = st.selectbox("Select Total Revolving Balance:", data['Total_Revolving_Bal'].unique())
    total_revolving_bal = st.number_input("Select Total Revolving Balance:")
    # avg_open_to_buy = st.selectbox("Select Avg Open To Buy:", data['Avg_Open_To_Buy'].unique())
    avg_open_to_buy = st.number_input("Input Avg Open To Buy:")
   
with col3:
    months_inactive_12_mon = st.selectbox("Select Months Inactive 12 Months:", data['Months_Inactive_12_mon'].unique())
    # total_amt_chng_q4_q1 = st.selectbox("Select Total Amount Change Q4 Q1:", data['Total_Amt_Chng_Q4_Q1'].unique())
    # total_trans_amt = st.selectbox("Select Total Transaction Amount:", data['Total_Trans_Amt'].unique())
    # total_trans_ct = st.selectbox("Select Total Transaction Count:", data['Total_Trans_Ct'].unique())
    # total_ct_chng_q4_q1 = st.selectbox("Select Total Count Change Q4 Q1:", data['Total_Ct_Chng_Q4_Q1'].unique())
    # avg_utilization_ratio = st.selectbox("Select Avg Utilization Ratio:", data['Avg_Utilization_Ratio'].unique())
    total_amt_chng_q4_q1 = st.number_input("Input total amount change Q4-Q1:")
    total_trans_amt = st.number_input("Input total transaction amount:")
    total_trans_ct = st.number_input("Input transaction count:")
    total_ct_chng_q4_q1 = st.number_input("Input Total transaction count Change Q4-Q1:")
    avg_utilization_ratio = st.number_input("Input Average_utilization ration:")

# Now you can use the selected values in your machine learning model
# For example, you can use these values as input features for prediction

# Display the selected values (for testing)
# st.write('---')
# st.write("Selected Values:")
# col1, col2, col3 = st.columns(3)

# with col1:
   
#     st.write("Customer Age:", customer_age)
#     st.write("Gender:", gender)
#     st.write("Dependent Count:", dependent_count)
#     st.write("Education Level:", education_level)
#     st.write("Marital Status:", marital_status)
#     st.write("Income Category:", income_category)
#     st.write("Card Category:", card_category)
# with col2:
#     st.write("Months on Book:", months_on_book)
#     st.write("Total Relationship Count:", total_relationship_count)
#     st.write("Contacts Count 12 Months:", contacts_count_12_mon)
#     st.write("Credit Limit:", credit_limit)
#     st.write("Total Revolving Balance:", total_revolving_bal)
#     st.write("Avg Open To Buy:", avg_open_to_buy)
#     st.write("Months Inactive 12 Months:", months_inactive_12_mon)
#     st.write("Total Amount Change Q4 Q1:", total_amt_chng_q4_q1)
# with col3:


#     st.write("Select Total Transaction Amount:"), total_trans_amt
#     st.write("Select Total Transaction Count:"), total_trans_ct
#     st.write("Select Total Count Change Q4 Q1:"), total_ct_chng_q4_q1
#     st.write("Select Avg Utilization Ratio:"), avg_utilization_ratio

# Create a dictionary with the selected values
input_data = {
    'Customer_Age': [customer_age],
    'Gender': [gender],
    'Dependent_count': [dependent_count],
    'Education_Level': [education_level],
    'Marital_Status': [marital_status],
    'Income_Category': [income_category],
    'Card_Category': [card_category],
    'Months_on_book': [months_on_book],
    'Total_Relationship_Count': [total_relationship_count],
    'Months_Inactive_12_mon': [months_inactive_12_mon],
    'Contacts_Count_12_mon': [contacts_count_12_mon],
    'Credit_Limit': [credit_limit],
    'Total_Revolving_Bal': [total_revolving_bal],
    'Avg_Open_To_Buy': [avg_open_to_buy],
    'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
    'Total_Trans_Amt': [total_trans_amt],
    'Total_Trans_Ct': [total_trans_ct],
    'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
    'Avg_Utilization_Ratio': [avg_utilization_ratio]
}

# Create a dictionary with the selected values
input_data2 = {
    # 'Customer_Age': [customer_age],
    # 'Gender': [gender],
    # 'Dependent_count': [dependent_count],
    # 'Education_Level': [education_level],
    # 'Marital_Status': [marital_status],
    # 'Income_Category': [income_category],
    # 'Card_Category': [card_category],
    # 'Months_on_book': [months_on_book],
    # 'Total_Relationship_Count': [total_relationship_count],
    # 'Contacts_Count_12_mon': [contacts_count_12_mon],
    'Credit_Limit': [credit_limit],
    'Total_Revolving_Bal': [total_revolving_bal],
    'Avg_Open_To_Buy': [avg_open_to_buy],
    'Months_Inactive_12_mon': [months_inactive_12_mon],
    'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
    'Total_Trans_Amt': [total_trans_amt],
    'Total_Trans_Ct': [total_trans_ct],
    'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
    'Avg_Utilization_Ratio': [avg_utilization_ratio]
}


# Convert the dictionary into a DataFrame
input_df = pd.DataFrame(input_data)
input_df2 = pd.DataFrame(input_data2)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
col = [
#    'CLIENTNUM',
    # 'Customer_Age',
    'Gender',
    # 'Dependent_count',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
    'Months_on_book',
    # 'Total_Relationship_Count',
#     'Months_Inactive_12_mon',
#     'Contacts_Count_12_mon',
#     'Credit_Limit',
#     'Total_Revolving_Bal',
#    'Avg_Open_To_Buy',
#     'Total_Amt_Chng_Q4_Q1',
#    'Total_Trans_Amt',
#    'Total_Trans_Ct',
#    'Total_Ct_Chng_Q4_Q1',
#    'Avg_Utilization_Ratio',
    # 'attrition'
]

for i in col:
    input_df[i] = encoder.fit_transform(input_df[i])
# st.write(input_df)

#assign user input to x
x = input_df
x2 = input_df2



scaler = StandardScaler()
xs = scaler.fit_transform(x)
#display x for user to see
# st.write(x)
# st.write(x2)
# Predict the output for user input
output0 = loaded_model.predict(x)
output1 = loaded_model2.predict(xs)
output2 = loaded_model3a.predict(x)
# Display the prediction
# You can use a dictionary or similar structure to make this output
# more human interpretable.
# st.write('---')
# st.write("Prediction:")
# if model == 'Model 1':
# st.write(output0)
# # elif model == 'Model 2':
# st.write(output1)
# # elif model == 'Model 3':
# st.write(output2)
# # else:
#     st.write('Please select a model.')

# image = Imag    e.open('Pass.png')
st.write('---')

#creeate new page for data visualization for describing the data
st.subheader('RESULT')
st.write('This app will use a predictor model to predict if a customer will churn or not. The results of the prediction are displayed below.')
st.write('---')
one = st.checkbox('Use one predictor only', value=True)
#------------------#
if one:
    st.write('**MODEL DELTA**')
    st.image('pages/delta.png')
    if output2 == 'existing cust':
        st.write('Model Delta predicted this customer as a Normal Customer')
        st.image('pages/Pass.png')
    else:
        st.write('Model Delta predicted this customer WILL churn')
        st.image('pages/fail.png')

three = st.checkbox('ADVANCED USER ONLY!: Use all three models to predict')

if three:
        
    #------------------#    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('**MODEL ALPHA**')
        st.image('pages/ALPHA.png')
        if output0 == 1:
            st.write('Model Alpha predicted this customer as a Normal Customer')
            st.image('pages/Pass.png')
            
        else:
            st.write('Model Alpha predicted this customer WILL churn')
            st.image('pages/fail.png')
    with col2:
        st.write('**MODEL BETA**')
        st.image('pages/beta.png')
        if output1 == 'existing cust':
            st.write('Model Beta predicted this customer as a Normal Customer')
            st.image('pages/Pass.png')
            
        else:
            st.write('Model Beta predicted this customer WILL churn')
            st.image('pages/fail.png')
    with col3:
        st.write('**MODEL DELTA**')
        st.image('pages/delta.png')
        if output2 == 'existing cust':
            st.write('Model Delta predicted this customer as a Normal Customer')
            st.image('pages/Pass.png')
        else:
            st.write('Model Deltaa predicted this customer WILL churn')
            st.image('pages/fail.png')




# st.write('**MODEL ALPHA**')
# st.image('pages/ALPHA.png')
# if output0 == 1:
#     st.image('pages/Pass.png')
#     st.write('Normal Customer')
# else:
#     st.write('Probable Attrited Customer')
#     st.image('pages/fail.png')

# st.write('**MODEL OMEGA**')
# st.image('pages/beta.png')
# if output0 == 1:
#     st.image('pages/Pass.png')
#     st.write('Normal Customer')
# else:
#     st.write('Probable Attrited Customer')
#     st.image('pages/fail.png')

# st.write('**MODEL DELTA**')
# st.image('pages/delta.png')
# if output0 == 1:
#     st.image('pages/Pass.png')
#     st.write('Normal Customer')
# else:
#     st.write('Probable Attrited Customer')
#     st.image('pages/fail.png')
