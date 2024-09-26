#import libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#set streamlit page config
st.set_page_config(page_title='Credit Card Customer Churn Prediction', layout='wide', initial_sidebar_state='auto')
st.image('./churn/banner.png')
st.header('Customer Churn Prediction')
st.write('This app predicts if a customer will churn or not by batch. Please upload the file according the template given.')
st.write('The results of the prediction will be displayed on the bottom of the page.')
#create sidebar
st.sidebar.header('Select Page to View')
template=pd.read_csv('./churn/pages/templatedownload.csv')
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(template)

st.download_button(
    label="Download template & sample data",
    data=csv,
    file_name='template.csv',
    mime='text/csv',
)

# import pickle

# import pickle

# Load the saved model from a file
with open('./churn/mlp_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    # Load the saved model from a file
with open('./churn/svm_model.pkl', 'rb') as file:
    loaded_model2 = pickle.load(file)
    # Load the saved model from a file
with open('./churn/svm_model2.pkl', 'rb') as file:
    loaded_model3 = pickle.load(file)
with open('./churn/rf_model.pkl', 'rb') as file:
    loaded_model3a = pickle.load(file)

# # Load the saved model from a file
# with open('mlp_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
#     # Load the saved model from a file
# with open('mlp_model2.pkl', 'rb') as file:
#     loaded_model2 = pickle.load(file)
#     # Load the saved model from a file
# with open('mlp_model3.pkl', 'rb') as file:
#     loaded_model3 = pickle.load(file)
# with open('mlp_model3a.pkl', 'rb') as file:
#     loaded_model3a = pickle.load(file)

uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file,  encoding='utf-8')
    # st.write(input_data)
    selected_columns = [
        # Columns you want to select
        #    'CLIENTNUM',
        'Customer_Age',
        'Gender',
        'Dependent_count',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        # 'attrition'
    ]
    input_data3 = input_data[selected_columns]  # Update input_data with selected columns
    # Now you can work with the 'input_data' DataFrame
    # input_data3 = input_data[selected_columns]
# # input_data3.drop(columns="CLIENTNUM", inplace=True)


    # List of columns to keep
    columns_to_keep = [
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Months_Inactive_12_mon',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
    ]
    # Create a new DataFrame with only the selected columns
    input_data4 = input_data3[columns_to_keep]
    # st.write(input_data3)
    st.write('---')


    #user inpur for prediction on streamlit

    # Assuming you have your DataFrame named 'data'
    # Replace this with your actual data loading code
    #Read the data
    data1=pd.read_csv('./churn/SetA-CreditCardCustomers.csv')
    data2=pd.read_csv('./churn/SetB-CreditCardCustomers.csv')

    # Add the 'attrition' column with labels to df1 and df2
    data1 = data1.assign(attrition='existing cust')
    data2 = data2.assign(attrition='attrited cust')

    # Concatenate df1 and df2
    data=pd.concat([data1,data2],ignore_index=True)
    data.drop(columns="Unnamed: 20", inplace=True)

    # Convert the dictionary into a DataFrame
    input_df = input_data3
    input_df2 = input_data4

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

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    # st.write(xs)
    #display x for user to see
    # st.write(x)
    # st.write(x2)
    # Predict the output for user input
    output0 = loaded_model.predict(x)
    output1 = loaded_model2.predict(xs)
    #output2 = loaded_model3a.predict(x)
    # st.write(output0)
    # st.write(output1)
    # st.write(output2)

    # Function to map prediction values to category names
    def map_predictions(output0):
        category_mapping = {0: 'Attrited', 1: 'Normal'}  # Define your mapping here
        return [category_mapping[pred] for pred in output0]
    def map_predictions1(output1):
        category_mapping = {'attrited cust': 'Attrited', 'existing cust': 'Normal'}  # Define your mapping here
        return [category_mapping[pred] for pred in output1]
    def map_predictions2(output2):
        category_mapping = {'attrited cust': 'Attrited', 'existing cust': 'Normal'}  # Define your mapping here
        return [category_mapping[pred] for pred in output2]
    # Map prediction values to category names
    category_predictions = map_predictions(output0)
    category_predictions1 = map_predictions1(output1)
    #category_predictions2 = map_predictions2(output2)
    # Include CLIENTNUM in the output_data DataFrame
    output_data00 = pd.DataFrame({'Prediction_Alpha':category_predictions})
    output_data01 = pd.DataFrame({'Prediction_Beta':category_predictions1})
    #output_data02 = pd.DataFrame({'Prediction_Omega':category_predictions2})
    #concatenate the input data and the prediction
    output_data = pd.concat([input_data, output_data00,output_data01,output_data02], axis=1)

    st.write('Predicted Result based on Model Alpha, Beta & Omega')
    st.write(output_data)

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(output_data)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='prediction_result.csv',
        mime='text/csv',
    )





else:
    st.write("Please upload a CSV file to proceed.")



# # input_data3=pd.read_csv(uploaded_files)
# selected_columns = [
# #    'CLIENTNUM',
#     'Customer_Age',
#     'Gender',
#     'Dependent_count',
#     'Education_Level',
#     'Marital_Status',
#     'Income_Category',
#     'Card_Category',
#     'Months_on_book',
#     'Total_Relationship_Count',
#     'Months_Inactive_12_mon',
#     'Contacts_Count_12_mon',
#     'Credit_Limit',
#     'Total_Revolving_Bal',
#     'Avg_Open_To_Buy',
#     'Total_Amt_Chng_Q4_Q1',
#    'Total_Trans_Amt',
#    'Total_Trans_Ct',
#    'Total_Ct_Chng_Q4_Q1',
#    'Avg_Utilization_Ratio',
#     # 'attrition'
# ]


#
