import pandas as pd
import plotly as px
import streamlit as st

#must be called as the first Streamlit command in your script.
st.set_page_config(page_title='Supermarkt Dashboard',
                   page_icon='🛒',
                   layout='wide')

#build cahce to improve web performance by reducing freq reading excel file
@st.cache_data
def get_data_from_excel():
    df = pd.read_excel('supermarkt_sales.xlsx', sheet_name='Sales', skiprows= 3, usecols ='B:R',nrows= 1000)
    #print(df_selection)
    # add 'hour column to dataframe
    df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    return df

df = get_data_from_excel()
df.info()



## Sidebar
st.sidebar.header('Please filter here:')
city = st.sidebar.multiselect(
    'Select the city:',
    options=df['City'].unique(),
    default=df['City'].unique()
)
customer_type = st.sidebar.multiselect(
    'Select the customer type:',
    options=df['Customer_type'].unique(),
    default=df['Customer_type'].unique()
)
gender = st.sidebar.multiselect(
    'Select the gender:',
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

df_selection = df.query(
    'City == @city & Customer_type == @customer_type & Gender == @gender'
)



##mainpage
st.title("Sales Dashboard")
st.markdown('---')

#top kpi
total_sales = int(df_selection['Total'].sum())
average_rating =round(df_selection['Rating'].mean(),1)
star_rating = ":star:"*int(round(average_rating,0))
average_sales_by_transaction = round(df_selection['Total'].mean(),2)

#streamlit columns
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader('Total Sales:')
    st.subheader(f"USD $ {total_sales:,}")
with middle_column:
    st.subheader('Average Rating:')
    st.subheader(f"{average_rating}")
    st.subheader(f"{star_rating}")
with right_column:
    st.subheader('Average Sales by Transaction:')
    st.subheader(f"USD $ {average_sales_by_transaction:,}")

st.markdown("---")

#define sales by product line
sales_by_product_line = (
    df_selection.groupby(by=['Product line'])[['Total']].sum().sort_values(by='Total')
)
#create bar chart using plotly
fig_product_sales = px.bar(
    sales_by_product_line,
    x='Total',
    y=sales_by_product_line.index,
    orientation='h',
    title = '<b>Sales by Product LIne</b>',
    color_discrete_sequence=['#008388'] * len(sales_by_product_line),
    template='plotly_white',
)
fig_product_sales.update_layout(
    # plot_bgcolor='rgba(0.3,0.3,0.3,0.3)',
    xaxis=(dict(showgrid=True))
)

#preprocess time column and 


st.plotly_chart(fig_product_sales)

st.dataframe(df_selection)


