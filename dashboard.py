import pandas as pd
import plotly.express as px
import streamlit as st

#must be called as the first Streamlit command in your script.
st.set_page_config(page_title='redframe Supermarket Dashboard',
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
st.title("Redframe Sales Dashboard")
st.markdown('---')



#top kpi
total_sales = int(df_selection['Total'].sum())
money_rating = ":dollar:"*int(round(total_sales/10000,0))
average_rating =round(df_selection['Rating'].mean(),1)
star_rating = ":star:"*int(round(average_rating,0))
average_sales_by_transaction = round(df_selection['Total'].mean(),2)
basket_size = ":shopping_bags:"*int(round(average_sales_by_transaction/100,0))




#streamlit columns
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"USD $ {total_sales:,}")
    # st.subheader('How much money:')
    st.subheader(f"{money_rating}")
with middle_column:
    st.subheader('Average Rating:')
    st.subheader(f"{average_rating}")
    # st.subheader('How many stars:')
    st.subheader(f"{star_rating}")
with right_column:
    st.subheader('Average Sales by Transaction:')
    st.subheader(f"USD $ {average_sales_by_transaction:,}")
    # st.subheader('How many bags per customer:')
    st.subheader(f"{basket_size}")


#define sales by product line
sales_by_product_line = (
    df_selection.groupby(by=['Product line'])[['Total']].sum().sort_values(by='Total')
)

#define sales by date
sales_by_date = df_selection.groupby(by=['Date'])[['Total']].sum()

#create bar chart using plotly
#define plotly sales by product
fig_product_sales = px.bar(
    sales_by_product_line,
    x='Total',
    y=sales_by_product_line.index,
    orientation='h',
    title = '<b>Total Sales by Product LIne</b>',
    # color_discrete_sequence=['#008388'] * len(sales_by_product_line),
    template='plotly',
)
fig_product_sales.update_layout(
    # plot_bgcolor='rgba(0.3,0.3,0.3,0.3)',
    xaxis=(dict(showgrid=True)),
    # plot_bgcolor='#035efc'
)
#define plotly sales by date
fig_sales_date = px.line(
    sales_by_date,
    # x='Total',
    # y=sales_by_date.index,
    y='Total',
    x=sales_by_date.index,
    # orientation='h',
    title = '<b>Total Sales by Date</b>',
    # color_discrete_sequence=['#008388'] * len(sales_by_date),
    template='plotly',
    markers = True
)
fig_sales_date.update_layout(
    # plot_bgcolor='rgba(0.3,0.3,0.3,0.3)',
    xaxis=(dict(showgrid=True)),
    # plot_bgcolor='#035efc'
)

#create sales by hour graph
sales_by_hour = df_selection.groupby(by=['hour'])[['Total']].sum()
fig_hourly_sales = px.bar(
    sales_by_hour,
    x = sales_by_hour.index,
    y='Total',
    title = '<b>Total Sales by Hour</b>',
    # color_discrete_sequence = ['#008388']* len(sales_by_hour),
    template = 'plotly_white',
)
fig_hourly_sales.update_layout(
    xaxis=dict(tickmode='linear'),
    yaxis=(dict(showgrid=True)),
    # plot_bgcolor='#035efc'
)

#define sunburst
fig_sunburst = px.sunburst(df, path=['City','Customer_type','Gender'], values='Total',
                    template='plotly', title='Total Company Sales Breakdown \n click at the chart for more detail view. long press to show amount')
fig_sunburst_filter = px.sunburst(df_selection, path=['City','Customer_type','Gender'], values='Total',
                    template='plotly', title= 'Total Sales by Filtered City Customer types & Gender ')

#st.plotly_chart(fig_product_sales)
# st.plotly_chart(fig_sales_date, use_container_width=True)

#original chart column
#sunburst chart
# left_column, right_column = st.columns(2)
# left_column.plotly_chart(fig_sunburst,use_container_width=True)
# right_column.plotly_chart(fig_sunburst_filter,use_container_width=True)
# # st.plotly_chart(fig_hourly_sales, use_container_width=True)
# # st.plotly_chart(fig_product_sales, use_container_width=True)
# #set the graph to appear next to each other
# left_column, right_column = st.columns(2)
# left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
# right_column.plotly_chart(fig_product_sales, use_container_width=True)
#end of original chart column

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_hourly_sales,use_container_width=True)
right_column.plotly_chart(fig_sunburst,use_container_width=True)
# st.plotly_chart(fig_hourly_sales, use_container_width=True)
# st.plotly_chart(fig_product_sales, use_container_width=True)
st.plotly_chart(fig_sales_date, use_container_width=True)
#set the graph to appear next to each other
left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_product_sales, use_container_width=True)
right_column.plotly_chart(fig_sunburst_filter, use_container_width=True)


st.dataframe(df_selection)


