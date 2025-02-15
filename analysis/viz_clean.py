import streamlit as st
import pandas as pd
import numpy as np
from plotly.tools import FigureFactory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="IAS Dataset Analysis", page_icon="📚", layout="wide")
#This streamlit library function sets the page title as evident on the tab where the application is running and the favicon

st.header('IAS Subject and Experience Analysis Tool')



body1="This application aims to provide an interactive tool to visualise variables <u>Subject</u> and <u>Department</u> from the TCPD-IAS Dataset. A good starting point to understand how it is build would be to refer to the linked **TCPD column here--provide the link**. This application utilizes [Streamlit.io](https://streamlit.io/) and [Plotly](https://plotly.com/). Streamlit.io provides a powerful Streamlit library, which has functions that can make visualisations and add features quickly with minimal code. Moreover, Streamlit.io provides a cloud solution to host the visualisations dashboard online and sharing easy. In this application Streamlit library functions and Plotly functions are used to render charts and tables. The dashboard is then hosted using Streamlit cloud utility with data source at Github."
st.markdown(body1, unsafe_allow_html=True)

body2="Expandable sections contains code blocks for replicating the application(like the one just below). Hence it can be ignored if not interested in code development."
st.markdown(body2, unsafe_allow_html=False)


#This dict will be used to map different colour for each subject from the 8 broad categories 
colors = {}

#Initial value
lower = 100


#Function to iterate over the list of 8 broad subject category and provide color value
def color_map(item):
    return colors[item]

#This function optimize performance by re-running the followed function if any change relevant to it happens. 
@st.cache
#Function to load the unigram mapping data from csv and filter
def load_unigram_data():
    #df = pd.read_csv('~/Downloads/ias-officers/analysis/processed/unigram maps.csv')
    df = pd.read_csv('./analysis/processed/unigram maps.csv')
    #Dropping rows where the experience is NA
    df = df.loc[df["Experience"] != "N.A."]
    
    #Adding a new column that contains the count value as string, for example 'size:count' 
    df["text"] = df["Count"].apply(lambda x: "size: "+str(x))
    
    #Providing each subject out of the 8 broad caategories a unique color code
    for i, item in enumerate(df["Subject"].unique()):
        colors[item] = lower + i
    
    #Populating the color code as defined above, available in the 'colors' dictionary to the whole dataset
    df["colors"] = df["Subject"].map(color_map)
    return df
   


#Below categories in the Department of Experience have most entries, will be utilised to remove these from graphs and zoom in other Department of Experience	
admin_categories = [
    "Land Revenue Mgmt & District Admn",
    "Personnel and General Administration",
    "Finance"
]




def filter_by_value(df, col, value, include_admin=True, number_of_rows=5, percentage=False, include_other=False):

    df = df.loc[df[col].str.contains(value)]
    df.sort_values(["Count"], ascending=False, inplace=True)

    if include_admin and col == "Subject":
        for category in admin_categories:
            df = df.loc[df["Experience"] != category]

    temp = df[number_of_rows:]
    df = pd.DataFrame(df.head(number_of_rows))
    value_sum = temp["Count"].sum()
    
    if include_other:
        df = df.append({
            df.columns[0]: "Remaining",
            df.columns[1]: "Remaining",
            "Count": value_sum
        }, ignore_index=True)

    df.drop(columns=[col], inplace=True)
    df.set_index(df.columns[0], inplace=True) 

    if percentage:
        sum = df["Count"].sum()
        df = pd.DataFrame(df["Count"].apply(lambda x: round(((x / sum) * 100), 2)))

    try:
        return df.drop(columns=["text", "colors"])
    except:
        return df





def scatter_plot(df, include_top_cat=True, min_value=50):
    if include_top_cat:
        df = df.loc[df["Experience"] != 'Land Revenue Mgmt & District Admn']
        df = df.loc[df["Experience"] != 'Personnel and General Administration']
        df = df.loc[df["Experience"] != 'Finance']

    df = df.loc[df["Count"] > min_value]
    max_value = df["Count"].max()

    length_y_axis = df["Subject"].unique().shape[0]
    length_x_axis = df["Experience"].unique().shape[0]

    fig = go.Figure(data=[go.Scatter(
        x=df["Experience"].to_list(),
        y=df["Subject"].to_list(),
        text=df["text"].to_list(),
        mode="markers",
        marker=dict(
            size=df["Count"].apply(lambda x: x*50/max_value).to_list(),
            color = df["colors"].to_list(),
        )
    )])

    fig = fig.update_layout(
        autosize=False,
        height=100 + (length_y_axis * 75),
        width=100 + (length_x_axis * 75),
    )

    # remove grid lines from the figure
    fig.update_xaxes(
        showgrid=False,
        type="category",
    )
    fig.update_yaxes(showgrid=False)
    
    return fig




def pie_chart(df):
    fig = px.pie(
        df, 
        names=df.index, 
        values=df["Count"], 
        title="Subjects",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
    title={
        'text': "Subjects",
        'y':0.9,
        'x':0.5,
        'xanchor':'center',
        'yanchor':'top'})
        
    return fig.update_traces(
        hoverinfo="label+percent",
        textinfo="value",
        insidetextfont=dict(
            color="white"
        ),
    )




def bar_chart(df, title="", x_axis_title=""):
    fig = px.bar(
        df,
        x=df.index,
        y="Count",
        title=title
    )

    fig.update_layout(
        autosize=False,
        height=600
    )

    fig.update_xaxes(
        title=x_axis_title,
    )

    fig.update_yaxes(
        title="Count",
        showgrid=False,
    )

    

    return fig.update_traces(
        hoverinfo="text",
        insidetextfont=dict(
            color="white"
        ),
    )



unigram_data = load_unigram_data()


st.subheader('Unigram maps')

body_unigram_maps='This table demonstrates the count value for each pair of Subject and Experience and acts as source.'
st.markdown(body_unigram_maps, unsafe_allow_html=True)

st.write(unigram_data.drop(columns=["text", "colors"]))

   
#Add a horizontal bar
st.markdown("""<hr/>""", unsafe_allow_html=True)


#Add sub heading on the sidebar
st.sidebar.subheader('Select a Category of Experience')
option_experience = st.sidebar.selectbox("", unigram_data['Experience'].unique())
include_other_experience = st.sidebar.checkbox("Include combined remaining entries")
number_of_rows_experience = st.sidebar.slider('Number of rows', min_value=1, max_value=8, value=5)
st.sidebar.markdown("""<hr/>""", unsafe_allow_html=True)

#Add sub heading on the main page
st.subheader('Subject occurances for the Category of Experience')
st.markdown("Choose the appropiate filter on the sidebar at left under the **Select a Category of Experience**",unsafe_allow_html=True)
st.write("\n")

#Filtering dataframe based on the filters selected on the sidebar
filtered_df_experience = filter_by_value(unigram_data.copy(), 'Experience', option_experience, number_of_rows=number_of_rows_experience, include_other=include_other_experience)
st.write(filtered_df_experience)        

#Utilising Streamlit container function to make a grid and place Bar and Pie Chart side-by-side
plot_container_experience = st.container()

col1, col2 = plot_container_experience.columns([7, 1])
col1.plotly_chart(bar_chart(filtered_df_experience, title="Number of Subject occurances with respect to chosen Category of Experience", x_axis_title="Subjects"))
col2.plotly_chart(pie_chart(filtered_df_experience))
st.markdown("""<hr/>""", unsafe_allow_html=True)


#Filter options for the second sub heading 
st.sidebar.subheader('Select a Subject')
option_subject = st.sidebar.selectbox("", unigram_data['Subject'].unique())
include_admin = st.sidebar.checkbox('Remove Top Admin Categories')
percentage = st.sidebar.checkbox('Show percentage', value=True)
number_of_rows = st.sidebar.slider('Number of rows', min_value=1, max_value=47, value=5)


st.subheader('Category of Experience occurances with respect to Subject')
st.markdown("Choose the appropiate filter on the sidebar at left under the **Select a Subject**",unsafe_allow_html=True)
st.write("\n")
filtered_df_subject = filter_by_value(unigram_data.copy(), 'Subject', option_subject, include_admin=include_admin, number_of_rows=number_of_rows, percentage=percentage)
st.write(filtered_df_subject)
    
#Containers in streamlit provide functionality to divide the area into columns and control the content to be placed in each column 
plot_container_subject = st.container()
col3, col4= plot_container_subject.columns([7,1])

col3.plotly_chart(bar_chart(filtered_df_subject, title="Number of Category of Experience occurances with respect to chosen Subject", x_axis_title="Category of Experience"))
col4.plotly_chart(pie_chart(filtered_df_subject))

st.sidebar.markdown("""<hr/>""", unsafe_allow_html=True)
st.markdown("""<hr/>""", unsafe_allow_html=True)

st.sidebar.subheader('Bubble map')
st.subheader('Bubble map')
st.markdown("Choose the appropiate filter on the sidebar under the heading **Bubble map**", unsafe_allow_html=True)

#Filter option for Bubble map
include_admin_scatter = st.sidebar.checkbox('Remove Top Admin Categories', key="include_admin_scatter")
num_range = st.sidebar.slider('Filter threshold', min_value=1, max_value=600, value=20)

st.plotly_chart(scatter_plot(unigram_data.copy(), include_top_cat=include_admin_scatter, min_value=num_range))

