from numpy.core.arrayprint import dtype_is_implied
import streamlit as st
import pandas as pd
import numpy as np
from plotly.tools import FigureFactory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import math
from PIL import Image

st.set_page_config(page_title="IAS Dataset Analysis", page_icon="📚", layout="wide")
#This streamlit library function sets the page title as evident on the tab where the application is running and the favicon



#image=Image.open(r'/home/srishti/Downloads/ias-officers/logo.png')


# change this in the final build

image=Image.open("./logo.png")
logo=st.container()
logo_col1, logo_col2=logo.columns([7,1])
logo_col1.header('IAS Subject and Experience Analysis Tool')
logo_col2.image(image)


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
    df = pd.read_csv('.analysis/processed/unigram_relations.csv')
    #Dropping rows where the experience is NA
    df = df.loc[df["Experience"] != "N.A."]
    
    #Adding a new column that contains the count value as string, for example 'size:count' 
    df["text"] = df["Count"].apply(lambda x: "size: "+str(x))
    
    #Adding a percentage column per subject
    df["p_Subject"] = [0] * df.shape[0]
    df["p_Subject_text"] = [""] * df.shape[0]
    for subject in df["Subject"].unique():
        sum_subject_count = df.loc[df["Subject"] == subject]["Count"].sum()
        df.loc[df["Subject"] == subject, "p_Subject"] = df.loc[df["Subject"] == subject]["Count"].apply(lambda x: round(((x / sum_subject_count) * 100), 2))
        df.loc[df["Subject"] == subject, "p_Subject_text"] = df.loc[df["Subject"] == subject]["p_Subject"].apply(lambda x: str(x) + f"% of all {subject}")


    df["p_Experience"] = [0] * df.shape[0]
    df["p_Experience_text"] = [""] * df.shape[0]
    for experience in df["Experience"].unique():
        sum_experience_count = df.loc[df["Experience"] == experience]["Count"].sum()
        df.loc[df["Experience"] == experience, "p_Experience"] = df.loc[df["Experience"] == experience]["Count"].apply(lambda x: round(((x / sum_experience_count) * 100), 2))
        df.loc[df["Experience"] == experience, "p_Experience_text"] = df.loc[df["Experience"] == experience]["p_Experience"].apply(lambda x: str(x) + f"% of all {experience}")

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
        df = df.drop(columns=["p_Subject", "p_Subject_text", "p_Experience", "p_Experience_text"])
    except:
        pass

    try:
        return df.drop(columns=["text", "colors"])
    except:
        return df


def scatter_plot(df, include_top_cat=True, min_value=50, filter_subject_list=[], filter_experience_list=[], invert_axis=False):
    scale_factor_y = 25
    bubble_scale = 1
    bubble_size_max = 30


    if include_top_cat:
        df = df.loc[df["Experience"] != 'Land Revenue Mgmt & District Admn']
        df = df.loc[df["Experience"] != 'Personnel and General Administration']
        df = df.loc[df["Experience"] != 'Finance']

    if len(filter_subject_list) > 0:
        new_temp = pd.DataFrame()
        for subject in filter_subject_list:
            new_df = df.loc[df["Subject"] == subject]
            new_temp = new_temp.append(new_df)
    
        df = new_temp

    if len(filter_experience_list) > 0:
        new_temp = pd.DataFrame()
        for experience in filter_experience_list:
            new_df = df.loc[df["Experience"] == experience]
            new_temp = new_temp.append(new_df)
    
        df = new_temp

    df = df.sort_values(["Count"], ascending=False)

    df = df.loc[df["Count"] > min_value]

    length_y_axis = df["Subject"].unique().shape[0]
    length_x_axis = df["Experience"].unique().shape[0]

    height = 400 + (length_y_axis * scale_factor_y)

    x = df["Experience"].to_list()
    y = df["Subject"].to_list()
    text = df["p_Subject_text"].to_list()

    max_value = df["p_Subject"].max()
    relevant_list = df["p_Subject"].copy()

    if invert_axis:
        x, y = y, x
        text = df["p_Experience_text"].to_list()
        max_value = df["p_Experience"].max()
        relevant_list = df["p_Experience"].copy()

    bubble_scale = (bubble_size_max / max_value)

    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        text=text,
        mode="markers",
        marker=dict(
            size=relevant_list.apply(lambda x: x*bubble_scale).to_list(),#apply(lambda x: x*50/max_value).to_list(),
            color = df["colors"].to_list(),
        )
    )])

    if not invert_axis:
        fig = fig.update_layout(
            autosize=True,
            height=height,
            width=1500,
        )
    else:
        fig = fig.update_layout(
            autosize=True,
            width=1000,
            height=1200,
        )

    # remove grid lines from the figure
    if not invert_axis:
        fig.update_xaxes(
            automargin=False,
            autorange=False,
            range=[-2, length_x_axis],
            dtick=1,
            tick0=0,
            showgrid=False,
            type="category",
        )
        fig.update_yaxes(showgrid=False)
    else:
        fig.update_xaxes(
            showgrid=False,
        )
        fig.update_yaxes(
            showgrid=False,
        )
    
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
            'yanchor':'top'}
    )
        
    return fig.update_traces(
        hoverinfo="label+percent",
        textinfo="percent",
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


with st.expander("Getting Started and Dependencies"):

    st.markdown("Firslty, Streamlit library will have to be installed, steps to install are available [here](https://docs.streamlit.io/library/get-started/installation). This step is required even if you want to build the application locally/server on your machine")
    st.write("\n")
    st.markdown("To host the application online from Github (database at Github) an application dependencies file such as Pipfile, environment.yml, requirements.txt or pyproject.toml will be required by Streamlit Cloud to download the right packages during hosting application online. More about it [here](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/app-dependencies), also refer Pipfile at the **Github repo** ")
    body='''import streamlit as st
        import pandas as pd
        import numpy as np
        from plotly.tools import FigureFactory as ff
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        import plotly.express as px
        ''' 
    st.code(body, language = 'python')
    
    
with st.expander('Page configuration'):
    body='''#This streamlit library function sets the page title as evident on the tab where the application is running and the favicon
st.set_page_config(page_title="IAS Dataset Analysis", page_icon="📚", layout="centered")


    '''
    st.code(body, language = 'python')
    

st.subheader('Unigram maps')

body_unigram_maps='This table demonstrates the count value for each pair of Subject and Experience and acts as source.'
st.markdown(body_unigram_maps, unsafe_allow_html=True)

st.write(unigram_data.drop(columns=["text", "colors","p_Subject", "p_Subject_text", "p_Experience", "p_Experience_text"]))
with st.expander("Code Block for Unigram maps table rendered above"):
    body='''
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
    df = pd.read_csv('~/Downloads/ias-officers/analysis/processed/unigram maps.csv')
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
    
#the library function that renders the dataframe, text and colors are temporary variable added to dataframe for providing color gradient to the bubbles in the bubble chart
st.write(unigram_data.drop(columns=["text", "colors"]))
    '''
    st.code(body, language = 'python')
  
  
  
   
#Add a horizontal bar
st.markdown("""<hr/>""", unsafe_allow_html=True)


#Add sub heading on the sidebar
st.sidebar.subheader('Select a Category of Experience')
option_experience = st.sidebar.selectbox("", unigram_data['Experience'].unique())
# include_other_experience = st.sidebar.checkbox("Include combined remaining entries")
number_of_rows_experience = 8 #st.sidebar.slider('Number of rows', min_value=1, max_value=8, value=5)
st.sidebar.markdown("""<hr/>""", unsafe_allow_html=True)

#Add sub heading on the main page
st.subheader('Subject occurances for the Category of Experience')
st.markdown("Choose the appropiate filter on the sidebar at left under the **Select a Category of Experience**",unsafe_allow_html=True)
st.write("\n")

#Filtering dataframe based on the filters selected on the sidebar
filtered_df_experience = filter_by_value(unigram_data.copy(), 'Experience', option_experience, number_of_rows=number_of_rows_experience)
st.write(filtered_df_experience)


with st.expander("Sidebar filter and Dataframe Filtering for 'Category of Experience'"):

    st.markdown("Below code explains how the filters on the sidebar are placed. Values selected in the filter are then used to subset the dataframe in the table above", unsafe_allow_html=True)
    experience_filter='''
#Add sub heading on the sidebar, refering the 1st heading
st.sidebar.subheader('Select a category of experience')

#Options to filter data
option_experience = st.sidebar.selectbox("", unigram_data['Experience'].unique())
include_other_experience = st.sidebar.checkbox("Include combined remaining entries")
number_of_rows_experience = st.sidebar.slider('Number of row	s', min_value=1, max_value=8, value=5)

#Filtering dataframe based on the filters selected on the sidebar
filtered_df_experience = filter_by_value(unigram_data.copy(), 'Experience', option_experience, number_of_rows=number_of_rows_experience, include_other=include_other_experience)
st.write(filtered_df_experience)
    '''
    st.code(experience_filter, language = 'python')
    
    st.markdown("Below code explains filter_by_value function, which is used to subset the main unigram data. Functions should come before the call to the function, however for understanding purpose it is placed below. Refer the **github script** for more.", unsafe_allow_html=True)
    
    filter_function='''
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
        
    '''
    st.code(filter_function, language = 'python')
            

#Utilising Streamlit container function to make a grid and place Bar and Pie Chart side-by-side
plot_container_experience = st.container()

col1, col2 = plot_container_experience.columns([2, 1])
col1.plotly_chart(bar_chart(filtered_df_experience, title="Number of Subject occurances with respect to chosen Category of Experience", x_axis_title="Subjects"))
col2.plotly_chart(pie_chart(filtered_df_experience))

with st.expander("Placing charts side-by-side"):
    container_code='''
#Utilising Streamlit container function to make a grid and place Bar and Pie Chart side-by-side
plot_container_experience = st.container()

#In the container the area is divided into two columns, the ratio for which is 7:1
col1, col2 = plot_container_experience.columns([7, 1])

#Bar Plot in Column 1
col1.plotly_chart(bar_chart(filtered_df_experience, title="Number of Subject occurances with respect to chosen Category of Experience", x_axis_title="Subjects"))

#Pie Chart in Column 2
col2.plotly_chart(pie_chart(filtered_df_experience))

    '''
    
    st.code(container_code,language = 'python')

with st.expander("Bar Chart for Category of Experience"):

    st.markdown("Function call to bar_chart function with the filtered dataframe", unsafe_allow_html=True)
    
    experience_bar_code='''
col1.plotly_chart(bar_chart(filtered_df_experience, title="Number of Subject occurances with respect to chosen Category of Experience", x_axis_title="Subjects"))
    '''
    st.code(experience_bar_code, language = 'python')
    
    st.markdown("Below code explains the bar_chart(). Function definition should come before call to the function, however only for understanding purpose it is placed below. Refer the **github script** for more.", unsafe_allow_html=True)
    
    bar_chart_code='''
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
    '''
    st.code(bar_chart_code, language = 'python')
    
    


st.markdown("""<hr/>""", unsafe_allow_html=True)


#Filter options for the second sub heading 
st.sidebar.subheader('Select a Subject')
option_subject = st.sidebar.selectbox("", unigram_data['Subject'].unique())
include_admin = st.sidebar.checkbox('Remove Top Admin Categories')
include_other_subject = st.sidebar.checkbox("Include combined remaining entries", key="subjects")
# percentage = st.sidebar.checkbox('Show percentage', value=True)
number_of_rows = st.sidebar.slider('Number of rows', min_value=1, max_value=47, value=5)


st.subheader('Category of Experience occurances with respect to Subject')
st.markdown("Choose the appropiate filter on the sidebar at left under the **Select a Subject**",unsafe_allow_html=True)
st.write("\n")
filtered_df_subject = filter_by_value(unigram_data.copy(), 'Subject', option_subject, include_admin=include_admin, number_of_rows=number_of_rows, include_other=include_other_subject)
st.write(filtered_df_subject)
with st.expander("Sidebar Filter and Dataframe Filtering for Subject"):

    st.markdown("Below code explains how the filters on the sidebar are placed. Values selected in the filter are then used to subset the dataframe in the table above", unsafe_allow_html=True)
    subject_filter='''
#Filter options for the second sub heading 
st.sidebar.subheader('Select a Subject')
option_subject = st.sidebar.selectbox("", unigram_data['Subject'].unique())
include_admin = st.sidebar.checkbox('Remove Top Admin Categories')
percentage = st.sidebar.checkbox('Show percentage', value=True)
number_of_rows = st.sidebar.slider('Number of rows', min_value=1, max_value=47, value=5)

#utilising the filtered value for subsetting the dataset
filtered_df_subject = filter_by_value(unigram_data.copy(), 'Subject', option_subject, include_admin=include_admin, number_of_rows=number_of_rows, percentage=percentage)
st.write(filtered_df_subject)


    '''
    st.code(subject_filter, language = 'python')
    
    st.markdown("Refer the second code snippet under the expanding section **Dataframe filtering for Category of Experience** for filter_by_value function code.", unsafe_allow_html=True)
    
    
    
#Containers in streamlit provide functionality to divide the area into columns and control the content to be placed in each column 
plot_container_subject = st.container()
col3, col4= plot_container_subject.columns([2,1])

col3.plotly_chart(bar_chart(filtered_df_subject, title="Number of Category of Experience occurances with respect to chosen Subject", x_axis_title="Category of Experience"))
col4.plotly_chart(pie_chart(filtered_df_subject))


with st.expander("Pie Chart for Subject"):

    st.markdown("Function call for pie_chart() for rendering Pie Chart on the filtered dataframe. Refer expandible section **Placing charts side-by-side** for placing bar and pie charts horizontally, as above. Similarly, a new container is declared and 2 columns made in the container named as Col3 and Col4.",unsafe_allow_html=True)
    pie_code='''
col4.plotly_chart(pie_chart(filtered_df_subject))
    '''
    st.code(pie_code, language = 'python')
    
    st.markdown("Below code explains the function pie_chart(). Function definition should come before function call, refer **github script** for more",unsafe_allow_html=True)
    pie_subject_code='''
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
    
    '''
    st.code(pie_subject_code, language = 'python')

    

st.sidebar.markdown("""<hr/>""", unsafe_allow_html=True)
st.markdown("""<hr/>""", unsafe_allow_html=True)

st.sidebar.subheader('Bubble map')
st.subheader('Bubble map')
st.markdown("Choose the appropiate filter on the sidebar under the heading **Bubble map**", unsafe_allow_html=True)

#Filter option for Bubble map
filter_subject_list = st.sidebar.multiselect('Subjects', unigram_data['Subject'].unique())
filter_experience_list = st.sidebar.multiselect('Category of Experience', unigram_data['Experience'].unique())
include_admin_scatter = st.sidebar.checkbox('Remove Top Admin Categories', key="include_admin_scatter", value=True)
num_range = st.sidebar.slider('Filter threshold', min_value=1, max_value=600, value=20)

st.plotly_chart(scatter_plot(unigram_data.copy(), include_top_cat=include_admin_scatter, min_value=num_range, filter_subject_list=filter_subject_list, filter_experience_list=filter_experience_list))
st.plotly_chart(scatter_plot(unigram_data.copy(), include_top_cat=include_admin_scatter, min_value=num_range, filter_subject_list=filter_subject_list, filter_experience_list=filter_experience_list, invert_axis=True))

with st.expander("Sidebar Filter and Bubble Map"):


    st.markdown("Function call to scatter_plot() for rendering the Bubble map above",unsafe_allow_html=True)
    
    bubble_call='''
st.plotly_chart(scatter_plot(unigram_data.copy(), include_top_cat=include_admin_scatter, min_value=num_range))
    '''
    st.code(bubble_call, language='python')
    
    st.markdown("Below code block explains the scatter_plot() on the filtered dataframe",unsafe_allow_html=True)
    bubble_map_code='''
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
    '''
    st.code(bubble_map_code, language = 'python')

