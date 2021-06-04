import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


with st.echo(code_location='below'):

    df = pd.read_csv('gdp_csv.csv')
    function_country = []
    function_yearmin = []
    function_yearmax = []
    choose_type = []
    function_plus_field = []
    function_show_df = []
    i = 0
    function_plus_field_to_func = {
        'Yes :)': 'yes',
        'No, I am tired': 'no'
    }
    choose_type_to_func = {
        'Annual GDP': 'GDP',
        'Annual GDP2': 'GDP2'
    }
    function_show_df_to_func = {
        'Yes!': 'yes',
        'O.o No...': 'no'
    }

    def plotGDP(fr):
        fig, ax = plt.subplots()
        for country in fr['Country Name'].unique():
            countryfr = fr.loc[fr['Country Name'] == country]
            countryfr.plot(y='Value', x='Year', ax=ax, xlabel='Year', ylabel='GDP', label=country)
        st.pyplot(fig)

    def any_graph(i):
        st.sidebar.write(f'Graph №{i+1}')
        st.write(f'Graph №{i+1}')
        st.write('There you can choose, what data to visualize and how to do it.'
                 ' Remember that there will not be correct visualization if you choose non-existing combination of parameters.')
        function_country.append(' ')
        function_yearmin.append(' ')
        function_yearmax.append(' ')
        function_country[i]=st.sidebar.multiselect('Choose regions', df['Country Name'].unique(), key=f"chooseregion_{i}")
        function_yearmin[i]=st.sidebar.slider('What year you want to start from?', 1960, 2016, 1960, 1, key=f"choosemin_{i}")
        function_yearmax[i]=st.sidebar.slider('What year you want to end with?', function_yearmin[i], 2016, 2016, 1, key=f"choosemax_{i}")
        year = list(range(function_yearmin[i], function_yearmax[i] + 1))
        ndf0 = df.loc[df['Country Name'].isin(function_country[i])]
        ndf0['Year'] = pd.to_numeric(ndf0['Year'])
        ndf1 = ndf0.loc[ndf0['Year'].isin(year)]
        ndf = ndf1.dropna()
        function_show_df.append(' ')
        function_show_df[i + 1]=st.selectbox('Do you want to show the dataframe?', ('Yes!', 'O.o No...'), key=f"chooseshow_{i+1}")
        if function_show_df_to_func[function_show_df[i + 1]] == 'yes':
            st.write(ndf)
        choose_type.append(' ')
        choose_type[i]=st.sidebar.selectbox('What type of graph do you want?', ('Annual GDP', 'Annual GDP2'), key=f"choosetype_{i}")
        if choose_type_to_func[choose_type[i]] == 'GDP':
            plotGDP(ndf)
        if choose_type_to_func[choose_type[i]] == 'GDP2':
            plotGDP(ndf)
        i = i + 1
        function_plus_field.append(' ')
        function_plus_field[i-1]=st.selectbox('Do you want one more graph?', ('Yes :)', 'No, I am tired'), index=1, key=f"choosefield_{i-1}")
        if function_plus_field_to_func[function_plus_field[i-1]] == 'no':
            st.write("Thank you for using me! (c) The app")
        if function_plus_field_to_func[function_plus_field[i-1]] == 'yes':
            any_graph(i)

    st.title("Welcome to GDP visualizer!!")
    st.write("In this app you can visualize data on country, regional and world GDP from 1960 to 2016 year."
             " The data is taken from Kaggle, where it is sourced from the World Bank."
             " You can find the source on Kaggle here: https://www.kaggle.com/tunguz/country-regional-and-world-gdp")
    function_show_df.append(' ')
    function_show_df[0] = st.selectbox('Do you want to show the original dataframe?', ('Yes!', 'O.o No...'), key=f"chooseshow_{0}")
    if function_show_df_to_func[function_show_df[0]]=='yes':
        st.write(df)
    any_graph(i)