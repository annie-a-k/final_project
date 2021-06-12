import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from IPython.display import HTML
import re
import geopandas as gpd
import random as rd
from selenium import webdriver
import os
from sklearn.linear_model import LinearRegression
import networkx as nx
from pyvis.network import Network

st.header("Сайт с заботой о вашем ментальном здоровье")
st.subheader("Почему это важно?")
st.write("Многие люди до сих пор продолжают игнорировать свои психологические и психические проблемы. Однако здоровье - это важно! Здесь вы сможете узнать больше о классификации психических расстройств и расстройств поведения по МКБ-10, об их распространённости, возможных прогнозах, а также получить страницу случайного психотерапевта с сайта Профессиональной Психотерапевтической Лиги.",
            "Как ориентироваться (для упрощения проверки): ",
            "1. Сначала идут все результаты, потом - весь код. Код программы разделён на подписанные смысловые блоки (с помощью #). ",
         "2. Чтобы сайт не грузился слишком долго, большая часть функционала подгружается последовательно при выборе пользователем соответствующей опции. ",
         "Например, чтобы посмотреть на несколько вариантов построения графиков, необходимо выбрать одну из опций (выбрать датафрейм, выбрать тип графика, выбрать данные). ",
         "Также можно добавить неограниченное количество визуализаций с любой информацией и типом графика из предложенных опций. ",
         "3. Как убедиться в том, что использовалось в программе? Смотрите на комментарии в коде в конце приложения. ",
         "4. Использовано: pandas (работаем с МКБ-10, объединяем; добавляем HTML для кликабельных ссылок; выбираем строчки по кодам заболеваний, записанным разными способами); веб-скреппинг через beautifulsoup и selenium плюс REST API (парсинг по МКБ-10, рандомный психотерапевт); ",
         "визуализация данных и геоданные (универсальные графики (для нескольких таблиц) двух типов с помощью библиотек matplotlib, geopandas); ",
         "streamlit (очевидно); регулярные выражения (выбор строк из МКБ-10, поиск мобильного телефона психотерапевта); ",
         "машинное обучение (регрессия с расстройствами пищевого поведения, сравнение качества предсказаний по разному количеству данных); граф (networkx). ")
st.write("Создаём dataframe с категориями психических расстройств и расстройств поведения по МКБ-10. По ссылочкам можно перейти в соответствующие разделы на сайте.")
with st.echo(code_location='below'):
    url = 'https://mkb-10.com/index.php?pid=4001'
    r = requests.get(url)
    soup = BeautifulSoup(r.text)
    heading = soup.find_all("h1")
    cod = soup.find_all("div", {"class": "code"})
    cat = soup.find_all("a", {"style": "font-weight:bold;"})
    codes = []
    for item in cod:
        codes.append(item.text)
    linkss = []
    for item in cat:
        linkss.append(f'<a href="{"https://mkb-10.com" + item.get("href")}">{item.text}</a>')
    d = {'Коды': codes, 'Категория': linkss}
    classification_categories = pd.DataFrame(data=d)
    table1 = st.slider('Сколько строк таблицы отобразить?', 1, len(classification_categories), 4, key=f"choosetablelength_{1}")
    classification_categories.index = classification_categories["Коды"]
    classification_categories = classification_categories.drop("Коды", axis=1)
    classification_categories_html=HTML(classification_categories.head(table1).to_html(escape=False))
    st.write(classification_categories_html)

    st.write("Парсим расстройства из категорий и объединяем их в одну табличку")
    classification = pd.DataFrame()
    for elem in cat:
        new = requests.get("https://mkb-10.com" + elem.get("href"))
        soup2 = BeautifulSoup(new.text)
        cat2 = soup2.find_all("div", {"class": "h2"})
        linkss2 = []
        codes2 = []
        for el in cat2:
            divide = el.text.index(" ")
            co = el.text[0: divide]
            cl = el.text[divide + 1:]
            try:
                link = el.find("a", {"style": "font-weight:bold;"}).get('href')
                codes2.append(co)
                linkss2.append(f'<a href="{"https://mkb-10.com" + link}">{cl}</a>')
            except AttributeError:
                codes2.append(co)
                linkss2.append(cl)
        d2 = {'Код': codes2, 'Расстройство': linkss2}
        classification_category = pd.DataFrame(data=d2)
        classification = pd.concat([classification, classification_category])
    table2 = st.slider('Сколько строк таблицы отобразить?', 1,
                       len(classification), 4, key=f"choosetablelength_{2}")
    classification.index = classification["Код"]
    classification = classification.drop("Код", axis=1)
    classification_html = HTML(classification.head(table2).to_html(escape=False))
    st.write(classification_html)

    st.write("Объединяем табличку с расстройствами и категориями по кодам")
    def pure_code(full_code):
        return (int(full_code.replace("F", "").replace("*", "")))

    def drop_extra_columns(df):
        df = df.drop(['Код'], axis=1)
        df = df.rename(columns={'Полный код': 'Код'})
        df.index = df["Код"]
        return (df.drop(['Код'], axis=1))

    temp = pd.DataFrame(columns=["Код", "Категория"])
    for index, element in classification_categories.reset_index().iterrows():
        group_of_codes = element["Коды"].replace("F", "").replace("*", "")
        divide_gr = group_of_codes.index("-")
        start_gr = group_of_codes[0: divide_gr]
        end_gr = group_of_codes[divide_gr + 1:]
        for i in range(int(end_gr) - int(start_gr) + 1):
            temp = temp.append({"Код": int(start_gr) + i, "Категория": element["Категория"]}, ignore_index=True)
    classification_temp = classification.reset_index().rename(columns={'Код': 'Полный код'})
    classification_temp["Код"] = classification_temp["Полный код"].apply(pure_code)
    full_classification = classification_temp.reset_index().merge(temp, how="left", on='Код')
    table3 = st.slider('Сколько строк таблицы отобразить?', 1,
                       len(full_classification), 4, key=f"choosetablelength_{3}")
    full_classification_for_search = full_classification.drop(["index"], axis=1)
    full_classification = drop_extra_columns(full_classification_for_search)
    full_classification_html = HTML(full_classification.head(table3).to_html(escape=False))
    st.write(full_classification_html)

    st.write("Here you can find information about disorders based on their' codes. ",
         "You can print full codes (for ex., F02*) or only numbers.",
         "Divide codes using space.",
         "If you want a series of codes you can print them via \"-\" (for ex., 15-F18).")
    find_code_raw = list(st.text_input("Write codes here:").split(" "))
    if not find_code_raw==[""]:
        find_code = []
        no_disorders = ""
        count_no = 0
        for element in find_code_raw:
            if "-" in element:
                divide_el = element.index("-")
                if not divide_el==0 and not divide_el==len(element)-1:
                    a = re.split(r'[^\d]', element[0: divide_el])
                    if "" in a:
                        a.remove("")
                    if not a == []:
                        start_el = int(a[0])
                    a = re.split(r'[^\d]', element[divide_el + 1:])
                    if "" in a:
                        a.remove("")
                    if not a == []:
                        end_el = int(a[0])
                    for i in range(end_el - start_el + 1):
                        find_code.append(start_el + i)
            else:
                a = re.split(r'[^\d]', element)
                if "" in a:
                    a.remove("")
                if not a==[]:
                    find_code.append(int(a[0]))
        find_code_final=[]
        for element in find_code:
            if element not in list(full_classification_for_search["Код"]):
                no_disorders = no_disorders + str(element) + ", "
                count_no = count_no + 1
            else:
                find_code_final.append(element)
        if not len(find_code_final) == 0:
            find_classification = full_classification_for_search[
                full_classification_for_search['Код'].astype("int").isin(find_code_final)]
            find_classification = drop_extra_columns(find_classification)
            find_classification_html = HTML(find_classification.to_html(escape=False))
            st.write(find_classification_html)
        if not no_disorders == "" and count_no > 1:
            st.write("Расстройств с кодами", no_disorders[:-2], "не существует.")
        if not no_disorders == "" and count_no == 1:
            st.write("Расстройства с кодом", no_disorders[:-2], "не существует.")
        st.write("Мы смогли выбрать данные из таблички!")
    else:
        st.write("Вы не выбрали ни одного кода.")

    st.write("Info is from https://ourworldindata.org/mental-health")
    st.write("Mental and substance use disorders are common globally. In the map we see that globally, mental and substance use disorders are very common: around 1-in-7 people (15%) have one or more mental or substance use disorders. (c) Our world in data")
    share_with_disorders=pd.read_csv('share-with-mental-and-substance-disorders.csv')
    #st.write(share_with_disorders)
    world_share_with_disorders=share_with_disorders[share_with_disorders["Entity"]=="World"]
    #st.write(world_share_with_disorders)
    prevalence_by_disorder=pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
    #st.write(prevalence_by_disorder)
    share_by_gender=pd.read_csv("share-with-mental-or-substance-disorders-by-sex.csv")
    #st.write(share_by_gender)
    share_of_all_diseases=pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
    #st.write(share_of_all_diseases)
    share_with_an_eating_disorder=pd.read_csv("share-with-an-eating-disorder.csv")
    #st.write(share_with_an_eating_disorder)
    prevalence_of_eating_disorders_by_gender=pd.read_csv("prevalence-of-eating-disorders-in-males-vs-females.csv")
    #st.write(prevalence_of_eating_disorders_by_gender)
    deaths_from_eating_disorders=pd.read_csv("deaths-from-eating-disorders.csv")
    #st.write(deaths_from_eating_disorders)

    share_with_depression = pd.read_csv("share-with-depression.csv")
    #st.write(share_with_depression)
    prevalence_of_depression_by_gender = pd.read_csv("prevalence-of-depression-males-vs-females.csv")
    #st.write(prevalence_of_depression_by_gender)
    share_with_anxiety_disorders = pd.read_csv("share-with-anxiety-disorders.csv")
    #st.write(share_with_anxiety_disorders)
    prevalence_of_anxiety_by_gender = pd.read_csv("prevalence-of-anxiety-disorders-males-vs-females.csv")
    #st.write(prevalence_of_anxiety_by_gender)



    function_country = []
    function_yearmin = []
    function_yearmax = []
    choose_type = []
    function_plus_field = []
    function_show_df = []
    function_choose_df = []
    function_all_or_some = []
    i = 0
    function_plus_field_to_func = {
        'Yes :)': 'yes',
        'No, I am tired': 'no'
    }
    function_choose_type_to_func={
        'График, оси XY': '1',
        'Карта': '2'
    }
    function_show_df_to_func = {
        'Yes!': 'yes',
        'O.o No...': 'no'
    }
    function_choose_df_to_func = {
        'Доля населения с психическими расстройствами и расстройствами поведения по странам': share_with_disorders,
        "Мир. Доля населения с психическими расстройствами и расстройствами поведения": world_share_with_disorders,
        "Распространённость расстройств в зависимости от пола": share_by_gender,
        "Доля психических расстройств и расстройств поведения среди всех заболеваний": share_of_all_diseases,
        "Доля населения с расстройствами пищевого поведения": share_with_an_eating_disorder,
        "Распространённость расстройств пищевого поведения в зависимости от пола": prevalence_of_eating_disorders_by_gender,
        'Смертность от расстройств пищевого поведения': deaths_from_eating_disorders,
        "Доля населения с депрессией": share_with_depression,
        "Распространённость депрессии в зависимости от пола": prevalence_of_depression_by_gender,
        "Доля населения с тревожным расстройством": share_with_anxiety_disorders,
        "Распространённость тревожного расстройства в зависимости от пола": prevalence_of_anxiety_by_gender
    }

    def plotType1(fr):
        fig, ax = plt.subplots()
        for country in fr['Entity'].unique():
            countryfr = fr.loc[fr['Entity'] == country]
            countryfr.plot(y=fr.columns.values.tolist()[3], x='Year', ax=ax, xlabel='Year', ylabel='Prevalence', label=country)
        st.pyplot(fig)

    def plotType2(fr):
        shapefile = 'ne_110m_admin_0_countries.shp'
        gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
        gdf.columns = ['country', 'country_code', 'geometry']
        data = fr
        merged = gdf.merge(data, left_on='country_code', right_on='Code', how='left')
        merged=merged.drop(["Code", "Year", "country", "country_code", "Entity"], axis=1)
        merged=merged.dropna(axis=0, subset=[merged.columns.values.tolist()[1]]).assign(
            prevalence=lambda x: x[merged.columns.values.tolist()[1]].astype("int64"))
        st.write(merged.head())
        fig, ax = plt.subplots()
        merged.plot(ax=ax, column="prevalence", legend=True)
        st.pyplot(fig)

    def any_graph(i, df):
        st.sidebar.write(f'Graph №{i+1}')
        st.write(f'Graph №{i+1}')
        st.write('There you can choose, what data to visualize and how to do it.'
                 ' Remember that there will not be correct visualization if you choose non-existing combination of parameters.')
        st.write("Стандартный график поддерживает выбор отдельных регионов. Для карты лучше выбрать все имеющиеся данные.")
        function_country.append(' ')
        function_yearmin.append(' ')
        function_yearmax.append(' ')
        function_all_or_some.append(" ")
        choose_type.append(' ')
        choose_type[i] = st.sidebar.selectbox('What type of graph do you want?', ('График, оси XY', 'Карта'),
                                              key=f"choosetype_{i}")
        if function_choose_type_to_func[choose_type[i]] == '1':
            function_country[i]=st.sidebar.multiselect('Choose regions', df['Entity'].unique(), key=f"chooseregion_{i}")
            function_yearmin[i] = st.sidebar.slider('What year you want to start from?', 1990, 2017, 1990, 1,
                                                    key=f"choosemin_{i}")
            function_yearmax[i] = st.sidebar.slider('What year you want to end with?', function_yearmin[i], 2017, 2017,
                                                    1, key=f"choosemax_{i}")
            year = list(range(function_yearmin[i], function_yearmax[i] + 1))
        if function_choose_type_to_func[choose_type[i]] == '2':
            function_country[i]=df['Entity'].unique()
            function_all_or_some[i]=st.sidebar.slider('Выберите год', 1990, 2017, 1990, 1,
                                                    key=f"choosemin_{i}")
            year=[function_all_or_some[i]]
        df=df.dropna()
        ndf0 = df.loc[df['Entity'].isin(function_country[i])]
        ndf0['Year'] = pd.to_numeric(ndf0['Year'])
        ndf = ndf0.loc[ndf0['Year'].isin(year)]
        function_show_df.append(' ')
        function_show_df[i + 1]=st.selectbox('Do you want to show the dataframe?', ('Yes!', 'O.o No...'), key=f"chooseshow_{i+1}")
        if function_show_df_to_func[function_show_df[i + 1]] == 'yes':
            st.write(ndf)
        if function_choose_type_to_func[choose_type[i]] == '1':
            plotType1(ndf)
        if function_choose_type_to_func[choose_type[i]] == '2':
            plotType2(ndf)
        i = i + 1
        function_plus_field.append(' ')
        function_plus_field[i-1]=st.selectbox('Do you want one more graph?', ('Yes :)', 'No, I am tired'), index=1, key=f"choosefield_{i-1}")
        if function_plus_field_to_func[function_plus_field[i-1]] == 'no':
            st.write("Thank you for using me! (c) The app")
        if function_plus_field_to_func[function_plus_field[i-1]] == 'yes':
            function_choose_df.append(' ')
            function_choose_df[i] = st.selectbox('What dataframe to work with?', ('Доля населения с психическими расстройствами и расстройствами поведения по странам',
        "Мир. Доля населения с психическими расстройствами и расстройствами поведения",
        "Распространённость расстройств в зависимости от пола",
        "Доля психических расстройств и расстройств поведения среди всех заболеваний",
        "Доля населения с расстройствами пищевого поведения",
        "Распространённость расстройств пищевого поведения в зависимости от пола",
        'Смертность от расстройств пищевого поведения',
        "Доля населения с депрессией",
        "Распространённость депрессии в зависимости от пола",
        "Доля населения с тревожным расстройством",
        "Распространённость тревожного расстройства в зависимости от пола"),
                                                 key=f"choosdf_{i}")
            df = function_choose_df_to_func[function_choose_df[i]]
            any_graph(i, df)

    function_show_df.append(' ')
    function_choose_df.append(' ')
    function_choose_df[0] = st.selectbox('What dataframe to work with?', ('Доля населения с психическими расстройствами и расстройствами поведения по странам',
        "Мир. Доля населения с психическими расстройствами и расстройствами поведения",
        "Распространённость расстройств в зависимости от пола",
        "Доля психических расстройств и расстройств поведения среди всех заболеваний",
        "Доля населения с расстройствами пищевого поведения",
        "Распространённость расстройств пищевого поведения в зависимости от пола",
        'Смертность от расстройств пищевого поведения',
        "Доля населения с депрессией",
        "Распространённость депрессии в зависимости от пола",
        "Доля населения с тревожным расстройством",
        "Распространённость тревожного расстройства в зависимости от пола"),
                                       key=f"chooseshow_{0}")
    df=function_choose_df_to_func[function_choose_df[0]]
    function_show_df[0] = st.selectbox('Do you want to show the original dataframe?', ('Yes!', 'O.o No...'), index=1, key=f"choosdf_{0}")
    if function_show_df_to_func[function_show_df[0]]=='yes':
        st.write(df)
    any_graph(i, df)



    st.write("Давайте построим простую предсказательную модель зависимости смертей по причине расстройств пищевого поведения от распространённости этих расстройств и от года. ",
             "Возьмём общемировую статистику и воспользуемся линейной регрессией.")
    ##Машинное обучение сделано с опорой на конспект лекции от 20 апреля 2021 года.
    df01_new = share_with_an_eating_disorder
    df02_new = deaths_from_eating_disorders
    df01_new = df01_new[df01_new["Entity"] == "World"].drop(["Code", "Entity"], axis=1)
    df02_new = df02_new[df02_new["Entity"] == "World"].drop(["Code", "Entity"], axis=1)
    df_new = df02_new.merge(df01_new, left_on='Year', right_on="Year", how='inner')
    st.write(df_new)
    regr = LinearRegression()
    X = df_new[['Year', df_new.columns.values.tolist()[2]]]
    y = df_new[df_new.columns.values.tolist()[1]]
    regr.fit(X, y)
    st.write("regr.coef_: ", regr.coef_, " regr.intercept_: ", regr.intercept_)
    #df_new = df_new.assign(year=lambda x: x[df_new.columns.values.tolist()[0]].astype("int64"))
    #fig, ax = plt.subplots()
    #df_new.plot.scatter("Year", df_new.columns.values.tolist()[1])
    #df_new.plot(x=df_new["Year"], y=regr.predict(X), color='C1')
    #st.pyplot(fig)
    #st.mpl_fig(fig)

    st.write("Проверим, действительно ли модель улучшается, когда предсказывает по двум параметрам, а не по одному.")
    df_new = df_new.sample(frac=1, random_state=1)
    train = df_new[:int(df_new.shape[0] * 0.7)]
    test = df_new[int(df_new.shape[0] * 0.7):]
    def get_RSS(Features, estimator):
        X_train = train[Features]
        y_train = train[df_new.columns.values.tolist()[1]]

        X_test = test[Features]
        y_test = test[df_new.columns.values.tolist()[1]]

        estimator.fit(X_train, y_train)

        def rss(y, y_hat):
            return ((y - y_hat) ** 2).sum()

        return rss(estimator.predict(X_test), y_test)
    st.write("Среднеквадратичная ошибка при использовании для предсказания только года:", get_RSS([df_new.columns.values.tolist()[0]], regr))
    st.write("Среднеквадратичная ошибка при использовании для предсказания только распространённости расстройств:", get_RSS([df_new.columns.values.tolist()[2]], regr))
    st.write("Среднеквадратичная ошибка при использовании для предсказания обоих параметров", get_RSS(["Year", df_new.columns.values.tolist()[2]], regr))
    st.write("Следовательно, предсказание на основе двух параметров наиболее эффективное. Как можно заметить, с течением времени смертность от расстройств пищевого поведения увеличивается.")




    st.write("А это граф, визуализирующий наличие корреляции между повышенной распространённостью различных расстройств (выше среднемирового уровня)")
    frrame = pd.read_csv('for_graf_by_site.csv')
    frrame = frrame.dropna(how='all')
    frrame = frrame[frrame['Year']== 2017]
    graph = nx.DiGraph()
    for _, row in frrame.iterrows():
        graph.add_edge(row['Unnamed: 2'], row["Unnamed: 3"])
    plt.figure(figsize=(10, 8))
    graph.remove_node("Nothing")
    #net = Network(width='800px', notebook=True)
    #net.from_nx(graph)
    fig=nx.draw_shell(graph, with_labels=True)
    #st.write(net.show("visualization.html"))
    st.pyplot(fig)




    st.write(
        'Если вы хотите обратиться за помощью к специалисту, но не знаете к кому, можете воспользоваться этим рандомайзером.',
        "Он выдаёт ссылку на страницу одного из специалистов Профессиональной Психотерапевтической Лиги и (при наличии) выписывает первый указанный им номер мобильного телефона для связи.")

    # Функция, которая позволяет перейти на страницу со случайным психотерапевтом
    def get_ps(name):

##Начало позаимствованного кода. Источник: https://www.andressevilla.com/running-chromedriver-with-python-selenium-on-heroku/
        chrome_options = webdriver.ChromeOptions()
        chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), chrome_options=chrome_options)
##Конец позаимствованного кода.

        entrypoint = str("https://oppl.ru/professionalyi-personalii/" + name + ".html")
        driver.get(entrypoint)
        element = driver.find_element_by_xpath("/html/body/div[6]/div[1]/div[3]")
        st.write(entrypoint)
        numbers = re.search(r'(\+)?[\+7,8](.)?(\()?\d{3,3}(\))?(.)?\d{3,3}((-)?\d{2,2}){2,2}', element.text)
        if not numbers==None:
            st.write(numbers.group(0))
        else:
            st.write("Телефонные номера не найдены")
        driver.get(entrypoint)


    # Парсим список ссылок на психотерапевтов с сайта

    want_consult=st.selectbox("Подобрать специалиста?", ("Нет", "Да"))
    if want_consult=="Да":
        URL1 = "https://oppl.ru/cat/professionalyi-personalii.html"
        res = requests.get(URL1)
        list_ps = []
        raw_material = BeautifulSoup(res.content, 'html.parser')
        bad_list = raw_material.find_all("a", {"class": "category sm"})
        for i in bad_list:
            url2 = i.get("href")
            list_ps.append(url2[27:-5])
        our_ps = rd.choice(list_ps)
        get_ps(our_ps)
        st.balloons()
    st.write("Хорошего вам дня! Заботьтесь о своём здоровье!")





