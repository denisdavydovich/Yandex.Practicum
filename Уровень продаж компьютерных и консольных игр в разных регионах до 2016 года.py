#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success">
# <font size="5", color= "seagreen"><b>✔️ Комментарий ревьюера в4</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Привет, Денис :)  Спасибо за работу, все исправления и комментарии
#         
# На проекте мы изучали и практиковали навыки из разделов о/об:
#                   
#  + выдвижении гипотез о пропусках в данных, заполнении или удалении,
#  + работе с категориальными данными
#  + подготовке данных для решения главной задачи проекта
#  + исследовании неочищенных данных,        
#  + подборе и создании графиков,
#  + программировании,
#  + автоматизации однотипных действий,
#  + выстраивании структуры проекта и обеспечении аккуратности кода,
#  + анализе данных,
#  + определении прибыльности и устойчивости продаж по выбранному параметру (жанр, платформа) на диаграмме размаха,
#  + составлении портрета пользователя, 
#  + формулировании и проверке двухсторонних гипотез, интерпретации значения p-value - можно повторить недели через две </b>(для профилактики :) )
#  + формировании рекомендаций для бизнеса 
#         
# 
# 
# <b>Поздравляю с завершением первого модуля на факультете дата-аналитики Я.Практикум</b>
# 
# <div class="alert alert-success">
#     <font size="5", color = "seagreen"><b>Успехов в дальнейшей учебе 🤝</b></font><br />
#     
# добавил бонус

# <div class="alert alert-success">
# <font size="5", color= "seagreen"><b>✔️ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />   
# Привет, Денис :)  Стоит оставить свои вопросы по проекту, если имеется недопонимание, каким образом исправлять комментарии, так я смогу более подробно объяснить, что стоит исправить
#         
# Критические ❌ комментарии связаны с неточностями: 
# 
#  + обработка пропусков в столбце рейтингов ESRB
#  + переопределить перечень перспективных платформ
#  + оценить прибыльность жанров на диаграмме размаха, на актуальной выборке, обновить вывод
#  + перестроить графики в ТОП-5 и пересмотреть раздел рейтингов
#  + в разделе проверки гипотез можно более подробно расшифровать значение p_value 
#  + перепроверить промежуточные и итоговый выводы после всех исправлений
# 
# Стоит обратить внимание на ⚠️ комментарии...        
#         
# Если будут вопросы про мои комментарии - задавай, если какой-то формат взаимодействия не устраивает или есть какие-то другие пожелания - пиши :)
# 
# <div class="alert alert-success">
#     <font size="5", color= "seagreen"><b>Жду твой проект и твои комментарии 🤝</b></font><br />

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# 
# Исправил: 
# 
#  + обработка пропусков в столбце рейтингов ESRB - строка 28
#  + переопределить перечень перспективных платформ - строка 49
#  + оценить прибыльность жанров на диаграмме размаха, на актуальной выборке, обновить вывод - строки 58-59
#  + перестроить графики в ТОП-5 и пересмотреть раздел рейтингов - строки 61-63
#  + в разделе проверки гипотез можно более подробно расшифровать значение p_value - комментарий после строки 71 дополнил 
#  + перепроверить промежуточные и итоговый выводы после всех исправлений - раздел итоговые выводы
# 
# 

# <div class="alert alert-success">
# <font size="5", color= "seagreen"><b>✔️ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />   
# Привет, Денис :) Спасибо за все исправления и комментарии
#         
# Критические ❌ комментарии связаны с неточностями: 
# 
#  + удалить 2 старые игры с пропусками
#  + обработка пропусков в столбце рейтингов ESRB
#  + сократить категории в рейтингах ESRB — на твое усмотрение
#  + переопределить перечень перспективных платформ
#  + исправить диаграмму размаха для анализа продаж на актуальных платформах
#  + оценить прибыльность жанров на диаграмме размаха
#  + перестроить графики в ТОП-5 и пересмотреть раздел рейтингов
#  + в разделе проверки гипотез можно более подробно расшифровать значение p_value 
#  + перепроверить промежуточные и итоговый выводы после всех исправлений
# 
# Стоит обратить внимание на ⚠️ комментарии...        
#         
# Если будут вопросы про мои комментарии - задавай, если какой-то формат взаимодействия не устраивает или есть какие-то другие пожелания - пиши :)
# 
# <div class="alert alert-success">
#     <font size="5", color= "seagreen"><b>Жду твой проект и твои комментарии 🤝</b></font><br />

# <div class="alert alert-success">
# <font size="4"><b>Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
#     Привет, Денис :) Спасибо, что прислал задание :) Меня зовут Ринат Хисамов и я буду проверять твой проект. Предлагаю обращаться друг к другу на ты. Так нам будет гораздо проще и удобней общаться
# 
# Мои комментарии обозначены пометкой <b>Комментарий ревьюера</b>. Далее в файле сможешь найти их в похожих ячейках (если фон комментария зелёный — всё сделано правильно (✔️), рекомендации таким же цветом. Отдельным цветом — блок ссылок (примеры ниже, 🍕). Оранжевым или светло желтым рекомендации, которые, хоть и не обязательны, но точно сделают ревью лучше. (⚠️); <u> красный комментарий</u>: код, график или вывод стоит переделать (❌)). 
# 
# Не удаляй все эти комментарии и постарайся учесть их в ходе выполнения данного проекта. 
# Будет замечательно, если добавишь свои комментарии и пояснения✍
#         
# Поехали 🚀
#     <br />
#     </font>
# 
# </div>

# <div style="border:solid steelblue 1px; padding: 20px">
#     
# <font size="4"><p style="text-align:center"><b>Примеры комментариев </b></p></font>
#     
# <div style="border:solid steelblue 3px; padding: 20px">
# <font size="4"><b>🍕 Пример комментария - совета, здесь м.б. просто ссылка</b></font>
#     <br /> 
#         <font size="3", color = "black">
# <br />
#     Тут всего такого разного и вкусного :), есть способы прокачать проект визуализациями (ценит большинство "боссов")  <br /><br />
#         <a href="https://pyprog.pro/mpl/mpl_short_guide.html">Краткое руководство по Matplotlib</a>
#         На сайте много полезных материалов, мне самому очень помогло в свое время, до сих пор подсматриваю :)
# 
# 
# </div>
#     
# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
#     <font size="3"><b>⚠️ Пример оформления некритичного комментария</b>
#     <br /> 
#     <font size="2", color = "black">
# <br />
#     Рекомендации, которые, хоть и не обязательны, но точно сделают ревью лучше
#     <br />
#     </font>
# 
# </div>
#     
# <div class="alert alert-danger">
# <font size="3"><b>❌ Пример оформления комментария к блоку(строке) программного кода (или выводу), который стоит переделать</b></font>
#     <br /> 
#     <font size="2", color = "black">
# <br />
#     Отправлен не тот проект, напиши в своих комментариях, что случилось? жду — <b>это пример</b>
#     <br />
#     </font>
# 
# </div>
#     
# <div class="alert alert-success">
# <font size="4"><b>✔️ Пример оформления комментария, который нравится большинству студентов</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
#     Круто, молодец, отлично, логично, или — 👌, 👍, или — выводы отвечают на все вопросы к данным и проекту
#     <br />
#     </font>
# 
# </div>

# # Для твоих вопросов или комментариев оставлю такую ячейку, чтобы было удобнее взаимодействовать на проекте

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Добавил все комментарии в процессе выполнения к красным комментам

# # Уровень продаж компьютерных и консольных игр в разных регионах до 2016 года

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Название придаст вес проекту
#         
# а подробности выполнения проекта дадут возможность вспомнить через полгода — что мы тут делали 

#  # **Общая информация о датасете**

# Имортируем библиотеки

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statistics import mean


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Молодец, что делишь блоки загрузки библиотек и датасета, в случае необходимости добавления новых библиотек не придется загружать весь датасет заново и перезапускать проект целиком

# ### Считываем файл, сохраняем в переменную gamesdf и выводим на экран первые 30 строк таблицы

# In[2]:


games_df = pd.read_csv('/datasets/games.csv')


# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />   Стоит применять «змеиный стиль» в названиях переменных
#         
#         gamesdf

# In[3]:


games_df.head(30)


# In[4]:


games_df.info()


# In[5]:


games_df.describe()


# Строим гистаграммы
# 

# ###    Порядок действий после изучения таблицы:
# 1. Поменять типы столбцов Year_of_Release и Critic_Score
# 2. Проверить на дубликаты
# 3. Заменить нули в таблицах с продажами (не верю, что копии поплуряных игр, тот же WoW, в Японии продавались в 0)
# 4. Переименовать столбцы (сделать буквы строчными)
# 5. Построить гистограммы

# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> загрузка и первичное знакомство с данными проведены корректно

#  # **Обработка данных**

# In[6]:


games_df.isnull().sum().sort_values()


# In[7]:


# check
# пропущенные значения бары

def pass_value_barh(games_df):
    try:
        (
            (games_df.isna().mean()*100)
            .to_frame()
            .rename(columns = {0:'space'})
            .query('space > 0')
            .sort_values(by = 'space', ascending = True)
            .plot(kind = 'barh', figsize = (19,6), rot = -5, legend = False, fontsize = 16)
            .set_title('Пример' + "\n", fontsize = 22, color = 'SteelBlue')    
        );    
    except:
        print('пропусков не осталось :) или произошла ошибка в первой части функции ')


# In[8]:


pass_value_barh(games_df)


# <div style="border:solid steelblue 3px; padding: 20px">
# <font size="4">🍕<b> Комментарий ревьюера</b></font>
# <br /> 
# <font size="3", color = "black">
# <br /> Наглядность представления информации одна из важных составляющих работы дата-аналитика или дата-сайентиста
#     
# мой график оформлен не совсем корректно, сможешь отметить, что стоило бы исправить в графике?
#   

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Думаю, здесь не совсем корректно оставлять genre и name, и возможно в начале нужно не mean, а sum

# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
#         
# За поворот текста отвечает параметр rot, можно добавить обозначение оси х (процент записей с пропусками)
#         
#         df.isna().mean()*100 — для расчета процента пропусков в колонке
#         
# На графике мы оцениваем масштаб проблемы с пропусками и возможное совпадение % пропущенных значений в колонках

# In[9]:


games_df.columns = games_df.columns.str.lower()


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />стоит привести к нижнему регистру и содержимое категориальных столбцов, в т.ч. с целью поиска дубликатов

# ### Проверяем уникальные значения в столбике жанр, вдруг есть повторения или ошибки

# In[10]:


games_df['genre'].unique()


# ### Есть пропущенные значения, проверяем через loc

# In[11]:


games_df.loc[games_df['genre'].isna()]


# ### Два пропущенных значения, причем они есть и в столбце name. Проверяем name на значения точно также

# In[12]:


games_df.loc[games_df['name'].isna()]


# ### Те же самые значения пропущены и в name. В целом, эти две строки смысловой нагрузки не имеют, так как пропущено название. Можно их удалить.

# In[13]:


games_df.dropna(subset = ['name'], inplace=True)


# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br />  
#     <font size="3", color = "black">
# <br />   верное решение, стоит поправить код ...
#         
#        games = gamesdf.loc[gamesdf['name'].isna()]
#         
# удаление не выполнено

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера в2</b></font>
#     <br />  
#     <font size="3", color = "black">
# <br />   
#         
#         games_df['name'].dropna()
#         
# удаление не выполнено
#         
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html?highlight=dropna#pandas.DataFrame.dropna
#         
#         inplace (bool, default False)
#         Whether to modify the DataFrame rather than creating a new one.

# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />верное решение

# In[14]:


len(games_df['name'].unique())


# ### Проверяем количество дубликатов, их нет, идем дальше

# In[15]:


games_df.duplicated().sum()


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Проверка на поиск дубликатов выполнена, молодец
#         
# Особенно это станет важным, когда мы перейдем к более сложным задачам на втором модуле курса
#           

# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />   
# На рабочих проектах стоит искать дубликаты по ключевым столбцам, для примера по сумме параметров: 
#    
#     ['name', 'platform', 'year_of_release']
#     
# С обязательным приведением содержимого столбцов к нижнему регистру
#         
# В выборке есть 2 строчки неполных дубликатов, если останется время стоит их поискать
#         
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html?highlight=duplicat#pandas.DataFrame.duplicated
#     

# Обработаем значения tbd (to be decided or to be determined). Аббревиатура означает неопределенность, поэтому такие значения можно заполнить NaN, также нельзя измерить тип данных.

# In[16]:


games_df.query('user_score == "tbd"')


# In[17]:


games_df.loc[games_df['user_score']=="tbd", 'user_score']=np.nan
games_df['user_score'] = games_df['user_score'].astype('float64')


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Верно, по своей сути tbd и является Nan. Отлично, что определяешь неявные пропущенные значения.

# Обработаем другие столбики, поменяем типы данных в столбце year_of_release, critic_score

# In[18]:


games_df['year_of_release'] = games_df['year_of_release'].astype('Int64')


# In[19]:


games_df['critic_score'] = games_df['critic_score'].astype('float64')


# In[20]:


games_df.head(30)


# Год и оценка критиков поменялись, идем дальше

# Посчитаем суммарные продажи во всех регионах

# In[21]:


games_df['total_sales'] = games_df[['na_sales','eu_sales','jp_sales', 'other_sales']].sum(axis = 1)


# <div style="border:solid steelblue 3px; padding: 20px">
# <font size="4">🍕<b> Комментарий ревьюера</b></font>
# <br /> 
# <font size="3", color = "black">
# <br />
# можно
#     
#     df['total_sales'] = df[['na_sales','eu_sales','jp_sales', 'other_sales']].sum(axis = 1)

# In[22]:


games_df.head(30)


# ### Строим гистаграммы
# 

# In[23]:


games_df.hist(figsize=(15, 20), color = 'maroon')
plt.show()


# In[24]:


# check
games_df.info()


# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />          
# Рейтинг ESRB — категориальные данные, стоит внимательно взглянуть на содержимое и предложить чем заполнить пропуски, возможно это поможет найти необычные различия в поведении клиентов, допом можно подумать над сокращением количества категорий
#         
# Т.к. записи с пропусками не учитываются при группировке данных, мы не сможем выявить реальный портрет клиента

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Обработал пропуски в категорию Missing, строка 28

# ### Обработка рейтинга ESRB

# In[25]:


games_df['rating'].unique()


# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Почитал про рейтинг K-A, он использовался до 1998 года и был заменен на E. В целом можно поступить также, меняем в таблице все значения K-A на E

# In[26]:


games_df.loc[(games_df['rating'] == 'EC') | (games_df['rating'] == 'K-A'),'rating'] = 'E'
games_df.loc[games_df['rating'] == 'AO', 'rating'] = 'M'
games_df['rating'].unique()


# In[27]:


games_df.loc[games_df['rating'] == 'RP', 'rating'] = np.nan


# In[28]:


games_df['rating'] = games_df['rating'].fillna('missing')


# In[29]:


games_df.loc[games_df['rating'].isna()].head(10)


# In[30]:


games_df['rating'].hist(color = 'maroon')


# In[31]:


# check
games_df.info()


# <div class="alert alert-dang er">
# <font size="4"><b>❌ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Без заполнения пропущенных значений в рейтиге ESRB мы не сможем сгруппировать данные и найти отличие, когда будем исследовать долю рынка у разных категорий рейтинга ESRB
#         
# Можно сократить одну категорию

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Сократил категории

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Без заполнения пропущенных значений в рейтиге ESRB мы не сможем сгруппировать данные и найти отличие, когда будем исследовать долю рынка у разных категорий рейтинга ESRB

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Обработал пропуски в категорию Missing, строка 28

# In[32]:


# check
games_df['rating'].value_counts()


# In[33]:


games_df['rating'].value_counts()


#  # **Исследовательский анализ данных**

# ### Для ответа на первый вопрос сортируем датасет по годам и посчитаем количество игр в столбце  name

# In[34]:


year_of_release = games_df.groupby('year_of_release')['name'].count().reset_index()
year_of_release.columns = ['year', 'sum']
year_of_release


# ### По таблице видно, что с 2012 года продажи сократились примерно в 2 раза. Чтобы предсказать продажи за 2017 год, будет лучше взять период 2012-2016.

# ### Сначала построим график для всех периодов, затем для 2012-2016

# In[35]:


plt.bar(year_of_release['year'], year_of_release['sum'], color ='maroon',
        width = 0.4)

plt.xlabel("Год выпуска")
plt.ylabel("Кол-во выпусков")
plt.title("Кол-во выпусков игр по годам")
plt.show()


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Классный график, все элементы добавлены, молодец

# ### Выделим период 2012-2016

# In[36]:


year_of_release2012 = games_df.loc[(games_df['year_of_release'] >= 2012) & (games_df['year_of_release'] <= 2016)].groupby('year_of_release')['name'].count().reset_index()
year_of_release2012.columns = ['year', 'sum']
year_of_release2012


# In[37]:


plt.bar(year_of_release2012['year'], year_of_release2012['sum'], color ='maroon',
        width = 0.4)

plt.xlabel("Год выпуска")
plt.ylabel("Кол-во выпусков")
plt.title("Кол-во выпусков игр по годам")
plt.show()


# ### Продажи по платформам, топ 10 платформ и распределение по годам

# In[38]:


# сводная таблица по платформам

platform_sales = games_df.pivot_table(index='platform', values='total_sales', aggfunc='sum')

# сортировка и вывод 10 первых

platform_sales.sort_values('total_sales', ascending=False).head(10)


# In[39]:


games_df.groupby('year_of_release')['total_sales'].sum().plot(x='year_of_release,', 
                                                              y='total_sales', 
                                                              kind='bar', 
                                                              figsize=(9,5),color = 'maroon')
plt.title('Продажи по годам\n  ')
plt.xlabel('Год выпуска')
plt.grid(True)
plt.show();


# In[40]:


# actual_periodвв.groupby('year_of_release')['total_sales'].sum().plot(x='year_of_release,', 
#                                                               y='total_sales', 
#                                                               kind='bar', 
#                                                               figsize=(9,5),color = 'maroon')
# plt.title('Продажи по годам\n  ')
# plt.xlabel('Год выпуска')
# plt.grid(True)
# plt.show();


# <div class="alert alert-danger">
# <font size="4"><b>❌ Комментарий ревьюера в4</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Код не работает ... закомментировал
#         
#         # actual_periodвв.groupby('year_of_release')['total_sales'].sum().plot(x='year_of_release,', 
#         #                                                               y='total_sales', 
#         #                                                               kind='bar', 
#         #                                                               figsize=(9,5),color = 'maroon')
#         # plt.title('Продажи по годам\n  ')
#         # plt.xlabel('Год выпуска')
#         # plt.grid(True)
#         # plt.show();

# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Для красоты («продаваемости») всего проекта не стоит выходить за видимую ширину тетради юпитер ноутбука, коллеги ценят удобство прочтения кода. Широкие строки кода рекомендуется — делить <a href="https://qastack.ru/programming/53162/how-can-i-do-a-line-break-line-continuation-in-python">Перенос длинных строк кода</a>. 
#     
# </div>

# In[41]:


platform_top_name=['PS2', 'X360', 'PS3', 'Wii', 'DS', 'PS', 'GBA', 'PS4', 'PSP', 'PC']

data_top10_sales= games_df.query('(platform in @platform_top_name) and (year_of_release>1)')


# In[42]:


for platform in platform_top_name:
    groups = data_top10_sales.query('platform==@platform')['year_of_release']
    counts = data_top10_sales.query('platform==@platform')['total_sales'] 

    plt.bar(groups, counts, color = 'maroon')
    plt.title(platform)
    plt.xlabel("год выпуска")
    plt.ylabel("суммарные продажи")
    plt.grid(True)
    color = 'red'
    plt.plot(color = color)
    plt.show()


# In[43]:


sum = []
for platform in platform_top_name:
    item=data_top10_sales.query('year_of_release>1 and platform==@platform')['year_of_release'].value_counts().count()
    sum.append(item)
print(mean(sum))


# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Импорт библиотек стоит выносить на первые строки проекта, так коллегам будет удобнее настроить свои рабочие места под наши требования

# Становится понятно, что средний возраст платформы - 11 лет. 

# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Отличное определение жизненного цикла продаж у платформ, молодец

# In[44]:


actual_period = games_df.query(' year_of_release > 2012')
actual_period.head()


# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Для целей прогнозирования продаж на следующий год даже в традиционных бизнесах редко берут данные более чем за 2-3 года. А в такой динамично меняющейся индустрии, как компьютерные игры и вовсе не стоит брать слишком большой временной интервал - иначе обязательно захватишь уже отжившие тренды. Но и слишком короткий период тоже брать не стоит
#         
#        actual_period = gamesdf.query(' year_of_release > 2012')

# In[45]:


purchase_rating = actual_period.groupby('platform')['total_sales'].sum().sort_values(ascending = False )
purchase_rating


# In[46]:


purchase_dinamics = actual_period.groupby(['platform' , 'year_of_release'])['total_sales'].sum()
purchase_dinamics


# In[47]:


dict_platforms = purchase_rating.reset_index().platform
dict_platforms


# In[48]:


def show_lines(data, year, title):
    """
    input: data [pd.DataFrame] - таблица данных для построения граафика
           year [int16] - год начала построения данных
           title [str] - название графика
    output: None
    description: Функция строит график суммарных продаж по годам для выбранных платформ
                 Используется функция lineplot() библиотеки seaborn
    """
    plt.figure(figsize=(16,8))
    plt.title(title, fontsize=18)
    sns.lineplot(x='year_of_release',
                y='total_sales',
                hue='platform',
                markers=True,
                data=(data
                      .query('year_of_release >= @year')              # почти все актуальные платформы появились после 2000 года
                      .groupby(['platform','year_of_release'])['total_sales']   # группируем по платформам и годам
                      .agg('sum')                                             # считаем суммы продаж по платформам и годам
                      .reset_index()
                     )
                )
show_lines(games_df.query('platform in @platform_top_name'), 
           1990, 
           'Продажи популярных игровых платформ по годам за всё время')
    


# <div class="alert alert-dang er">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Странный выброс 1985 года у DS, можно посмотреть когда платформу выпустили на рынок, стоит удалить аномалию 

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Исправил

# <div class="alert alert-danger">
# <font size="4"><b>❌ Комментарий ревьюера в4</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Аномалия до сих пор в нашей выборке, стоит повторить приемы фильтрации данных, без этих навыков будет тяжело выполнить проекты второго модуля
#     
#         Странный выброс 1985 года у DS, можно посмотреть когда платформу выпустили на рынок, стоит удалить аномалию 

# In[49]:


# check
games_df.query("year_of_release == 1985 and platform == 'DS'")


# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />что-то у нас с содержанием надписей оси х

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Исправил

# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Скажи пожалуйста, а что они означают (надписи оси х)
#         
# ![image.png](attachment:image.png)        
#         
#         что-то у нас с содержанием надписей оси х

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Поменял код, добавил другую диаграмму, где ось x - год релиза

# In[50]:


dict_boxplot = purchase_rating.reset_index()['platform'][0:5]
dict_boxplot

perspective_platforms = actual_period.query('platform in @dict_boxplot').reset_index()
perspective_platforms.head()


# ### Ящик с усами самых прибылных платформ

# In[51]:


actual_period.boxplot(column="total_sales", by="platform", figsize = (15,7))
plt.ylim(0,2)
plt.show()


# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Отлично, ты используешь диаграмму размаха для определения успешности платформы, молодец
#         
# Для более полной оценки продаж на платформах стоит добавить график со 100% масштабом, посмотреть на максимальные продажи 

# <div class="alert alert-dang er">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# 
# Не стоит ограничивать перечень платформ из актуального периода для оценки прибыльности платформ__
#         
# диаграмму размаха перерисовать на актуальной выборке с полным перечнем платформ

# <div class="alert alert-da nger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# X360 и PS3 уже исполнилось 11 лет и они с малой вероятностью станут перспективными в 2017 году
#         
# Стоит пересмотреть перечень перспективных платформ

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Добавил график для всех платформ + отдельно для всех продаж

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
#         
#         Добавил график для всех платформ + отдельно для всех продаж
#         
# Не стоит, по заданию проекта мы показываем перечень только актуальных платформ из выборки
#         
#         actual_period
#         
# и после её создания, уже не возвращаемся к другим выборкам, а работаем только с выборкой  actual_period
# 
# __Не стоит ограничивать перечень платформ из актуального периода для оценки прибыльности платформ__
#         
# диаграмму размаха перерисовать на актуальной выборке с полным перечнем платформ

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Сделал диаграмму для actual_period, получился список актуальных платформ

# <div class="alert alert-da nger">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# X360 и PS3 уже исполнилось 11 лет и они с малой вероятностью станут перспективными в 2017 году
#         
# __Стоит сформировать вывод, что в актуальном периоде такие платформы перспективные, а такие доживают свой век__

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Добавил вывод в конце

# ### Диаграмма рассеяния продажи от отзывов пользователей 

# In[52]:


ps4_gamesdf = perspective_platforms.query('platform == "PS4"')

ps4_gamesdf.plot(x='total_sales', y='user_score', kind='hexbin', 
              gridsize=30, figsize=(15, 7), sharex=False, grid=True)
ps4_gamesdf;


# ### Диаграмма рассеяния продажи от отзывов критиков

# In[53]:


ps4_gamesdf.plot(x='total_sales', y='critic_score', kind='hexbin', 
              gridsize=30, figsize=(15, 7), sharex=False, grid=True )
ps4_gamesdf;    


# ### Далее для других платформ по аналогии

# In[54]:



ps3_gamesdf = perspective_platforms.query('platform == "PS3"')

ps3_gamesdf.plot(x='total_sales', y='user_score', kind='hexbin', 
              gridsize=30, figsize=(15, 7), sharex=False, grid=True)
ps3_gamesdf;


# In[55]:


ps3_gamesdf.plot(x='total_sales', y='critic_score', kind='hexbin', 
              gridsize=30, figsize=(15, 7), sharex=False, grid=True )
ps3_gamesdf; 


# In[56]:



x360_gamesdf = perspective_platforms.query('platform == "X360"')
x360_gamesdf.plot(x='total_sales', y='user_score', kind='hexbin', 
              gridsize=30, figsize=(15, 7), sharex=False, grid=True)
x360_gamesdf;


# In[57]:


x360_gamesdf.plot(x='total_sales', y='critic_score', kind='hexbin', 
              gridsize=30, figsize=(15, 7), sharex=False, grid=True )
x360_gamesdf;
    


# ### Корреляция между отзывами пользователей и продажами
# 

# In[58]:


corelation = perspective_platforms.groupby('platform')
for platform, purchase_df in corelation:
    subset = corelation.get_group(platform) 
    coef_corr = subset.total_sales.corr(subset.user_score).round(3)
    print('Коэффициент кореляции' , platform , 'от отзывов пользователей', coef_corr.round(3))
    print()


# ### Корреляцию между отзывами критиков и продажами

# In[59]:


for platform, purchase_df in corelation:
    subset = corelation.get_group(platform) 
    coef_corr = subset.total_sales.corr(subset.critic_score).round(3)
    print('Коэффициент кореляции' , platform , 'от отзывов критиков', coef_corr.round(3))
    print()


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Сравнение показателей на нескольких платформах, позволяет набрать вес твоему исследованию, молодец

# ### Перспективные жанры

# In[60]:


# отсортируем список жанров игр по убыванию медианы глобальных продаж
genres_sorted = (actual_period
                 .groupby('genre')['total_sales']
                 .agg('median')
                 .sort_values(ascending=False)
                 .index
                )
# построим распределение глобальных продаж с помощью метода boxplot() библиотеки seaborn().
# графики построим на логарифмиеской оси для наглядности (в данных много выбросов, которые искажают boxplot)
plt.figure(figsize=(14,8))
sns.set(style="ticks")
ax = sns.boxplot(data=actual_period, y="total_sales", x='genre', order = genres_sorted)
# order = genres_sorted - для расстановки ящиков в порядке убывания глобальных продаж
ax.set_title('Распределение глобальных продаж по жанрам', fontsize=20)
ax.set_xlabel('Жанр игры', fontsize=15)
ax.set_ylabel('Глобальные продажи', fontsize=15)
plt.show()


# In[61]:


# отсортируем список жанров игр по убыванию медианы глобальных продаж
genres_sorted = (actual_period
                 .groupby('genre')['total_sales']
                 .agg('median')
                 .sort_values(ascending=False)
                 .index
                )
# построим распределение глобальных продаж с помощью метода boxplot() библиотеки seaborn().
# графики построим на логарифмиеской оси для наглядности (в данных много выбросов, которые искажают boxplot)
plt.figure(figsize=(14,8))
sns.set(style="ticks")
ax = sns.boxplot(data=games_df, y="total_sales", x='genre', order = genres_sorted)
# order = genres_sorted - для расстановки ящиков в порядке убывания глобальных продаж
ax.set_title('Распределение глобальных продаж по жанрам', fontsize=20)
ax.set_xlabel('Жанр игры', fontsize=15)
ax.set_ylabel('Глобальные продажи', fontsize=15)
ax.set(yscale="log")
plt.show()


# <div class="alert alert-dang er">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />        
# Стоит проанализировать прибыльность жанров на диаграмме размаха, сравнить медианные продажи на каждом жанре и проверить какая из них более стабильна и имеет более длинный ряд успешно продающихся игр
#         
# График нарисовать __в двух масштабах с выбросами и без__ (чтобы было видно 0.75-квантиль)

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Добавил один график, не очень понимаю зачем рисовать в двух масштабах

# <div class="alert alert-da nger">
# <font size="4"><b>❌ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />        
# Стоит нарисовать в двух обычных масштабах, не применяя ax.set(yscale="log")
#         
# Что показывают нам две диаграммы размаха — кол-во выбросов, это игры, которые принесли максимум выручки. Т.е. можно платформы/жанры сравнить по кол-ву игр-рекордсменов, а значит определить, какая из них способна выпустить наиболее привлекательные для игроманов игры. Это про выбросы
#         
# Второй вид мы используем для того, чтобы сравнить медианные продажи по платформе/жанру, чтобы уточнить в каком кол-ве продаются игры на платформе/жанре, какая из них более стабильна в продажах...
#         
#         
# Две хорошие статьи про диаграмму размаха
#         
# [Визуализация категориальных данных](https://pyprog.pro/sns/sns_7_categorical_data.html?)
#         
# [Исследуем отношение между переменными](https://dfedorov.spb.ru/pandas/downey/%D0%98%D1%81%D1%81%D0%BB%D0%B5%D0%B4%D1%83%D0%B5%D0%BC%20%D0%BE%D1%82%D0%BD%D0%BE%D1%88%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%B5%D0%B6%D0%B4%D1%83%20%D0%BF%D0%B5%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D0%BC%D0%B8.html?)
#        

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />        
# Стоит добавить вывод в подраздел о самых перспективных жанрах для рекламной кампании
#        

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Добавил в финальный вывод по жанрам

# In[62]:


# check
actual_period


# <div class="alert alert-da nger">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />  Диграммы размаха нарисованы на выборке, которая содержит неактуальный период 
#         
#         games_df
#         
# стоит поменять выборку на актуальную и пересмотреть выводы

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Поменял выборку на actual period

# ### Выводы

# **Из диаграммы рассеяния** следует что отзывов пользователей не влияют сушественно на продажи по скольку практически весь диапазон отзивов не меняет тенденция на продаж. Поэтому величина коэффициента коррелляции находится около нуля.
# Так же можно отметить, что **положительные отзывы критиков** влияют на продажи. Коэффициент коррелляции продаж от отзывов критиков больше, чем коэффициент корреляции продажи от отзывов пользователей. 
# **Средний возраст платформы** - 11 лет. Долгожитель - PC.
# **PS2 X360 PS3** - лидеры по продажам. Скорее всего, в списке получились две приставки от одной компнаии - возможно, как раз это связано с недолгим уровнем жизни платформ - когда выходит новая консоль, старая продолжает существовать какое то время и игры продолжаются продаваться, затем поколение умирает. X360 составляет отличную конкуренцию ps3.
# **Наиболее актуальный период** - от 2012 до 2016 года. Так как после 2011 года продажи резко упали в два раза. **Перспективные жанры** - Action, Sporsts, Shooter. В Японии вместо шутеров предпочитают role-playing

# <div class="alert alert-warning", style="border:solid coral 3px; padding: 20px">
# <font size="4", color = "DimGrey"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Возможно пользователи более критичны к играм, чем критики, но мы не сможем оценить какие действия повлияли на рост продаж в рамках нашего проекта (ограниченность имеющихся данных)
#         
#     
#         
# __Достаточно много игр с высокой оценкой критиков и слабой выручкой__
#         
# Приведу пример ложной корреляции, весьма известный в статистической литературе. Была исследована корреляционная связь между числом аистов, свивших гнезда в южных районах Швеции, и рождаемостью в эти же годы в Швеции. Расчёты, выполненные ради шутки, показали существенную положительную корреляцию между этими явлениями, хотя любому понятно, что это ложная корреляция.
# 
# Ещё пример ложной корреляции между приемом на работу новых менеджеров и созданием новых производственных мощностей. Возможно, именно менеджеры являются «причиной» капиталовложений в новые производственные мощности? Или же, наоборот, создание новых производственных мощностей послужило «причиной» приема на работу новых менеджеров?
# 
# Например, можно обнаружить сильную положительную связь (корреляцию) между разрушениями, вызванными пожаром, и числом пожарных, тушивших пожар. Следует ли заключить, что пожарные вызывают разрушения? Конечно, наиболее вероятное объяснение этой корреляции состоит в том, что размер пожара (внешняя переменная, которую забыли включить в исследование) оказывает влияние, как на масштаб разрушений, так и на числе привлеченных пожарных (т. е. чем больше пожар, тем большее количество пожарных вызывается на его тушение) .

# # **Портрет пользователя каждого региона**

# ### Самые популярные платформы (топ-5) 

# In[63]:



actual_period.groupby(by='platform').agg({'jp_sales':'sum'}).sort_values(by='jp_sales', ascending=False).head(5).plot(kind='bar')
actual_period.groupby(by='platform').agg({'eu_sales':'sum'}).sort_values(by='eu_sales', ascending=False).head(5).plot(kind='bar')
actual_period.groupby(by='platform').agg({'na_sales':'sum'}).sort_values(by='na_sales', ascending=False).head(5).plot(kind='bar')


# ### Самые популярные жанры (топ-5) 

# In[64]:


actual_period.groupby(by='genre').agg({'jp_sales':'sum'}).sort_values(by='jp_sales', ascending=False).head(5).plot(kind='bar')
actual_period.groupby(by='genre').agg({'eu_sales':'sum'}).sort_values(by='eu_sales', ascending=False).head(5).plot(kind='bar')
actual_period.groupby(by='genre').agg({'na_sales':'sum'}).sort_values(by='na_sales', ascending=False).head(5).plot(kind='bar')


# ### Самые популярные возрастные рейтинги (топ-5) 
# 

# In[65]:


actual_period.groupby(by='rating').agg({'jp_sales':'sum'}).sort_values(by='jp_sales', ascending=False).head(5).plot(kind='bar')
actual_period.groupby(by='rating').agg({'eu_sales':'sum'}).sort_values(by='eu_sales', ascending=False).head(5).plot(kind='bar')
actual_period.groupby(by='rating').agg({'na_sales':'sum'}).sort_values(by='na_sales', ascending=False).head(5).plot(kind='bar')


# <div class="alert alert-da nger">
# <font size="4"><b>❌ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Портреты клиентов нарисованы, молодец, значительное влияние на портрет оказывает период с 2012 г. ..., можем совершить ошибку при формировании рекомендации маркетологам
#         
#         platform_fracs = (games_df
#                   .loc[games_df['year_of_release'] >= 2012] 
#         
# Стоит перестроить графики раздела TOП-5:
# 
# + выбрать актуальный период;
# + обновить промежуточные и итоговые выводы.

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Диаграмма размаха по актуальному периоду

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> В топ-5 мы не строим диаграммы размаха, они нам требуется в подразделах проекта: для оценки платформ и жанров.
#         
# В разделе ТОП-5 мы ищем какую долю рынка занимают платформы, жанры и рейтинги ESRB, в каждом регионе на актуальной выборке

# <div class="alert alert-da nger">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Портреты клиентов нарисованы, молодец, значительное влияние на портрет оказывает период с 1980 г. ..., можем совершить ошибку при формировании рекомендации маркетологам
#         
#         games_df.groupby(by='genre').agg({'na_sales':'sum'}).sort_values(by='na_sales', ascending=False).head(5).plot(kind='bar', label='Япония').legend()
#         plt.title('Продажи по жанрам в Северной Америке за весь период')
#         
# Стоит перестроить графики раздела TOП-5:
# 
# + выбрать актуальный период;
# + обновить промежуточные и итоговые выводы
#         
# __Для всего проекта, после создания актуальной выборки actual_period, используем только её, стоит «забыть», что у нас существует выборка  games_df после 13-го  пункта или строчки кода__
#         
#         actual_period = games_df.query(' year_of_release > 2012')
#         
# Стоит исправить все подразделы ТОП-5

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Исправил раздел, добавил везде actual period и поменял графики

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />код не работает, переменной rating_na не существует
#         
#         # fig = plt.figure(figsize=(16,4))
#         # fig.suptitle('Рейтинг по платформам для трех регионов', fontsize=20, y=1.02)
#         # ax1 = plt.subplot(131)
#         # # построим quantile-quantile plots методом probplot() из библиотеки stats для обеих выборок
#         # plt.pie(rating_na['na_sales'], labels=rating_na.index)
#         # ax1.set_title('Северная Америка', fontsize=15)
# 
#         # ax2 = plt.subplot(132)
#         # plt.pie(rating_eu['eu_sales'], labels=rating_eu.index)
#         # ax2.set_title('Европа', fontsize=15)
# 
#         # ax3 = plt.subplot(133)
#         # plt.pie(rating_jp['jp_sales'], labels=rating_jp.index)
#         # ax3.set_title('Япония', fontsize=15)
#         # plt.show()

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Переделал графики, убрал этот код

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Портреты клиентов рассчитаны, молодец, значительное влияние на портрет оказывает период с 1980 г. ..., можем совершить ошибку при формировании рекомендации маркетологам
#         
# Стоит оформить графики раздела TOП-5:
# 
# + выбрать актуальный период;
# + для каждого ТОП-5 - построить 3 графика рядом с помощью subplots, оптимальнее сравнивать три региона по каждому виду портрета вместе;
# + оформить "двухуровневый заголовок" - и у всех трех графиков вместе, и у каждого из трех по отдельности;
# + при анализе платформ и жанров стоит все, что не вошло в ТОП-5, объединять в категорию "другие" - так картина анализа будет более полной
# 
#         
# Если столкнешься с трудностью выполнения данного пункта — присылай код, который не получился и вопрос, подумаем вместе
#         
# https://proproprogs.ru/modules/matplotlib-otobrazhenie-neskolkih-koordinatnyh-osey-v-odnom-okne
#         
# https://nagornyy.me/it/vizualizatsiia-dannykh-v-matplotlib/?ysclid=l4q3l4q0p8940570437
#     
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html?highlight=append#pandas.DataFrame.append

# ### Смотрим влияние рейтинга ESRB на продажи в отдельном регионе

# In[66]:


ersb = actual_period.groupby('rating')['eu_sales'].sum()
ersb


# In[67]:


ersb = actual_period.groupby('rating')['na_sales'].sum()
ersb


# In[68]:


ersb = actual_period.groupby('rating')['jp_sales'].sum()
ersb


# <div class="alert alert-da nger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# При твоем способе подсчета игры без рейтинга оказываются полностью исключенными из анализа. Но продажи именно этих игр могут указать на ключевое различие в регионах
#         
# Стоит поработать со столбцом рейтингов, заменить пропуски, посмотреть на частотность использования всех категорий рейтинга

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />  Добавил правильный рейтинг, получился более точный портрет

# <div class="alert alert-d anger">
# <font size="4"><b>❌  Комментарий ревьюера в2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Стоит заполнить пропуски в колонке рейтингов, подсказку оставил в первой части

# <div class="alert alert-d anger">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Стоит заполнить пропуски в колонке рейтингов на неизвестную категорию, чтобы мы смогли рассчитать влияние рейтингов ESRB в разных регионах

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Добавил категорию missing

# ### Выводы
# Самые популярные консоли по странам:
# Северная Америка - PS4 
# Европа - PS4
# Япония - Nintendo 3DS
#     Самые популярные жанры по странам:
#     В целом - Action, Sports, Shooter
#     Северная Америка - Action, Sports, Shooter
#     Европа - Action, Sports, Shooter
#     Япония - Action, Sports, Platform
#     Идентичные предпочтения в жанрах у Европы и Северной Америки. Япония отличилась любовью к платформерам. В целом, классификация по ERSB не оказывает влияние на продажи по регионам.
#     

# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера в4</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Интересно в чем причина, что на рынке Японии продается так много игр без рейтинга ESRB

# # **Проверяем гипотезы**

# In[69]:


xone_df = actual_period.query('platform == "XOne"').dropna(subset = ['user_score']).reset_index()

xone_df.head()


# In[70]:


xone_user_score = pd.Series(xone_df['user_score'])

xone_user_score


# In[71]:


pc_df = actual_period.query('platform == "PC" ').dropna(subset = ['user_score']).reset_index()

pc_df.head()


# In[72]:


pc_user_score = pd.Series(pc_df['user_score'])

pc_user_score


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Важно удалить пропуски и «заглушки» перед проведением теста, молодец

# In[73]:


from scipy import stats as st

alpha = 0.05

results_1 = st.ttest_ind(
    xone_user_score, 
    pc_user_score)

print('p-значение: ', results_1.pvalue)

if (results_1.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


#     Нулевая гипотеза: Средные значения отзывов пользователей платформы Xbox One равно средному значению отзывов пользователей платформы PC 
#     Алтернативная гипотеза: Средные значения отзывов пользователей платформы Xbox One не равно средному значению отзывов пользователей платформы PC
# 
# Для проверки гипотезы исползуем гипотезу о равенстве средних двух средных генеральных совокупностей.
# С вероятностью в почти 14% такое или большее различие можно получить случайно. Это явно слишком большая вероятность, чтобы делать вывод о значимом различии между средними значениями отзывов.

# <div class="alert alert-dan ger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Что означает на языке статистики p-значение:  0.14012658403611647

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Если значение больше 0,05 то мы не отвергаем гипотезу,  и не можем сделать вывод о существовании большой разницы

# <div class="alert alert-info">
# <font size="4", color = "black"><b>✍ Комментарий студента в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Дополнил вывод, спасибо за детальное объяснение! Про  гипотезы теперь понял

# <div style="border:solid steelblue 3px; padding: 20px">
# <font size="4">🍕<b> Комментарий ревьюера в2</b></font>
# <br /> 
# <font size="3", color = "black">
# <br />
# Приведу пример и теорию для понимания формулировок и интерпретации итогов проведения гипотез</b>
#         
#       
# Задача. Приведены два датасета: сумма покупок, совершённых за месяц посетителями, привлечёнными по двум разным каналам. В вашем распоряжении случайная выборка из 30 покупок для каждого канала.
#         
# H0 - средние чеки равны
#         
#         
# H1 - средние чеки НЕ равны
# 
# Да сама формулировка нулевой и альтернативной гипотезы звучит именно так, но результат теста интерпретируется другими словами
#         
# 
# <b>Из теории на тренажере</b>
#         
# Формулирование двусторонних гипотез. <br>
#         
# <b>Никакие экспериментально полученные данные никогда не подтвердят какую-либо гипотезу. Это наше фундаментальное ограничение. </b>Данные могут лишь не противоречить ей или, наоборот, показывать крайне маловероятные результаты (при условии, что гипотеза верна). Но и в том, и в другом случае нет оснований утверждать, что выдвинутая гипотеза доказана.
# Допустим, данные гипотезе не противоречат, тогда мы её не отвергаем. Если же мы приходим к выводу, что получить такие данные в рамках этой гипотезы вряд ли возможно, у нас появляется основание отбросить эту гипотезу.
#     
# P-значение (англ. P-value) — величина, используемая при тестировании статистических гипотез. <b>Фактически это вероятность ошибки при отклонении нулевой гипотезы (ошибки первого рода)</b> пример ниже

# In[74]:


# Приведены два датасета: сумма покупок, совершённых за месяц посетителями ...

sample_1 = [3071, 3636, 3454, 3151, 2185, 3259, 1727, 2263, 2015,
2582, 4815, 633, 3186, 887, 2028, 3589, 2564, 1422, 1785,
3180, 1770, 2716, 2546, 1848, 4644, 3134, 475, 2686,
1838, 3352]
sample_2 = [1211, 1228, 2157, 3699, 600, 1898, 1688, 1420, 5048, 3007,
509, 3777, 5583, 3949, 121, 1674, 4300, 1338, 3066,
3562, 1010, 2311, 462, 863, 2021, 528, 1849, 255,
1740, 2596]
alpha = .05 # критический уровень статистической значимости
# если p-value окажется меньше него - отвергнем гипотезу
results = st.ttest_ind(
sample_1,
sample_2)
print('p-значение:', results.pvalue)
if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:

    print("Не получилось отвергнуть нулевую гипотезу")


# <div style="border:solid steelblue 3px; padding: 20px">
# <font size="4">🍕<b> Комментарий ревьюера в2</b></font>
# <br /> 
# <font size="3", color = "black">
# <br />
# Интерпретация результата:
# 
# Полученное значение p-value говорит о том, что хотя средний чек пришедших из разных каналов и неодинаков, <b>с вероятностью в почти 19% такое или большее различие можно получить случайно. </b>Это явно слишком большая вероятность, чтобы делать вывод о значимом различии между средними чеками.
#         
# А если p-value  будет равно 0,9999, то это значит, что с вероятностью почти 100% <u>такое различие</u> можно получить случайно — то есть почти никогда :)  (но учитываем, что тест проводится на выборке из генеральной совокупности, все может поменяться)
# 

# In[75]:


action_df = actual_period.query(' genre ==  "Action"').dropna(subset = ['user_score']).reset_index()

action_df


# In[76]:


action_user_score = pd.Series(action_df['user_score'])
action_user_score.head


# In[77]:


sports_df = actual_period.query(' genre ==  "Sports"').dropna(subset = ['user_score']).reset_index()

sports_df


# In[78]:


sports_user_score =  pd.Series(sports_df['user_score'])
sports_user_score.head()


# In[79]:


results_2 = st.ttest_ind(
    action_user_score, 
    sports_user_score)

print('p-значение: ', results_2.pvalue)

if (results_2.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# Нулевая гипотеза: Средние пользовательские рейтинги жанров Action (англ. «действие») равно Sports (англ. «виды спорта»).
# 
# Алтернативная гипотеза:Средние пользовательские рейтинги жанров Action (англ. «действие») разные Sports (англ. «виды спорта»).
# 
# Для проверки гипотезы исползуем гипотезу о равенстве средних двух средных генеральных совокупностей.
# 
# Отвергаем нулевую гипотезу поскольку с вероятностью 1,27х10-26% что средные значения одинаковые. То есть она очень близко к нулю.

# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> Гипотезы сформулированы верно

# ### check

# # **Итоговые выводы**

# По результатам обработки и исследовательского анализа данных удалось получить следующую информацию:
# **Средний возраст платформы** - 11 лет. Долгожитель - PC. За период с 2011 по 2016 года лидеры среди платформ - PS4, X360, Wii, Xone и PS3. Скорее всего, в списке получились две приставки от одной компнаии - возможно, как раз это связано с недолгим уровнем жизни платформ - когда выходит новая консоль, старая продолжает существовать какое то время и игры продолжаются продаваться, затем поколение умирает. X360 составляет отличную конкуренцию ps3.
#     
#     **Наиболее актуальный период** - от 2011 года.  как после 2011 года продажи резко упали в два раза.
# 
#     Самые популярные консоли по странам:
# Северная Америка - PS4, XOne, X360
# Европа - PS4, PS3, XOne
# Япония - Nintendo 3DS, PS3, PSV
# 
#     Самые популярные жанры по странам:
#     В целом - Action, Sports, Shooter
#     Северная Америка - Action, Sports, Shooter
#     Европа - Action, Sports, Shooter
#     Япония - Role-Playing, Action, Misc
#     
#     Игры именно в этих жанрах являются наиболее динамичными и интересными для представителей разных возрастов, многие знаменитые продаваемые серии игры именно в этих жанрах.
#     
#     Идентичные предпочтения в жанрах у Европы и Северной Америки. Япония отличилась любовью к жанру ролевых игр. В целом, классификация по ERSB не оказывает влияние на продажи по регионам. По возрасту - в основном игроки от 17 лет. В Японии наиболее распространено не указывать возрастной рейтинг, а самая популярная категория - Teens.

# <div class="alert alert-danger">
# <font size="4"><b>❌ Комментарий ревьюера в4</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# X360 и PS3 уже исполнилось 11 лет и они с малой вероятностью станут перспективными в 2017 году, не стоит их включать в итоговый вывод

# <div class="alert alert-da <div class="alert alert-danger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# X360 и PS3 уже исполнилось 11 лет и они с малой вероятностью станут перспективными в 2017 годуnger">
# <font size="4"><b>❌ Комментарий ревьюера в3</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Стоит перепроверить весь итоговый вывод и составить итоговую рекомендацию бизнесу для решения главной задачи проекта
#         
#         Вы работаете в интернет-магазине «Стримчик», который продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Вам нужно выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.
#         
#         Перед вами данные до 2016 года. Представим, что сейчас декабрь 2016 г., и вы планируете кампанию на 2017-й. Нужно отработать принцип работы с данными. Неважно, прогнозируете ли вы продажи на 2017 год по данным 2016-го или же 2027-й — по данным 2026 года.
#         

# <div class="alert alert-d anger">
# <font size="4"><b>❌ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />Итоговый вывод технически составлен грамотно
#         
# стоит перепроверить результаты после определения актуального периода и исправления всех комментариев, __можно обновить названия самых актуальных платформ, жанров и рейтингов, какую долю они занимают на исследуемых рынках__
#         
# стоит перепроверить весь вывод, как пример →
#         
# ![image.png](attachment:image.png)        
#         

# <div class="alert alert-success">
# <font size="5", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />   
# Ты выполнил практически все пункты проекта, молодец! Проведен значительный объем исследования 
#         
# Критические ❌ комментарии связаны с неточностями: 
# 
#  + добавить название
#  + 2 старые игры с пропусками
#  + обработка пропусков в столбце рейтингов ESRB
#  + сократить категории в рейтингах ESRB — на твое усмотрение
#  + переопределить перечень перспективных платформ
#  + исправить диаграмму размаха для анализа продаж на актуальных платформах
#  + оценить прибыльность жанров на диаграмме размаха
#  + построить графики в ТОП-5 и пересмотреть раздел рейтингов
#  + в разделе проверки гипотез можно более подробно расшифровать значение p_value 
#  + перепроверить промежуточные и итоговый выводы после всех исправлений
# 
# Стоит обратить внимание на ⚠️ комментарии...        
#         
# Если будут вопросы про мои комментарии - задавай, если какой-то формат взаимодействия не устраивает или есть какие-то другие пожелания - пиши :)
# 
# <div class="alert alert-success">
#     <font size="5", color= "seagreen"><b>Жду твой проект и твои комментарии 🤝</b></font><br />

# <div style="border:solid steelblue 3px; padding: 20px">
# <font size="4">🍕<b> Комментарий ревьюера</b></font>
# <br /> 
# <font size="3", color = "black">
# <br />
# Может пригодиться  
#     
# [Подборка статей о работе с библиотеками для анализа данных на языке Python](https://dfedorov.spb.ru/pandas/)
#     
#    
# [Визуализация](https://dfedorov.spb.ru/pandas/%D0%AD%D1%84%D1%84%D0%B5%D0%BA%D1%82%D0%B8%D0%B2%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20Matplotlib.html)
# 
# 
# [Искусство статистики](https://www.mann-ivanov-ferber.ru/books/iskusstvo-statistiki/)
#         
# [Постер «Графики, которые убеждают всех»](https://www.notion.so/6c5ae8ceb8b5411e907c93c9b5e6a44e)        
#         
#         
# В помощь — как реализовать интерактивный план проекта вручную (для собственных проектов), смотри по <a href="https://stackoverflow.com/questions/49535664/how-to-hyperlink-in-a-jupyter-notebook/49717704">ссылке</a>
#     
# пара ссылок и по разделам проекта можно будет переходить без пролистывания всего кода, особенно актуально на проектах длина которых >  10 страниц (и там где не установлен плагин TOC)      

# ### Бонус

# In[80]:


data_games = pd.read_csv('/datasets/games.csv')


# In[81]:


data_games.columns = map(str.lower, data_games.columns)


# In[82]:


data_games = data_games.dropna(subset = ['year_of_release', 'name', 'genre'])


# In[83]:


data_games['user_score'] = data_games['user_score'].replace('tbd', np.nan).astype('float')


# In[84]:


data_games['rating'] = data_games['rating'].fillna('unknown')


# In[85]:


data_games['total_sales'] = data_games[['na_sales','eu_sales','jp_sales', 'other_sales']].sum(axis = 1)


# In[86]:


# check
# круги + категория другие
def graph (df, year, region, name, axes):
    
    df = df.query('year_of_release >= @year')
    
    sales = df.pivot_table(index='platform', 
                           values=region, 
                           aggfunc='sum').nlargest(5, region)
    
    sales = sales.reset_index()
    
    sales = (
            sales.append({'platform': 'Other', region: df[region].sum() 
                       - sales[region].sum()}, ignore_index= True)
         )
    
    
    sales.columns = ['platform', 'sales']
      
    labels_c=sales.platform
    colours = {'Wii':'C0', 'NES':'C1', 'GB':'C2', 'DS':'C3', 'X360':'C4', 
    'PS3':'C5', 'PS2':'C6', 'SNES':'C7', 'GBA':'C8',
               'PS4':'steelblue', '3DS':'orange', 
               'N64':'C11', 'PS':'C12', 'XB':'C13', 'PC':'C14', '2600':'C15', 'PSP':'C16', 
               'XOne':'C17',
               'WiiU':'C18', 'GC':'C19', 'GEN':'C20', 'DC':'C21', 'PSV':'C22', 
               'SAT':'C23', 'SCD':'C24', 'WS':'C25', 'NG':'C26', 
               'TG16':'C27', '3DO':'C28', 'GG':'C29', 'PCFX':'C30', 'Other':'darkred'}
   
    sales.plot(kind='pie',
               y="sales",
               
               autopct='%1.0f%%',
               wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
               textprops={'size': 'x-large'}, 
               labels= labels_c,
               colors=[colours[key] for key in labels_c],
               legend=False, 
               title = f"Популярность платформ в {name} ", 
               ax = axes).set(ylabel='')
    
    plt.tight_layout()


# In[87]:


# check
# круги в ряд
fig, axes = plt.subplots(1, 3, figsize = (15,6))
fig.suptitle('Обзор рынка платформ (портрет покупателя)', fontsize = 30, fontweight='bold')

x_year = 2012

graph(data_games, x_year, 'na_sales', 'Северной Америке', axes[0])
graph(data_games, x_year,'eu_sales', 'Европе', axes[1])
graph(data_games, x_year, 'jp_sales', 'Японии', axes[2])


# In[88]:


# check
# круги в ряд
fig, axes = plt.subplots(1, 3, figsize = (15,6))
fig.suptitle('Обзор рынка платформ (портрет покупателя)', fontsize = 30, fontweight='bold')

x_year = 2014

graph(data_games, x_year, 'na_sales', 'Северной Америке', axes[0])
graph(data_games, x_year,'eu_sales', 'Европе', axes[1])
graph(data_games, x_year, 'jp_sales', 'Японии', axes[2])


# <div class="alert alert-success">
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера в4</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br /> 2015 год взят для акцентирования на изменении доли в продажах современных платформ

# In[89]:


# check
# круги в ряд
fig, axes = plt.subplots(1, 3, figsize = (15,6))
fig.suptitle('Обзор рынка платформ (портрет покупателя)', fontsize = 30, fontweight='bold')

x_year = 2015

graph(data_games, x_year, 'na_sales', 'Северной Америке', axes[0])
graph(data_games, x_year,'eu_sales', 'Европе', axes[1])
graph(data_games, x_year, 'jp_sales', 'Японии', axes[2])


# In[ ]:




