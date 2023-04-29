#!/usr/bin/env python
# coding: utf-8

#  # **Общая информация о датасете**

# Имортируем библиотеки

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statistics import mean




# ### Считываем файл, сохраняем в переменную gamesdf и выводим на экран первые 30 строк таблицы

# In[2]:


games_df = pd.read_csv('/datasets/games.csv')


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

# In[9]:


games_df.columns = games_df.columns.str.lower()


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

# In[14]:


len(games_df['name'].unique())


# ### Проверяем количество дубликатов, их нет, идем дальше

# In[15]:


games_df.duplicated().sum()

# Обработаем значения tbd (to be decided or to be determined). Аббревиатура означает неопределенность, поэтому такие значения можно заполнить NaN, также нельзя измерить тип данных.

# In[16]:


games_df.query('user_score == "tbd"')


# In[17]:


games_df.loc[games_df['user_score']=="tbd", 'user_score']=np.nan
games_df['user_score'] = games_df['user_score'].astype('float64')

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

# ### Обработка рейтинга ESRB

# In[25]:


games_df['rating'].unique()

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

platform_sales = games_df.pivot_table(inde​
713
# <div class="alert alert-success">
714
# <font size="4", color= "seagreen"><b>✔️ Комментарий ревьюера</b></font>
715
#     <br /> 
716
#     <font size="3", color = "black">
717
# <br />Важно удалить пропуски и «заглушки» перед проведением теста, молодец
718
x='platform', values='total_sales', aggfunc='sum')

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

# In[44]:


actual_period = games_df.query(' year_of_release > 2012')
actual_period.head()

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
# In[49]:


# check
games_df.query("year_of_release == 1985 and platform == 'DS'")

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

# In[62]:


# check
actual_period

# ### Выводы

# **Из диаграммы рассеяния** следует что отзывов пользователей не влияют сушественно на продажи по скольку практически весь диапазон отзивов не меняет тенденция на продаж. Поэтому величина коэффициента коррелляции находится около нуля.
# Так же можно отметить, что **положительные отзывы критиков** влияют на продажи. Коэффициент коррелляции продаж от отзывов критиков больше, чем коэффициент корреляции продажи от отзывов пользователей. 
# **Средний возраст платформы** - 11 лет. Долгожитель - PC.
# **PS2 X360 PS3** - лидеры по продажам. Скорее всего, в списке получились две приставки от одной компнаии - возможно, как раз это связано с недолгим уровнем жизни платформ - когда выходит новая консоль, старая продолжает существовать какое то время и игры продолжаются продаваться, затем поколение умирает. X360 составляет отличную конкуренцию ps3.
# **Наиболее актуальный период** - от 2012 до 2016 года. Так как после 2011 года продажи резко упали в два раза. **Перспективные жанры** - Action, Sporsts, Shooter. В Японии вместо шутеров предпочитают role-playing

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
