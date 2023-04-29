#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# ### Откройте файл с данными и изучите общую информацию. 

# Импортируем необходимые библиотеки

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


# Считаем файл, задаем формат отображения таблицы

# In[2]:


flats = pd.read_csv('/datasets/real_estate_data.csv', sep='\t')
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.max_columns', None)


# Выводим полную информацию о таблице на экран

# In[3]:


flats.info()


# Испольузем методы head(),tail(),sample() для вывода значений в таблице, чтобы изучить данные в ней

# In[4]:


flats.head()


# In[5]:


flats.tail()


# In[6]:


flats.sample()


# In[7]:


flats.duplicated().sum()



# Переименуем названия столбиков на более адекватные

# In[8]:


flats.columns = ['total_images', 'last_price','total_area','first_day_exposition','rooms','ceiling_height','floors_total','living_area','floor','is_apartment','studio','open_plan','kitchen_area','balcony','locality_name','airports_nearest','city_centers_nearest','parks_around_3km','parks_nearest','ponds_around_3km','ponds_nearest','days_exposition']
    
        
        
        
        
        


# Убедимся, что столбики переименованы 

# In[9]:


flats.head()


# Отсортируем количество пропусков в каждом столбце

# In[10]:


flats.isnull().sum().sort_values()


# На самом деле методы дальше не играют большой роли, но для более детально просмотра их можно использовать. describe() для статистики (дальше понадобиться), shape для понимания как выглядит таблица, dtypes и dtypes.value_counts для того, чтобы понять столбики каких типов в таблице и сколько их

# In[11]:


flats.describe()


# In[12]:


flats.shape


# In[13]:


flats.dtypes


# In[ ]:





# In[14]:


flats.dtypes.value_counts()


# In[15]:


flats.hist(figsize=(15, 20))
plt.show()

# In[16]:


print("Количество квартир с потолками выше 15 метров:", flats[flats['ceiling_height'] > 15].count()[0])


# In[17]:


flats.drop(flats.loc[flats['ceiling_height'] > 15].index, inplace=True)


# In[18]:


print("Количество квартир с потолками выше 15 метров:", flats[flats['ceiling_height'] > 15].count()[0])


# ### Предобработка данных

# Ищем отсуствующие значения

# In[19]:


flats.isna().sum()

# Приступаем к анализу необычных значений в столбиках

# Смотрим уникальныке значения по названиям городов

# In[20]:


len(flats['locality_name'].unique())


# In[21]:


print(flats['locality_name'].unique().tolist())


# Есть проблемы с буквой ё, меняем на е, плюс поселок и поселок городского типа по рекомендациям в описании это одно и тоже, поэтому меняем и их

# In[22]:


flats['locality_name'] = flats['locality_name'].str.replace('ё', 'е', regex=True)


# In[23]:


flats['locality_name'] = flats['locality_name'].str.replace('поселок городского типа', 'поселок', regex=True)

# Снова считаем уникальные значения и пропуски

# In[24]:


len(flats['locality_name'].unique())


# In[25]:


flats['locality_name'].isna().sum()


# In[26]:


flats['locality_name'].isna().sum() * 100 / len(flats['locality_name'])


# In[27]:


flats = flats.dropna(subset=['locality_name'])


# In[28]:


flats['locality_name'].isna().sum()


# Проверяем is_apartment на уникальные значения

# In[29]:


flats['is_apartment'].unique()


# Заполняем пропущенные значения в is_apartment нулями, так как либо является апартаментом, либо нет. Не является - 0.

# In[30]:


flats['is_apartment'] = flats['is_apartment'].fillna('False')

# In[31]:


flats['is_apartment'] = flats['is_apartment'].map({True: 1, False: 0})


# Снова проверям таблицу

# In[32]:


flats.head()


# Ищем пропуки в ceiling_light и заполняем медианным значением

# In[33]:


flats['ceiling_height'].isnull().sum()


# In[34]:


flats['ceiling_height'].value_counts()


# In[35]:


flats['ceiling_height'].median()


# In[36]:


flats['ceiling_height'].fillna(flats['ceiling_height'].median(), inplace=True)


# Проверяем пропуски

# In[37]:


flats['ceiling_height'].isna().sum()


# Заполняем пропущенные значения в balcony нулем

# In[38]:


flats['balcony'].fillna(0, inplace=True)

# In[39]:


flats['balcony'].isna().sum()


# Заполняем медианным значением floors_total и проверяем правильность заполнения

# In[40]:


#flats['floors_total'].median()


# In[41]:


flats['floors_total'] = flats['floors_total'].fillna(flats['floor'])

# In[42]:


flats['floors_total'].dropna(inplace=True)


# In[43]:


flats['floors_total'].isna().sum()


# Заполняем kitchen_area медианным значением

# In[44]:


flats['kitchen_area'].median()


# In[45]:


flats['kitchen_area'].fillna(flats['kitchen_area'].median(), inplace=True)


# Переводим firts_day_exposition в понятный формат

# In[46]:


flats['first_day_exposition'] = pd.to_datetime(flats['first_day_exposition'], format="%Y-%m-%dT%H:%M:%S")


# In[47]:


flats.head(10)


# Заполняем airports_nearest средним значением

# In[48]:


flats['airports_nearest'].mean()


# In[49]:


flats['airports_nearest'].fillna(flats['airports_nearest'].mean(), inplace=True)


# Посмотрим еще раз и убедимся что пропуски заполнены в нужных местах

# In[50]:


flats.isna().sum()


# In[51]:


flats['floors_total'] = flats['floors_total'].astype('int64')


# In[52]:


flats['balcony'] = flats['balcony'].astype('int64')


# In[53]:


flats['last_price'] = flats['last_price'].astype('int64')


# In[54]:


flats['parks_around_3km'] = flats['parks_around_3km'].astype('Int64')


# In[55]:


flats['ponds_around_3km'] = flats['ponds_around_3km'].astype('Int64')

# In[56]:


flats.head()


# In[57]:


flats.describe()

# ### Посчитайте и добавьте в таблицу новые столбцы

# Добавляем цену одного квадратного метра

# In[58]:


flats['price_per_meter'] = flats['last_price'] / flats['total_area']
flats['price_per_meter'] = flats['price_per_meter'].round(decimals=2)


# Добавляем тип этажа с помощью функции

# In[59]:


#тип этажа
def floor_status(row):
    if row['floor'] == 1:
        return 'первый'
    elif row['floor'] == row['floors_total']:
        return 'последний'
    else:
        return 'другой'


# In[60]:


flats['floor_status'] = flats.apply(floor_status, axis=1)


# Добавляем день недели, месяц и год

# In[61]:


flats['day'] = flats['first_day_exposition'].apply(lambda x: x.isoweekday())
weekdays = {0: 'пн', 1: 'вт', 2: 'ср', 3: 'чт', 
                4: 'пт', 5: 'сб', 6: 'вс'}
flats['day'] = flats['day'].map(weekdays)


# In[ ]:





# In[62]:


flats['month'] = flats['first_day_exposition'].apply(lambda x: x.month)


# In[63]:


flats['year'] = flats['first_day_exposition'].apply(lambda x: x.year)

# Категоризируем расстояние до центра города в километрах

# In[64]:


def  city_centre_km(row):
    if row['city_centers_nearest'] < 2500:
        return 'центр'
    elif row['city_centers_nearest'] < 5000:
        return 'рядом с центром'
    elif row['city_centers_nearest'] < 17000:
        return 'в черте города'
    elif row['city_centers_nearest'] >= 17000:
        return 'область'


# In[65]:


flats['city_centre_km'] = flats.apply(city_centre_km, axis=1)


# In[66]:


flats.head()


# In[67]:


flats['city_centers_nearest_km'] = flats['city_centers_nearest'] / 1000
flats['city_centers_nearest_km'] = flats['city_centers_nearest_km'].round()

# ### Проведите исследовательский анализ данных

# Изучаем параметры

# In[68]:


#flats[['total_area','living_area','kitchen_area','last_price','rooms','ceiling_height','floor','floor_status','l','city_centfloors_totaers_nearest','airports_nearest','parks_nearest','day','month']].describe()

# После изучения описательной таблицы, проведем анализ характеристик, для этого составим гистограммы для каждого условия по задаче

# In[69]:








flats[['total_area','living_area','kitchen_area','last_price','rooms','ceiling_height','floor','floor_status','floors_total','city_centers_nearest','airports_nearest','parks_nearest','day','month']].hist(bins = 70, figsize = (10,10), ec = 'black')
plt.show()







# In[70]:


flats[['total_area']].hist(bins = 30, figsize = (10,10), ec = 'black')
plt.show()

# In[71]:


flats[['living_area']].hist(bins = 30, figsize = (10,10), ec = 'black')
plt.show()


# In[72]:


flats[['kitchen_area']].hist(bins = 70, figsize = (10,10), ec = 'black')
plt.show()


# In[73]:


flats.plot(y = 'last_price', 
          kind = 'hist', 
          bins = 100, 
          grid=True, 
          figsize = (12,6), 
          range = (0,17000000))
plt.title("Flats price", y=1.02)
plt.xlabel("1.00 = 10 mln rub.", labelpad=20)
plt.show()


# In[74]:


flats[['rooms']].hist(bins = 70, figsize = (10,10), ec = 'black')
plt.show()


# In[75]:


flats.plot(y = 'ceiling_height', 
          kind = 'hist', 
          bins = 30, 
          grid=True, 
          figsize = (12,6), 
          range = (2,5))
plt.title("Ceiling height", y=1.02)
plt.xlabel("In meters", labelpad=10)
plt.show()


# In[76]:


flats[['floor']].hist(bins = 70, figsize = (10,10), ec = 'black')
plt.show()


# In[77]:


flats.plot(y = 'floors_total', 
          kind = 'hist', 
          bins = 30, 
          grid=True, 
          figsize = (12,6), 
          range = (2,5))
plt.title("floors_total", y=1.02)
plt.xlabel("in meters", labelpad=10)
plt.show()


# In[78]:


flats[['city_centers_nearest']].hist(bins = 70, figsize = (10,10), ec = 'black')
plt.show()


# In[79]:


flats[['airports_nearest']].hist(bins = 70, figsize = (10,10), ec = 'black')
plt.show()


# In[80]:


flats[['parks_nearest']].hist(bins = 70, figsize = (10,10), ec = 'black')
plt.show()

# In[81]:


plt.figure(figsize=(10,10))    
flats.day.hist()
plt.xlabel('Sales days')
plt.title('Distribution of sales days')
plt.show()


# In[82]:


plt.figure(figsize=(10,10))    
flats.month.hist()
plt.xlabel('Sales months')
plt.title('Distribution of sales months')
plt.show()


# In[83]:


plt.figure(figsize=(10,10))    
flats.floor_status.hist()
plt.xlabel('Floor status')
plt.title('Distribution of floor statuses')
plt.show()

# Строим гистограмму для столбика days_exposition    

# In[84]:


plt.figure(figsize=(10,10))    
flats.total_area.hist()
plt.xlabel('Total area')
plt.title('Distribution of total area')
plt.xlim(0,150)
plt.show()


# In[85]:


plt.figure(figsize=(10,10))    
flats.days_exposition.hist(bins=360, ec = 'black')
plt.xlabel('Days from exposition')
plt.title('Distribution days from exposition')
plt.xlim(0,150)
plt.show()


# In[86]:


#flats['day'].describe()


# In[87]:


#flats['floor_status'].describe()

# Изучаем days_exposition на среднее и медианные значения

# In[88]:


flats['days_exposition'].describe()


#     1.Минимальное значение продажи квартиры - 1 день, что не очень правдободобно, так как процедура продажи квартиры уже после просмотра занимает значительное время.
#     2.Максимальное значение продаже квартиры - больше 4 лет, что является верным, так как квартиры могут продаваться и более длительный срок, в основном это зависит отрасположения и внешнего вида квартиры.
#     3.Cреднее время продажи квартиры в 95 дней - это снова зависит от расположения(насколько близко или далеко к центру города), возможно от других факторов типа метража квартиры. Плюс не стоит забывать про сам  процесс продажи квартиры - оформление, договора и т.д.
#     

#  **Задание 2. Какие факторы больше всего влияют на общую (полную) стоимость объекта?**

# Построим диаграммы корреляции, чтобы понять зависимость цены от каждой переменной

# In[89]:


sns.scatterplot(x='last_price', y='total_area',data=flats)
plt.title('Зависимость стоимостьи жилья от общей площади', fontsize= 12)
plt.show()

# In[90]:


sns.scatterplot(x='last_price', y='living_area',data=flats)
plt.title('Зависимость стоимостьи жилья от жилой площади', fontsize= 12)
plt.show()


# In[91]:


sns.scatterplot(x='last_price', y='kitchen_area',data=flats)
plt.title('Зависимость стоимостьи жилья от  площади кухни', fontsize= 12)
plt.show()


# In[92]:


sns.scatterplot(x='last_price', y='rooms',data=flats)
plt.title('Зависимость стоимостьи жилья от  количества комнат', fontsize= 12)
plt.show()


# In[93]:


sns.scatterplot(x='last_price', y='floor_status',data=flats)
plt.title('Зависимость стоимостьи жилья от типа этажа', fontsize= 12)
plt.show()

# In[94]:


sns.scatterplot(x='last_price', y='day',data=flats)
plt.title('Зависимость стоимостьи жилья от дня публикации', fontsize= 12)
plt.show()


# In[95]:


sns.scatterplot(x='last_price', y='year',data=flats)
plt.title('Зависимость стоимостьи жилья от года публикации', fontsize= 12)
plt.show()


# In[96]:


sns.scatterplot(x='last_price', y='month',data=flats)
plt.title('Зависимость стоимостьи жилья от года публикации', fontsize= 12)
plt.show()


# In[97]:


plt.figure(figsize=(15,15))
sns.heatmap(flats.corr(), annot=True, cmap="Reds", fmt='.2f')

# **ВЫВОДЫ**

# По heatmap видим, что цена сильно коррелирует с размером общей площади, жилой площади и с площадью кухни. Действительно, c гистограмм также убедились, что есть значимая зависимость между этими переменными.
# Вообще нет разницы, на сколько сильно квартира удалена от центра, в пригороде могут быть так и очень дорогие квартиры, так и очень дешевые.
# Зависимость цены за квадратный метр и количества комнат проявляется не сущетственно.

# **Посчитаем населеные пункты**

# In[98]:


flats['locality_name'].value_counts()    


# In[99]:


top_flats = flats[flats.locality_name.isin(flats.locality_name.value_counts().index[:10])]


# In[100]:


fig, ax = plt.subplots(figsize=(10, 5)) 

for locality in top_flats.locality_name.unique():
    sns.kdeplot(top_flats[top_flats.locality_name == locality].price_per_meter, label = locality)

plt.grid(True) # сетка
plt.legend(bbox_to_anchor = (2,2)) # положение легенды
plt.title('Распределение за квадратный метр по населенным пунктам', loc = 'left') # название графика
plt.xlabel('Цена за квадратный метр') # подпись оси x
plt.xlim((0,200000)) # ограничение значений оси X
plt.show()

# In[101]:


display(top_flats.pivot_table(index='locality_name', values='price_per_meter',  aggfunc='mean')
        .sort_values(by='price_per_meter')
        .round(2))

# In[102]:


def city_status(row):
    if row['locality_name'] == 'Санкт-Петербург':
        return 'Санкт-Петербург'
    else:
        return 'Пригород'


# In[103]:


flats['city_status'] = flats.apply(city_status, axis=1)


# In[104]:


flats.groupby('city_status').agg({'price_per_meter':'mean'}).sort_values(by='price_per_meter')


# **ВЫВОДЫ:**
# Самый дорогой город по цене квартир за квадратный метр - Санкт-Петербург (114849).B
# Самый дешевые квартиры в пригороде, а конкретнее - в Выборге (58141).

# **Определяем центр**

#  Добавляем столбец для обозначения расстояния в км 

# In[105]:


#flats['cityCenters_nearest_km'] = flats['cityCenters_nearest'] / 1000
#flats['cityCenters_nearest_km'] = flats['cityCenters_nearest_km'].round()

# In[106]:


display(flats.pivot_table(index='city_centers_nearest_km', values='price_per_meter',  aggfunc='mean')
        .sort_values(by='price_per_meter')
        .round(2)
        .hist())

# ### Общий вывод

#     В ходе анализа было выявлено, что средняя площадь квартир в объявлениях составляет 60,35 кв м. Стоимость - 6,5 млн руб. Количество комнат - 2. Высота потолка - 2,73 м. Среднее время продаж - 173 дня. В столбце со временем публикации наиболее частые значения - 45 и 60 дней, т.к. именно на столько дней по умолчанию размещаются объявления в сервисе Яндекс.Недвижимость.
#     Есть явная зависимость цены от размера площади квартиры.
#     Санкт-Петербург является самым дорогим городом - 104,4 тыс. руб. за 1 кв. метр. Топ-10 населенных пунктов замкнул Выборг, где стоимость почти в 2 раза меньше.
#     В центре площади квартир больше, больше всего квартир с 50, 70, 75 кв.м. так же есть подьем в районе 95 и 100 кв.м. Это объясняется тем, что в центре больше всего 3-х комнатных квартир, а после идут 2-х комнатные, после 4-х комнтатные, и 1-а комнатные и 5-и комнатные на равне, ситация за пределами центра совсем другая, больше всего 1 комнатных, псоле по порядке мягкий спад и резкий на 4-х комнатных их совсем мало за пределами центра.
#     В центре имеется больше намного больше квартир с высотой потолков выше 3-х метров, но во всех районах более популрным значением остается 2,7 метров.
#     Квартиры в центре продаются дольше чем квартиры за пределами центра, как мы видим по графикам, в пределах центра, квартиры чаще всего продаются в течении 45 дней, а так же после идет 60 дней. В центре же большинство квартир продатеся на 90-й день с момента появления объявления, а после идет цифра в 60 дней.
