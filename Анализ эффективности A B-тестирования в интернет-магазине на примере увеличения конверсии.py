#!/usr/bin/env python
# coding: utf-8

# **Денис, привет!**
# 
# Меня зовут Алексей Каргин, я буду проверять проект **Принятие решений в бизнесе**. Приятно познакомиться! Предлагаю общаться на «ты». Если это неприемлемо, будет здорово, если я об этом узнаю, и мы перейдем на «вы». Для удобства я оставлю комментарии в отдельных ячейках Markdown с заголовком «Комментарий ревьюера». Пожалуйста, не перемещай, не изменяй и не удаляй их - так наше общение будет более продуктивным. Я буду использовать цветовую разметку:
# 
# <div class="alert alert-danger">
# <b>Комментарий ревьюера</b> 
#     
# ✍ Так выделены самые важные замечания. Без их отработки проект не будет принят. Проверяя проект, я буду делать акцент не только на его решение и выполнение всех необходимых пунктов, но и на интерпретацию результата.
# </div>
# 
# <div class="alert alert-warning">
# <b>Комментарий ревьюера</b> 
#     
# 📝 Так выделены небольшие замечания или рекомендации. Постарайся учесть эти комментарии в этом проекте, а также взять что-то на будущее.
# </div>
# 
# <div class="alert alert-success">
# <b>Комментарий ревьюера</b> 
#     
# 👍 Так выделены все остальные комментарии, включая общие рекомендации, позитивные моменты или какие-то рассуждения и пояснения.
# </div>
# 
# Давай работать над проектом в диалоге: если ты что-то меняешь в проекте или отвечаешь на мои комментарии — пиши об этом. Мне будет легче отследить изменения, если ты выделишь свои комментарии:
# 
# <div class="alert alert-info"> 
# <b>Комментарий студента</b> 
#     
# Например, вот так. Также в таких блоках у меня можно спросить то, что осталось неясным.
# </div>
# 
# ---
# 
# <div class="alert alert-success">
# <b>Обратная связь v.1</b> 
#     
# 👋 Денис, спасибо, что прислал доработанную версию проекта. Как мне кажется, была проделана большая работа с проектом в ходе предыдущих итераций и теперь проект может быть принят! Спасибо за продуктивную работу. Посмотри, пожалуйста, я оставил дополнительные комментарии.
#     
# Отличного обучения да последующих модулях!
# 
#     
# <br>    
# С наилучшими пожеланиями, <br>
# Алексей
# </div>
# 
# 
# ---

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <b>Привет, Денис!👋
# 
# Меня зовут Эльвира, я буду ревьюером твоего проекта. Предлагаю общаться на «ты», но если это не удобно - дай мне знать, и мы перейдем на «вы».
# 
# 
# Ты можешь найти мои комментарии, обозначенные <font color='green'>зеленым</font>, <font color='gold'>желтым</font> и <font color='red'>красным</font> цветами, например:
# 
# 
# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊<a class="tocSkip"> </h2>
#     
# Такими комментариями я буду помечать отлично проделаную работу😉
# </div>
#     
# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
#         
# В таких комментариях я постараюсь подсказать тебе более элегантное или легкое решение, некоторые хитрости и фишки. Части проекта, помеченные такими комментариями, можно не исправлять, но рекомендую обратить на них внимание.</div>
# 
# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔<a class="tocSkip"></h2>
#     
# В случае, когда решение на отдельном шаге требует существенной переработки и внесения правок. Если ты видишь такой комментарий, значит здесть есть недочет, который следует исправить.
# </div>
# 
# Ты также можешь реагировать на мои комментарии своими, выделяя их цветами и наиболее понравившимся тебе способом оформления, но явно  отличающимся от моих комментариев. Это нужно, чтобы не создавалась путаница🙃
#     
# <div class="alert alert-info"> <b>Комментарий студента:</b> Например, вот так.</div>
# 
# Чтобы сделать подобный блок, кликни здесь дважды и скопируй всю предыдущую строку ;)
#     
# Пожалуйста, не удаляй и не перемещай мои комментарии, они будут особенно полезны для нашей работы в случае повторной проверки проекта.</div></b>

# # <h1>Анализ эффективности A/B-тестирования в интернет-магазине на примере увеличения конверсии<span class="tocSkip"></span></h1><div class="toc"><ul class="toc-item"></ul></div>

# Цель исследования: изучить результаты A/B-теста и принять решение о внедрении изменений на сайте.

# Задачи исследования:
# 
# Провести исследовательский анализ данных и оценить качество данных.
# Проверить корректность проведения теста и обнаружить возможные ошибки.
# Определить статистическую значимость различий между группами по основным метрикам: конверсии и среднему чеку.
# Изучить поведение пользователей и выявить различия в их сценариях поведения.
# Сделать выводы по результатам теста и принять решение о внедрении изменений на сайте.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Хорошо, что в начале есть вводная часть. В этой части можно также детализировать цель АВ теста (что было изучение изменения размера среднего чека и числа заказов на пользователя, а не конверсии).
# </div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊<a class="tocSkip"> </h2>
#     
# Очень здорово, что ты начинаешь проект с такого подробного введения, так держать!
# </div>

# ## Приоритизация гипотез

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu


# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊<a class="tocSkip"> </h2>
# 
# Молодец, что загружаешь все библиотеки в начале проекта. Так сложней случайно удалить ячейку с загрузкой нужной библиотеки и сделать код не работающим. А получатель отчета сразу поймет, какие библиотеки ты используешь и какие ему нужно установить для запуска проекта.
# 
# </div>

# In[2]:


hypothesis = pd.read_csv('/datasets/hypothesis.csv')
display(hypothesis.head())


# In[3]:


pd.options.display.max_colwidth = 130


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
#         
# Расширить вывод содержимого ячеек мы можем с помощью следующей настройки, так мы сможем увидеть полный текст гипотез:
#     
# ```python
# pd.options.display.max_colwidth = 130
# ```
# </div>

# <div class="alert alert-info"> <b>Добавил вывод всего содержимого</b></div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_2 <a class="tocSkip"> </h2>
# 
# 👍

# In[4]:


hypothesis['ICE'] = (hypothesis['Impact'] * hypothesis['Confidence']) / hypothesis['Efforts']
display(hypothesis[['Hypothesis', 'ICE']].sort_values(by='ICE', ascending=False).round(2))


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
#         
# Будет лучше округлить значения до 2 знаков после запятой. </div>   

# <div class="alert alert-info"> <b>Округлил</b></div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_2 <a class="tocSkip"> </h2>
# 
# 👍

# In[5]:


hypothesis['RICE'] = (hypothesis['Reach'] * hypothesis['Impact'] * hypothesis['Confidence']) / hypothesis['Efforts']
display(hypothesis[['Hypothesis', 'RICE']].sort_values(by='RICE', ascending=False))


# ### Выводы по гипотезам

# При применении фреймворка RICE вместо ICE произошли изменения в приоритизации гипотез. Некоторые гипотезы поднялись выше или опустились ниже в списке приоритетов.
# 
# Так, например, гипотеза № 8 "Запустить акцию, дающую скидку на товар в день рождения" при применении фреймворка ICE занимала первое место по приоритету, но при использовании фреймворка RICE она опустилась до 5-го места. Это связано с тем, что охват пользователей (Reach) у этой гипотезы равен 1, что сильно влияет на ее общий приоритет при использовании фреймворка RICE.
# 
# Таким образом, при применении фреймворка RICE, гипотезы, которые имеют высокий охват пользователей, могут получить более высокий приоритет, чем гипотезы, которые имеют более высокий потенциал влияния на пользователей, но низкий охват.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 В этом блоке была сделана хорошая работа - гипотезы приоритизировоаны по двум фреймворкам, указано, по каким критериям они различаются. Как рекомендация - можно также отобразить значения двух фреймворков в одной таблице, а в завершении указать гипотезы, которые ты считаешь наиболее всего приоритетными. Это будет хорошей рекомендаций заказчику.
# </div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊<a class="tocSkip"> </h2>
#     
# Приоритизация гипотез проведена успешно! Молодец, что выводишь таблицу дважды, каждый раз с сортировкой по одному из фреймоврков, так различия в их работе будут максимально наглядными. Согласна с твоими выводами по поводу их различия. Будет здорово добавить в начало небольшое описание фремворков. 
# </div>   

# ## Подготовка данных

# In[6]:


orders = pd.read_csv('/datasets/orders.csv')
visitors = pd.read_csv('/datasets/visitors.csv')


# In[7]:


# проверка наличия пропущенных значений
orders.isna().sum()
visitors.isna().sum()

# проверка наличия дубликатов
orders.duplicated().sum()
visitors.duplicated().sum()

# проверка типов данных
orders['date'] = pd.to_datetime(orders['date'])
visitors['date'] = pd.to_datetime(visitors['date'])


# In[8]:


orders.head(10)


# In[9]:


visitors.head(10)


# In[10]:


hypothesis.info()


# In[11]:


groups = orders['group'].unique()
print('Количество групп в А/В-тесте:', len(groups))


# In[12]:


# временной интервал теста
print('Дата начала теста:', orders['date'].min())
print('Дата окончания теста:', orders['date'].max())


# In[13]:


visitors_by_group = visitors.groupby('group').agg({'visitors': 'sum'})
print(visitors_by_group)


# In[14]:


visitors_by_day = visitors.groupby(['date', 'group']).agg({'visitors': 'sum'}).reset_index()
display(visitors_by_day.head())


# <div class="alert alert-warning">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 📝 Обрати, пожалуйста, внимание, что при проведении АВ тестов важно проследить, чтобы один и тот же пользователь не участвовал как в тестовой, так и в контрольной группах теста. Это важно, поскольку, в этом случае мы не знаем, что могло повлиять на результат или действие пользователя.
#     
# Чтобы проверить этот момент, мы можем сгруппировать данные по номеру пользователя и подсчитать количество уникальных групп. Это пригодится в последующих проектах.
#     
# В этом кейсе стоит отметить, что мы сможем проверить наличие таких пользователей только по данным по покупкам, тогда как данные по визитам агрегированы по дням. Здорово, что контролируешь итоговые результаты - проверяешь значения исходных данных.
# 
# 
# </div>

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
#         
# Можно провести дополнительные исследования, например: 
#     
# * Посмотреть сколько у нас групп в АВ-тесте;
# * Изучить временной интервал, узнав даты начала и окончания теста;
# * Рассмотреть количество пользователей в каждой группе - по таблице с заказами;
# * Посмотреть не попадают ли какие-то пользователи в обе группы - по таблице с заказами.
# * Посмотреть динамику посетителей по дням по группам (visitos).
# </div>

# <div class="alert alert-info"> <b>Добавил показатели + провел доп исследования </b></div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_2 <a class="tocSkip"> </h2>
# 
# 👍

# In[15]:


datesGroups = orders[['date','group']].drop_duplicates()


# In[16]:


ordersAggregated = datesGroups.apply(lambda x: orders[np.logical_and(orders['date'] <= x['date'], orders['group'] == x['group'])]                                     .agg({'date' : 'max', 'group' : 'max', 'transactionId' : pd.Series.nunique, 'visitorId' : pd.Series.nunique, 'revenue' : 'sum'}), axis=1)                                     .sort_values(by=['date','group'])
visitorsAggregated = datesGroups.apply(lambda x: visitors[np.logical_and(visitors['date'] <= x['date'], visitors['group'] == x['group'])]                                        .agg({'date' : 'max', 'group' : 'max', 'visitors' : 'sum'}), axis=1)                                        .sort_values(by=['date','group'])


# In[17]:


cumulativeData = ordersAggregated.merge(visitorsAggregated, left_on=['date', 'group'], right_on=['date', 'group'])
cumulativeData.columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
#         
# Можно проверить, корректно ли был содан датафрейм cummulativeData. Например, совпадают ли минимальная и максимальная даты в этом датафрейме с минимальной и максимальной датой в исходных данных. </div>   

# <div class="alert alert-info"> <b>Сделал проверку, все корректно </b></div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_2 <a class="tocSkip"> </h2>
# 
# Отлично)

# In[18]:


cumulativeRevenueA = cumulativeData[cumulativeData['group']=='A'][['date','revenue', 'orders']]


# In[19]:


cumulativeRevenueB = cumulativeData[cumulativeData['group']=='B'][['date','revenue', 'orders']]


# In[20]:


# Сравнение минимальных дат
print(orders['date'].min() == cumulativeData['date'].min())

# Сравнение максимальных дат
print(orders['date'].max() == cumulativeData['date'].max())


# Минимальная и максимальная даты в cummulativeData совпадают с соответствующими значениями в orders, что подтвердждает корректность создания датафрейма cummulativeData.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Здорово, что контролируешь итоговые результаты - проверяешь значения исходных данных.
# </div>

# ### Выводы по данным

# Данные в целом выглядят корректными и содержат необходимые для анализа параметры. Однако, в датасете orders.csv были обнаружены выбросы (заказы с очень высокой стоимостью), которые были удалены из дальнейшего анализа. Кроме того, данные по датам в обоих датасетах необходимо было привести к типу datetime для более удобной работы с ними.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Хорошо, что есть промежуточные выводы с фиксацией основных ошибок или проблем с данным. Только немного неясным остался вопрос с выбросами - в выводах указано, что они удалены, но выше, в самом коде, на этом не было сделано акцента.
# 
# </div>

# ## Графики

# In[21]:


cumulativeData['conversion'] = cumulativeData['orders'] / cumulativeData['visitors']

cumulativeConversionA = cumulativeData[cumulativeData['group']=='A'][['date','conversion']].reset_index(drop=True)
cumulativeConversionA['cumulativeConversion'] = cumulativeConversionA['conversion'].cumsum()
cumulativeConversionB = cumulativeData[cumulativeData['group']=='B'][['date','conversion']].reset_index(drop=True)
cumulativeConversionB['cumulativeConversion'] = cumulativeConversionB['conversion'].cumsum()


# In[22]:


# построение графика кумулятивной конверсии по группам
plt.figure(figsize=(12,5))
plt.plot(cumulativeConversionA['date'], cumulativeConversionA['cumulativeConversion'], label='A')
plt.plot(cumulativeConversionB['date'], cumulativeConversionB['cumulativeConversion'], label='B')
plt.title('Кумулятивная конверсия по группам')
plt.xlabel('Дата')
plt.ylabel('Кумулятивная конверсия')
plt.legend()
plt.show()


# На графике кумулятивной конверсии по группам видно, что группа B показывает лучший результат, чем группа A. В начале теста кумулятивная конверсия группы B была ниже, чем у группы A, но затем она начала расти и стабильно превышает конверсию группы A.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔<a class="tocSkip"></h2>
#     <s>
# Сейчас при анализе графика мы не учитываем, что на графике виден рост выручки. Если упростить, то выручка = конверсия * средний чек. Если выручка выросла, то мог увеличится средний чек, а могло увеличится количество заказов (т.е. конверсия).
# 
# Здесь нужно  дать полное описание возможных причин такого поведения графика. Если мы предлагаем только одну причину скачка, можно предположить, что мы не вполне понимаем всю картину.
# 
# 

# <div class="alert alert-info"> <b>Добавил дополнительно график конверсии </b></div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_2 <a class="tocSkip"> </h2>
# 
# Отлично)

# In[23]:


plt.figure(figsize=(12,5))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue'], label='A')
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue'], label='B')
plt.title('Кумулятивная выручка по группам')
plt.xlabel('Дата')
plt.ylabel('Кумулятивная выручка')
plt.legend()
plt.show()


# На графике кумулятивной выручки по группам видно, что группа B опережает группу A по выручке на протяжении большей части теста. В середине теста произошел резкий скачок кумулятивной выручки группы B, который можно объяснить возможным появлением крупного заказа. На конец теста группа B оказалась лидером по кумулятивной выручке.

# In[24]:


plt.figure(figsize=(12,5))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue']/cumulativeRevenueA['orders'], label='A')
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue']/cumulativeRevenueB['orders'], label='B')
plt.title('Кумулятивный средний чек по группам')


# <div class="alert alert-warning">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 📝 Не забывай, пожалуйста, про оформление графиков - сейчас нет подписи по оси Y. Конечно название дает представление о том, что отображено, но все же. Также лучше указывать размерность данных (руб, например).
#     
# Кстати, чтобы избежать вывод технической информации типа `Text(0.5, 1.0, 'Кумулятивный средний чек по группам')` можно использовать метод `plt.show()`.
# 
# 
# </div>

# График кумулятивного среднего чека показывает сильную нестабильность для обеих групп в начале теста. Затем средний чек группы A стабилизировался и оставался примерно на одном уровне, тогда как группа B также стабилизировалась после резкого скачка в середине теста. Группа B на конец теста имеет более высокий средний чек, но результаты по этому показателю менее стабильны, чем по кумулятивной выручке.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Все верно! В группе В можно зафиксировать выбросы в данных, которые влияют на результат.
# 
# </div>

# In[25]:


# отфильтруем данные по группам теста A и B
cumulative_revenue_a = cumulativeData[cumulativeData['group'] == 'A'][['date', 'revenue', 'orders']]
cumulative_revenue_b = cumulativeData[cumulativeData['group'] == 'B'][['date', 'revenue', 'orders']]

# соединим таблицы cumulative_revenue_a и cumulative_revenue_b по столбцу 'date'
merged_cumulative_revenue = cumulative_revenue_a.merge(cumulative_revenue_b, on='date', suffixes=['_a', '_b'])


# In[26]:


# объединение датафреймов с кумулятивными данными по группам A и B
mergedCumulativeRevenue = cumulativeRevenueA.merge(cumulativeRevenueB, left_on='date', right_on='date', how='left', suffixes=['A', 'B'])

# построение графика относительного изменения кумулятивного среднего чека группы B к группе A
plt.figure(figsize=(12,5))
plt.plot(mergedCumulativeRevenue['date'], (mergedCumulativeRevenue['revenueB']/mergedCumulativeRevenue['ordersB'])/(mergedCumulativeRevenue['revenueA']/mergedCumulativeRevenue['ordersA'])-1)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Относительное изменение кумулятивного среднего чека группы B к группе A')
plt.xlabel('Дата')
plt.ylabel('Относительное изменение')
plt.show()


# In[27]:


# рассчет кумулятивного среднего количества заказов на посетителя по группам
cumulativeData['order_per_visitor'] = cumulativeData['orders'] / cumulativeData['visitors']

cumulativeDataA = cumulativeData[cumulativeData['group'] == 'A']
cumulativeDataB = cumulativeData[cumulativeData['group'] == 'B']

plt.figure(figsize=(12,5))
plt.plot(cumulativeDataA['date'], cumulativeDataA['order_per_visitor'], label='A')
plt.plot(cumulativeDataB['date'], cumulativeDataB['order_per_visitor'], label='B')
plt.title('Кумулятивное среднее количество заказов на посететиля')


# <div class="alert alert-warning">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 📝 Не забывай, пожалуйста, про интерпретацию результатов - например, при описании этого графика важно отметить, что в целом значения стабилизировались во второй половине теста.
# 
# 
# </div>

# In[28]:


# объединение данных по группам A и B
cumulativeDataA = cumulativeData[cumulativeData['group']=='A'][['date','visitors']]
cumulativeDataA = cumulativeDataA.merge(cumulativeRevenueA, on='date')
cumulativeDataA['orders_per_visitor'] = cumulativeDataA['orders'] / cumulativeDataA['visitors']

cumulativeDataB = cumulativeData[cumulativeData['group']=='B'][['date','visitors']]
cumulativeDataB = cumulativeDataB.merge(cumulativeRevenueB, on='date')
cumulativeDataB['orders_per_visitor'] = cumulativeDataB['orders'] / cumulativeDataB['visitors']

# вычисление относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A
mergedCumulativeData = cumulativeDataA.merge(cumulativeDataB, on='date', suffixes=['A', 'B'])
mergedCumulativeData['ratio'] = mergedCumulativeData['orders_per_visitorB'] / mergedCumulativeData['orders_per_visitorA']

# построение графика относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A
plt.figure(figsize=(12,5))
plt.plot(mergedCumulativeData['date'], mergedCumulativeData['ratio'])
plt.axhline(y=1, color='black', linestyle='--')
plt.title('Относительное изменение кумулятивного среднего количества заказов на посетителя группы B к группе A')
plt.xlabel('Дата')
plt.ylabel('Отношение среднего количества заказов на посетителя группы B к группе A')
plt.show()


# На графике относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A можно увидеть, что кумулятивное среднее количество заказов на посетителя в группе B сначала было ниже, чем в группе A, но затем резко выросло и примерно на уровне 1.0-1.2 начало устанавливаться на отметке примерно 1.1. Это может указывать на наличие каких-то значимых изменений, влияющих на поведение пользователей, например, изменения цен, скидок или акций, изменения дизайна сайта, улучшения процесса оформления заказа и т.д

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 В этом блоке была проделана хорошая работа по сбору кумулятивных данных, работе с ними, визуализации (но не забывай, пожалуйста, про оформление графиков) и интерпретации результатов. Была предложена зависимость результатов группы В от выбросов.
# 
# 
# </div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊<a class="tocSkip"> </h2>
# 
# Графики построены и интерпретированы верно, молодец)
# </div>

# In[29]:


# подсчет количества заказов по пользователям
ordersByUsers = orders.groupby('visitorId', as_index=False).agg({'transactionId' : pd.Series.nunique})
ordersByUsers.columns = ['userId','orders']

# построение точечного графика количества заказов по пользователям
x_values = pd.Series(range(0,len(ordersByUsers)))
plt.figure(figsize=(8,6))
plt.scatter(x_values, ordersByUsers['orders'])
plt.title('Количество заказов по пользователям')
plt.xlabel('Пользователи')
plt.ylabel('Количество заказов')
plt.show()


# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Можно немного улучшить графику - показать разным цветом значения для групп А и В - тогда мы сможем определить, характерны аномальные данные для одной группы или для двух. 
#     
# https://moonbooks.org/Articles/How-to-create-a-scatter-plot-with-several-colors-in-matplotlib-/    
#     
# </div>

# На графике количество заказов на одного пользователя распределено довольно неравномерно. Большинство пользователей сделали только один заказ, но есть и те, кто сделал до 5 заказов, а также выбросы до 11 заказов на одного пользователя. Это может указывать на наличие аномальных пользователей, которые могут исказить результаты тестирования.

# In[30]:


ordersPerUser = orders.groupby('visitorId', as_index=False).agg({'transactionId' : pd.Series.nunique})
ordersPerUser.columns = ['visitorId','orders']


# In[31]:


plt.hist(ordersPerUser['orders'])
plt.title('Распределение количества заказов на пользователя')
plt.xlabel('Количество заказов на пользователя')
plt.ylabel('Частота')
plt.show()


# ## Расчет пецентилей

# In[32]:


print(np.percentile(ordersPerUser['orders'], [95, 99]))


# На основе расчета перцентилей мы можем сделать вывод, что:
# 
# 95% пользователей сделали не более 2 заказов;
# 99% пользователей сделали не более 4 заказов.

# Конкретно в нашем случае, для определения аномальных значений среднего чека и количества заказов в группах A и B были использованы следующие границы:
# 
# Средний чек: более 28 000 рублей в группе A и более 36 000 рублей в группе B;
# Количество заказов: более 3 заказов на одного пользователя.
# Если значение параметра превышало границу, то это считалось аномальным.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Лучше для группы А и В при тестах фиксировать единый порог определения выбросов.
# </div>

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
# Расчеты верные, а вот по выводам так и не ясно, что же мы будем принимать за аномалии. Можешь детализировать этот момент и указать принятые границы аномалий.
# </div>

# In[33]:


#создание датафрейма с информацией о каждом заказе и его стоимости
orderPrices = orders[['transactionId', 'revenue']]


# In[34]:


# построение точечного графика стоимостей заказов
plt.figure(figsize=(12,5))
plt.scatter(range(0,len(orderPrices)), orderPrices['revenue'])
plt.title('Стоимость заказов')
plt.xlabel('№ заказа')
plt.ylabel('Стоимость')
plt.show()


# Большинство заказов имеют стоимость до 20 000 у.е.
# Однако есть некоторое количество заказов со значительно более высокой стоимостью, которые можно рассматривать как потенциальные выбросы (например, заказы со стоимостью более 100 000 у.е.)

# In[35]:


# расчет 95-го и 99-го перцентилей стоимости заказов
percentiles = np.percentile(orders['revenue'], [95, 99])
print('95-й перцентиль стоимости заказов:', percentiles[0].round(2))
print('99-й перцентиль стоимости заказов:', percentiles[1].round(2))

# выбор границы для определения аномальных заказов
boundary = percentiles[1] # выбираем 99-й перцентиль в качестве границы
print('Граница для определения аномальных заказов:', boundary.round(2))


# Таким образом, граница для определения аномальных заказов будет  58233.2 рублей, что соответствует 99-му перцентилю стоимости заказов.

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
# 
# Использовать 95% перцентиль для отсечения выбросов - возможный вариант. Но в данном случае лучше использовать 99% перцентиль. Мы фильтруем по двум параметрам. А значит при последовательном отсечении 5%, мы отбросим больше 5%, а это не очень хорошо.

# <div class="alert alert-info"> <b> Будем использовать 99-ый перцентиль</b></div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 В этом блоке тоже все по делу - выбросы, которые могут влиять на результат (как это было выше видно по графикам) зафиксированы корректно. Границы выбросов также определены по значению 99 персентиля.
# </div>

# ## Сырые данные

# Для расчёта статистической значимости различий в среднем чеке заказа между группами по "сырым" данным воспользуемся непараметрическим тестом Манна-Уитни, так как выборки не распределены нормально и имеют выбросы.
# 
# Сформулируем нулевую гипотезу: различий в среднем количестве заказов на посетителя между группами нет. Альтернативная гипотеза: среднее количество заказов на посетителя в группах А и В различается.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Гипотезы в этом блоке и аналогичных сформулированы корректно. Возможно, стоит обратить внимание на то, что в тесте Манна-Уитни по факту мы сравниваем распределение данных, а вот t-тест или z-тест будут сравнивать средние значения.
# </div>

# In[36]:


orders_by_users_a = (
    orders[orders['group'] == 'A']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.sum})
)
orders_by_users_a.columns = ['visitor_id', 'orders']

orders_by_users_b = (
    orders[orders['group'] == 'B']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.sum})
)
orders_by_users_b.columns = ['visitorId', 'orders']


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_7 <a class="tocSkip"></h2>
# <s>
# Здесь нужно использовать уникальное количество в расчетах orders_by_users_a  и orders_by_users_b, зачем мы поменяли на сумму? 

# In[37]:


visitors_per_group = visitors.groupby('group')['visitors'].sum()


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_6 <a class="tocSkip"></h2>
# <s>
# Должна быть СУММА посетителей по каждой группе, а не уникальное кол-во. Куратор должен был предупредить, что дается 6 проверок на ревью. 

# In[38]:


# добавление sampleA и sampleB через pd.concat и учет 0 заказов по всем посетителям    
sampleA = pd.concat([orders_by_users_a['orders'], pd.Series(0, index=np.arange(visitors_per_group['A'] - len(orders_by_users_a['orders'])), name='orders')], axis=0)
sampleB = pd.concat([orders_by_users_b['orders'], pd.Series(0, index=np.arange(visitors_per_group['B'] - len(orders_by_users_b['orders'])), name='orders')], axis=0)

# расчет среднего количества заказов на посетителя для каждой группы
mean_orders_a = sampleA.mean()
mean_orders_b = sampleB.mean()

# расчет p-значения
p_value = stats.mannwhitneyu(sampleA, sampleB, alternative='two-sided')[1]

print('p-значение для сравнения среднего количества заказов на посетителя между группами по "сырым" данным: {:.3f}'.format(p_value))

if p_value < 0.05:
    print('Отвергаем нулевую гипотезу: среднее количество заказов на посетителя в группах А и В различается.')
else:
    print('Не удалось отвергнуть нулевую гипотезу: различий в среднем количестве заказов на посетителя между группами нет.')
print('Различие в среднем количестве заказов на посетителя между группами: {:.1%}'.format(mean_orders_b / mean_orders_a - 1))


# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Хороший подход к выводу информации результатов теста.
# 
# 
# </div>

# P-значение равное 0.017 означает, что вероятность получения различий в количестве заказов на посетителя между группами А и В равной или более выраженной, чем те, что мы наблюдаем, составляет 1.7%. Обычно, при выбранном уровне значимости 5%, такой результат является статистически значимым, и мы можем отвергнуть нулевую гипотезу о том, что средние значения количества заказов на посетителя в группах А и В равны.
# 
# Следовательно, можно сделать вывод, что есть статистически значимые различия в количестве заказов на посетителя между группами А и В, причем среднее значение в одной из групп значительно отличается от среднего значения в другой группе. 

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Результаты верные.
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔<a class="tocSkip"></h2>
#     <s>
# Расчет проведен неверно, прошу ознакомиться с материалом курса и скорректировать расчеты здесь и ниже, обновить выводы. Также необходимо добавить различие в средних значениях.

# <div class="alert alert-info"> <b>Исправил вычисления по среднемму количеству заказов и добавил расчет по средним чекам по сырым данным</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_2 <a class="tocSkip"> </h2>
# <s>
# К сожалению, расчет неверный, нам нужно использовать не просто ordersByGroupA , а добавить 0 заказов всем посетителям, которые ничего не купили, создав при этом переменные sampleA и sampleB. Давай скорректируем здесь и ниже и обновим выводы

# <div class="alert alert-info"> <b>Исправил вычисления по среднемму количеству заказов, добавил sampleA и sampleB</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_3 <a class="tocSkip"> </h2>
# <s>
# Допущена ошибка в расчетах. Прошу изучить материалы урока и скорректировать статистическую часть. Различие в  средних расчитывается как отношение двух средних показателей с вычетом 1. Нужно добавить по каждому показателю. Хочу обратить внимание, что если мы корректируем расчет по сырым данным, то по ОЧИЩЕННЫМ данным нам также нужно скорректировать расчеты, ведь они рассчитаны неверно. 

# <div class="alert alert-info"> <b>Исправил вычисления по среднемму количеству заказов</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_4 <a class="tocSkip"> </h2>
# <s>
# А где 0 заказов всем посетителям, которые ничего не купили?

# <div class="alert alert-info"> <b>Добавил через reindex(visitors, fill_value=0)</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_5 <a class="tocSkip"> </h2>
# <s>
# p_value в 1 нас не смущает? У нас осталась одна проверка ревью.  По учебному материалу нам нужно для расчетов использовать сумму визитов по каждой группе в переменной sampleA, и создавать ее через pd.concat(). 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_8 <a class="tocSkip"> </h2>
# 
# Вот этот расчет по среднему чеку верный, по сырым данным. Который ниже до зеленого комментария.

# Нулевая гипотеза: Средние значения суммы заказа не различаются между группами A и B.
# 
# Альтернативная гипотеза: Средние значения суммы заказа статистически значимо различаются между группами A и B.

# In[39]:


p_value1=stats.mannwhitneyu(orders[orders['group']=='A']['revenue'], orders[orders['group']=='B']['revenue'])[1]

print('p-value: {0:.3f}'.format(p_value1))

if p_value1 < 0.05:
    print('Отвергаем нулевую гипотезу: среднний чек на посетителя в группах А и В различается.')
else:
    print(
        'Не удалось отвергнуть нулевую гипотезу: различий в среднем чеке на посетителя между группами нет.') 
print('Различие в среднем чеке на посетителя между группами: {:.1%}'.format(orders[orders['group']=='B']['revenue'].mean()/orders[orders['group']=='A']['revenue'].mean()-1)) 


# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_8 <a class="tocSkip"> </h2>
# 
# Вот этот расчет по среднему чеку верный, по сырым данным.

# Рассчитанное значение p-value равно 0.729. Это означает, что вероятность получить такое же или более экстремальное различие в среднем чеке на посетителя между группами случайно равна 0.729. Таким образом, нет статистически значимых различий между группами.
# 
# Нулевая гипотеза не была отвергнута, что означает, что различий в среднем чеке на посетителя между группами нет.
# 
# Относительное различие в среднем чеке на посетителя между группами составляет 25.9%. Это может быть интересным наблюдением, но, учитывая вычисленное значение p-value, мы не можем считать эту разницу статистически значимой.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 Тут тоже все хорошо - несмотря на то, что мы наблюдаем различия между группами, они были получены случайным образом.
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔v_8 <a class="tocSkip"></h2>
# 
# Вывод по среднему количеству заказов, а расечт по среднему чеку

# <div class="alert alert-info"> <b> Изменил вывод, поменял количество заказов на средний чек</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_2 <a class="tocSkip"> </h2>
# <s>
# Нужно добавить различие в средних

# <div class="alert alert-info"> <b> Добавил различие</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_3  <a class="tocSkip"> </h2>
# <s>
# Расчет неверный.

# <div class="alert alert-info"> <b> Добавил расчет в переменную diff</b></div>

# ## Очищенные данные

# In[40]:


visitorsADaily = visitors[visitors['group'] == 'A'][['date', 'visitors']]
visitorsADaily.columns = ['date', 'visitorsPerDateA']

visitorsACummulative = visitorsADaily.apply(
    lambda x: visitorsADaily[visitorsADaily['date'] <= x['date']].agg(
        {'date': 'max', 'visitorsPerDateA': 'sum'}
    ),
    axis=1,
)
visitorsACummulative.columns = ['date', 'visitorsCummulativeA']

visitorsBDaily = visitors[visitors['group'] == 'B'][['date', 'visitors']]
visitorsBDaily.columns = ['date', 'visitorsPerDateB']

visitorsBCummulative = visitorsBDaily.apply(
    lambda x: visitorsBDaily[visitorsBDaily['date'] <= x['date']].agg(
        {'date': 'max', 'visitorsPerDateB': 'sum'}
    ),
    axis=1,
)
visitorsBCummulative.columns = ['date', 'visitorsCummulativeB']

ordersADaily = (
    orders[orders['group'] == 'A'][['date', 'transactionId', 'visitorId', 'revenue']]
    .groupby('date', as_index=False)
    .agg({'transactionId': pd.Series.nunique, 'revenue': 'sum'})
)
ordersADaily.columns = ['date', 'ordersPerDateA', 'revenuePerDateA']

ordersACummulative = ordersADaily.apply(
    lambda x: ordersADaily[ordersADaily['date'] <= x['date']].agg(
        {'date': 'max', 'ordersPerDateA': 'sum', 'revenuePerDateA': 'sum'}
    ),
    axis=1,
).sort_values(by=['date'])
ordersACummulative.columns = [
    'date',
    'ordersCummulativeA',
    'revenueCummulativeA',
]

ordersBDaily = (
    orders[orders['group'] == 'B'][['date', 'transactionId', 'visitorId', 'revenue']]
    .groupby('date', as_index=False)
    .agg({'transactionId': pd.Series.nunique, 'revenue': 'sum'})
)
ordersBDaily.columns = ['date', 'ordersPerDateB', 'revenuePerDateB']

ordersBCummulative = ordersBDaily.apply(
    lambda x: ordersBDaily[ordersBDaily['date'] <= x['date']].agg(
        {'date': 'max', 'ordersPerDateB': 'sum', 'revenuePerDateB': 'sum'}
    ),
    axis=1,
).sort_values(by=['date'])
ordersBCummulative.columns = [
    'date',
    'ordersCummulativeB',
    'revenueCummulativeB',
]

data = (
    ordersADaily.merge(
        ordersBDaily, left_on='date', right_on='date', how='left'
    )
    .merge(ordersACummulative, left_on='date', right_on='date', how='left')
    .merge(ordersBCummulative, left_on='date', right_on='date', how='left')
    .merge(visitorsADaily, left_on='date', right_on='date', how='left')
    .merge(visitorsBDaily, left_on='date', right_on='date', how='left')
    .merge(visitorsACummulative, left_on='date', right_on='date', how='left')
    .merge(visitorsBCummulative, left_on='date', right_on='date', how='left')
)


# Нулевая гипотеза: Средние значения чека не различаются между группами A и B.
# 
# Альтернативная гипотеза: Средние значения чека статистически значимо различаются между группами A и B.

# In[41]:


ordersByUsersA = (
    orders[orders['group'] == 'A']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersA.columns = ['visitorId', 'orders']

ordersByUsersB = (
    orders[orders['group'] == 'B']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersB.columns = ['visitorId', 'orders']

sampleA = pd.concat(
    [
        ordersByUsersA['orders'],
        pd.Series(
            0,
            index=np.arange(
                data['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)

sampleB = pd.concat(
    [
        ordersByUsersB['orders'],
        pd.Series(
            0,
            index=np.arange(
                data['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)

alpha = .05

p_value=stats.mannwhitneyu(sampleA, sampleB)[1]

print("p-value: {0:.3f}".format(p_value))

if p_value < alpha:
    print('Отвергаем нулевую гипотезу: между группами есть статистически значимые различия в среднем чеке заказа по "очищенным" данным.')
else:
    print(
        'Не удалось отвергнуть нулевую гипотезу: между группами нет статистически значимых различий в среднем чеке заказа по "очищенным" данным.'
    ) 
print("Различие в среднем чеке заказа: {:.1%}".format(sampleB.mean() / sampleA.mean() - 1))


# Используя U-критерий Манна-Уитни на «очищенных» данных, мы получаем p-значение 0,017, что меньше уровня значимости альфа, равного 0,05. Таким образом, мы можем отвергнуть нулевую гипотезу, утверждающую, что нет существенной разницы в средних значениях чека между группами А и В. Между группами A и B есть статистически значимые различия в среднем чеке заказа по сырым данным.
# Средний чек заказа в группе B оказался на 13.8% выше, чем в группе A.
# Эти результаты подтверждают необходимость проведения A/B-теста для проверки различных гипотез, в данном случае гипотезы о том, какие изменения на сайте приведут к увеличению конверсии и выручки.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔v_9<a class="tocSkip"></h2>
# 
# Денис, у нас выводы не соответствуют расчетам, мы считаем средний чек, а пишем количество заказов. Сверь пожалуйста выводы и расчеты. Два расчета верны, но выводы по ним указаны неверно.

# <div class="alert alert-info"> <b> Поменял выводы, добавил средний чек</b></div>

# <div class="alert alert-info"> <b> Перенес код сюда</b></div>

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера 🤓<a class="tocSkip"> </h2>
# 
# А этот тест у нас к чему относится?

# <div class="alert alert-info"> <b> Это сырые данные, проверка на выбросы в количестве заказов</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 <a class="tocSkip"></h2>
# 
# Но расчет неверный, подобный был ранее, сделан выше и отмечен как неправильный.

# <div class="alert alert-info"> <b> Удалил расчет</b></div>

# In[42]:


ordersByUsers = (
    orders.groupby('visitorId', as_index=False)
    .agg({'transactionId': 'sum'})
)

ordersByUsers.columns = ['visitorId', 'orders']


# In[43]:


# Вычисляем 99-й перцентиль распределения доходов заказов
quantile_value = np.percentile(orders['revenue'], 99)

# Вычисляем 99-й перцентиль распределения количества заказов
y = np.percentile(pd.concat([ordersByUsersA['orders'], ordersByUsersB['orders']], axis=0), 99)

# Находим аномальных пользователей
usersWithManyOrders = pd.concat(
    [ordersByUsersA[ordersByUsersA['orders'] > y]['visitorId'],
     ordersByUsersB[ordersByUsersB['orders'] > y]['visitorId']],
    axis=0
)

usersWithExpensiveOrders = orders[orders['revenue'] > quantile_value]['visitorId']

abnormalUsers = pd.concat([usersWithManyOrders, usersWithExpensiveOrders], axis=0).drop_duplicates().sort_values()

print(abnormalUsers.head(5))
print('Количество аномальных пользователей:', abnormalUsers.shape[0])


# Был проведен анализ для выявления аномальных пользователей в выборке.
# 
# Аномальными пользователями были определены те, кто совершил больше y заказов или совершил заказы с выручкой, превышающей значение квантиля.
# 
# В результате анализа были выявлены 15 аномальных пользователей

#     Нулевая гипотеза: среднее количество заказов на посетителя в группах A и B не различается
#     Альтернативная гипотеза: среднее количество заказов на посетителя в группах A и B различается

# In[48]:


sampleAFiltered = pd.concat(
    [
        ordersByUsersA[
            np.logical_not(ordersByUsersA['visitorId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                data['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)

sampleBFiltered = pd.concat(
    [
        ordersByUsersB[
            np.logical_not(ordersByUsersB['visitorId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                data['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)

p_value2 =stats.mannwhitneyu(sampleAFiltered, sampleBFiltered)[1]

print('p-value: {0:.3f} '.format(p_value2))

if p_value2 < 0.05:
    print('Отвергаем нулевую гипотезу: среднее количество заказов на посетителя в группах А и В различается.')
else:
    print('Не удалось отвергнуть нулевую гипотезу: различий в среднем количестве заказов на посетителя между группами нет.')
print('Различие в среднем количестве заказов на посетителя между группами: {:.1%}'.format(sampleBFiltered.mean()/sampleAFiltered.mean()-1))


# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_8 <a class="tocSkip"> </h2>
# 
# Вот этот расчет по КОНВЕРСИИ тоже верный, он у нас по ОЧИЩЕННЫМ данным.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_7 <a class="tocSkip"></h2>
# <s>
# Перед отправкой, необходимо перезапустить проект и убедиться, что все ячейки работают корректно. Перед отправкой проекта стоит проверять работоспособность кода — это можно сделать, нажав на панели Jupiter Hub ``Kernel`` и ``Restart & Run All`` (см скриншот ниже).
# ![](https://i.postimg.cc/yd19rYf6/Screenshot-428.png)

# На уровне значимости 0.05 есть статистически значимые различия в количестве заказов на одного посетителя между группами A и B. Таким образом, мы отвергаем нулевую гипотезу о том, что средние значения количества заказов на одного посетителя в группах A и B равны.
# 
# Различие в среднем количестве заказов на одного посетителя между группами составляет 15.3%. Это означает, что группа B имеет более высокий уровень среднего количества заказов на одного посетителя по сравнению с группой A. 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_8  <a class="tocSkip"></h2>
# <s>
# Вывод написан по среднему чеку, а расчет мы проводим по КОЛИЧЕСТВУ ЗАКАЗОВ. Необходимо скорректировать вывод.

# <div class="alert alert-info"> <b> Поменял вывод по тесту</b></div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_9 <a class="tocSkip"> </h2>
# 
# 👍

# Нулевая гипотеза: Среднее количество заказов на посетителя в группе А и В не различается.
# 
# Альтернативная гипотеза: Среднее количество заказов на посетителя в группе А и В различается.

# In[45]:


orders_clean = orders.query('visitorId not in @abnormalUsers')
revenue_A = orders_clean.query('group == "A"')['revenue']
revenue_B = orders_clean.query('group == "B"')['revenue']

mean_order_value_A = orders_clean.query('group == "A"')['revenue'].mean()
mean_order_value_B = orders_clean.query('group == "B"')['revenue'].mean()

relative_difference = (mean_order_value_B / mean_order_value_A - 1) * 100
statistic, pvalue = mannwhitneyu(revenue_A, revenue_B, alternative='two-sided')

print("p-value: {:.4f}".format(pvalue))
print("Средний чек в группе A: {:.2f}".format(mean_order_value_A))
print("Средний чек в группе B: {:.2f}".format(mean_order_value_B))
print("Относительная разница между средними чеками: {:.2f}%".format(relative_difference))


#   Значение p 0,8509 намного больше, чем обычно используемый альфа-уровень 0,05, что означает, что мы не можем отвергнуть нулевую гипотезу.   
#     Из результатов расчета видно, что после удаления аномалий средний чек в группе A - 6436.81, а в группе B - 6399.81. Относительная разница между средними чеками составила -0.57%, что говорит о том, что средний чек в группе B незначительно ниже, чем в группе A.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊 v_8 <a class="tocSkip"> </h2>
# 
# Этот расчет тоже верный, но он у нас по СЫРЫМ данным, его нужно перенести и расположить перед тестом по очищенным данным.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_8<a class="tocSkip"></h2>
# <s>
# Разве мы проводили фильтрацию в расчетах? Расчет по СЫРЫМ данным по КОЛИЧЕСТВУ ЗАКАЗОВ, а не среднему чеку. необходимо скорректировать вывод и перенести расчет выше расчета по очищенным данным. Ведь сначала нужно проверить сырые, а уже потом проводиь фильтрацию и делать повторный анализ.

# <div class="alert alert-info"> <b>Перенес код до очищенных данных</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_9<a class="tocSkip"></h2>
# 
# А как же  уровень значимости (p-value), нужно его рассчитать, это ведь самая главная часть анализа?

# <div class="alert alert-info"> <b>Добавил расчет p-value</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_8<a class="tocSkip"></h2>
# <s>
# У нас отсутствует расчет по очищенным данным по СРЕДНЕМУ ЧЕКУ, нужно его добавить с различием в средних и выводом.

# <div class="alert alert-info"> <b>Добавил расчет по среднему чеку по очищенным данным и выводы по результатам</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔<a class="tocSkip"></h2>
#     <s>
# Не нашла расчет по сырым данным по средним чекам) Нужно добавить и также расчет различия в средних значениях) Не забудь обновить выводы)

# <div class="alert alert-info"> <b>Добавил расчет, обновил выводы</b></div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_2 <a class="tocSkip"> </h2>
# <s>
# Расчет различия в средних нужно добавить по каждому параметру, и скорректировать тесты, обновить выводы.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера 🤔 v_3 <a class="tocSkip"> </h2>
# <s>
# Расчеты произведены неверно и нуждаются в доработке.

# <div class="alert alert-info"> <b> Добавил расчеты по очищенным данным</b></div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 В начале этого блока кумулятивные данные были собраны заново - этого можно было бы избежать и использовать уже подготовленные данные. Также тесты по исходным данным уже были выполнены, поэтому  в начале их тоже можно опустить.
#     
# Также здорово, что при фильтрации данных используешь значения персентилей напрямую.
# </div>

# ## Выводы

# Исходя из результатов тестирования, не удалось обнаружить статистически значимых различий в среднем количестве заказов на посетителя между группами, но были обнаружены небольшие (2%) статистически значимые различия в среднем чеке заказа между группами.
# 
# При таком раскладе рекомендуется остановить тест и зафиксировать отсутствие различий в среднем количестве заказов на посетителя между группами, а также отсутствие статистически значимых различий в конверсии между группами.
# 
# Однако, стоит обратить внимание на статистически значимые различия в среднем чеке заказа между группами. Возможно, имеет смысл провести дополнительные исследования, чтобы выявить причины этого различия и проверить гипотезы о возможных изменениях пользовательского поведения. Если такие исследования окажутся успешными, можно будет принять решение о внедрении изменений на сайте для увеличения среднего чека. В этом случае, продолжение теста может оказаться полезным.

# <div class="alert alert-success">
# <b>Комментарий ревьюера v.1 (Алексей)</b> 
#     
# 👍 В целом выводы корректны, только обрати, пожалуйста, внимание, что в целом тест проведен, различия в среднем чеке не были обнаружены. Поэтому, стоит скорректировать итоговый вывод. По поводу продолжения теста - тут важно понять, какая вероятность совершить ошибку второго рода и если она приемлема - то мы получили достоверные результаты, поэтому продолжать тест не имеет смысла. Это будет дополнительной тратой времени и денег.
# </div>

# ### Дополнительные выводы

# В рамках проекта был проведен анализ результатов A/B-теста, целью которого было увеличение выручки интернет-магазина.
# 
# В первой части проекта были проанализированы и приоритезированы гипотезы с помощью фреймворков ICE и RICE. В результате было выявлено, что наиболее приоритетными гипотезами являются те, которые связаны с изменением способов оплаты, добавлением скидок и акций для покупателей.
# 
# Во второй части проекта были проанализированы результаты A/B-теста. Были построены графики кумулятивной выручки, среднего чека и конверсии по группам. Было выявлено, что на протяжении всего теста группа B показывала лучшие результаты, но в середине теста был резкий скачок в кумулятивной выручке группы B, что привело к тому, что тест был остановлен и были проанализированы "очищенные" данные.
# 
# По результатам анализа очищенных данных можно сделать следующие выводы:
# 
# Есть статистически значимое различие по конверсии между группами A и B. Конверсия в группе B значительно выше, чем в группе A.
# 
# Нет статистически значимого различия по среднему чеку между группами A и B, однако стоит отметить, что в начале теста в группе B были аномально высокие значения среднего чека, которые позже снизились до уровня группы A.
# 
# График кумулятивной конверсии показывает, что конверсия в группе B стабильно выше, чем в группе A.
# 
# График кумулятивного среднего чека показывает, что средний чек в обеих группах колеблется, но в конце теста в группе B стал немного выше, чем в группе A.
# 
# Анализ аномалий показал, что есть пользователи, которые совершали аномально большое количество заказов или делали очень дорогие покупки. Эти пользователи могут искажать результаты теста, поэтому их нужно убрать из анализа.
# 
# Таким образом, на основе анализа данных можно сделать вывод, что группа B показывает лучшие результаты по конверсии, но нет статистически значимого различия между группами по среднему чеку. Однако, средний чек в группе B в конце теста стал немного выше, чем в группе A. Рекомендуется продолжить тестирование для подтверждения результатов.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 😊<a class="tocSkip"> </h2>
# 
# Содержательный вывод по проделанной работе, не забудь обновить его в соответствии с работой и возможно мы изменим решение по тесту.

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_9<a class="tocSkip"> </h2>
#     <br/>
#     
# Денис, выводы не соответствуют нашим расчетам, пожалуйста проверь и скорректируй их. Также нужно добавить расчет по среднему чеку по очищенным данным. 

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_8<a class="tocSkip"> </h2>
#     <br/>
#     
# Денис, у нас три верных расчета, и неправильных вывода по ним. Нужно скорректировать и добавить еще один расчет по очищенным данным СРЕДНЕМУ чеку. Код который не ограничен зелеными комментариями лучше или удалить или закомментировать, так как он вносит путанницу и сложно найти правильное решение и вывод к нему. Давай почистим, осталось уже совсем немного, надеюсь мы на финишной прямой.

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_7<a class="tocSkip"> </h2>
#     <br/>
# 1) Скорректировать расчет и по сырым и по тому же принципу по ОЧИЩЕННЫМ данным, т.е. оба расчета среднего количества заказов на пользователя нужно доработать 
#     
# 2) Обновить выводы
#     
# 3) Неработающие ячейки

# <div class="alert alert-info"> <b> Исправил неработающие ячейки, с расчетами все верно, выводы обновил 

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_6<a class="tocSkip"> </h2>
#     <br/>
#     
# Расчет по среднему кол-ву заказов по прежнему неверный. Нужно доработать ВСЕ пункты, различия в средних по кол-ву заказов на пользователя по очищенным так и нет. 
# 
# 1) Скорректировать расчеты по среднему количеству заказов на покупателя, обращаясь к учебному модулю;
#     
# 2) К КАЖДОМУ тесту, т.е. по кол-ву заказов нам тоже нужно добавить различие в средних значениях 
#     
# 3) Обновить все выводы
#     
# 4) Скорректировать решение по тесту в соответствии с полученными выводами
#     

# <div class="alert alert-info"> <b> Добавил изменения по тестам, выводы обновил, различия добавил к каждому тесту

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_5<a class="tocSkip"> </h2>
#     <br/>
#     
# Денис, у нас осталась последнее ревью работы, поэтому резюмирую ошибки, которые необходимо исправить:
#     
# 1) Скорректировать расчеты по среднему количеству заказов на покупателя, обращаясь к учебному модулю;
#     
# 2) К КАЖДОМУ тесту, т.е. по кол-ву заказов нам тоже нужно добавить различие в средних значениях 
#     
# 3) Обновить все выводы
#     
# 4) Скорректировать решение по тесту в соответствии с полученными выводами
#     
# 5) указать статистические гипотезы по среднему чеку

# <div class="alert alert-info"> <b> Расчеты обновил, добавил к каждому тесту различие и гипотезы, обновил выводы. Не очень понял, почему ревью последнее, так как на прошлом проекте давалось неограниченное количество попыток</b></div>

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_4<a class="tocSkip"> </h2>
#     <br/>
#     
# Денис, пожалуйста, изучи материалы урока, все расчеты нуждаются в дорботке.

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_3<a class="tocSkip"> </h2>
#     <br/>
#     
# Денис, пожалуйста изучи материалы урока и сорректируй расчеты, необходимо добавить различие в средних по КАЖДОМУ тесту, А также обновить расчеты как по СЫРЫМ так и по ОЧИЩЕННЫМ данным. Нам нужно использовать не просто посетителей, а сумму посетителей в расчетах переменных simple.  Не забудь обновить все выводы и решение по тесту.

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊 v_2<a class="tocSkip"> </h2>
#     <br/>
#     
# Работа нуждается в доработке)

# <div style="border:solid Purple 2px; padding: 40px">
# 
# <h2> Общий комментарий ревьюера 😊<a class="tocSkip"> </h2>
#     <br/>
#  Денис, ты хорошо поработал по проекту,  отлично справился с графиком)
#     
# Необходимо доработать:
#     
# * Вывести предобработку
# * Скорректировать расчеты
# * Обновить выводы
# * Остальные комментарии ты найдешь в работе
#     
#  
# **Желаю удачи и жду твой проект на повторное ревью! Если вдруг у тебя возникнут вопросы, то я с радостью отвечу на них.    😊**
#     
# ![gif](https://i.gifer.com/378.gif)
#     
# <br>   
#              
# **Дополнительные материалы:**
#  
# [Вебинары под эгидой Практикума](https://vk.com/yandex.praktikum?w=wall-176471180_2144), 
#     
# [Лекции Анатолия Карпова 1](https://www.youtube.com/watch?v=jnFVmtaeSA0&list=WL&index=19&t=2s) [2](https://www.youtube.com/watch?v=gljfGAkgX_o&list=WL&index=4)
#     
# Материалы [gopractice](https://gopractice.ru/summary/)
# 
# Лекцию Карта статистических методов [Смотреть видео](https://www.youtube.com/watch?v=-zps6hm0nX8&t=1269s)
#         
# </div>
#     

# In[ ]:




