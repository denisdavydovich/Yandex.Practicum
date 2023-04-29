#!/usr/bin/env python
# coding: utf-8


# 
# # Анализ эффективности A/B-тестирования в интернет-магазине на примере увеличения конверсии

# Цель исследования: изучить результаты A/B-теста и принять решение о внедрении изменений на сайте.

# Задачи исследования:
# 
# Провести исследовательский анализ данных и оценить качество данных.
# Проверить корректность проведения теста и обнаружить возможные ошибки.
# Определить статистическую значимость различий между группами по основным метрикам: конверсии и среднему чеку.
# Изучить поведение пользователей и выявить различия в их сценариях поведения.
# Сделать выводы по результатам теста и принять решение о внедрении изменений на сайте.



# ## Приоритизация гипотез

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu




# In[2]:


hypothesis = pd.read_csv('/datasets/hypothesis.csv')
display(hypothesis.head())


# In[3]:


pd.options.display.max_colwidth = 130



# In[4]:


hypothesis['ICE'] = (hypothesis['Impact'] * hypothesis['Confidence']) / hypothesis['Efforts']
display(hypothesis[['Hypothesis', 'ICE']].sort_values(by='ICE', ascending=False).round(2))


# In[5]:


hypothesis['RICE'] = (hypothesis['Reach'] * hypothesis['Impact'] * hypothesis['Confidence']) / hypothesis['Efforts']
display(hypothesis[['Hypothesis', 'RICE']].sort_values(by='RICE', ascending=False))


# ### Выводы по гипотезам

# При применении фреймворка RICE вместо ICE произошли изменения в приоритизации гипотез. Некоторые гипотезы поднялись выше или опустились ниже в списке приоритетов.
# 
# Так, например, гипотеза № 8 "Запустить акцию, дающую скидку на товар в день рождения" при применении фреймворка ICE занимала первое место по приоритету, но при использовании фреймворка RICE она опустилась до 5-го места. Это связано с тем, что охват пользователей (Reach) у этой гипотезы равен 1, что сильно влияет на ее общий приоритет при использовании фреймворка RICE.
# 
# Таким образом, при применении фреймворка RICE, гипотезы, которые имеют высокий охват пользователей, могут получить более высокий приоритет, чем гипотезы, которые имеют более высокий потенциал влияния на пользователей, но низкий охват.


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



# In[15]:


datesGroups = orders[['date','group']].drop_duplicates()


# In[16]:


ordersAggregated = datesGroups.apply(lambda x: orders[np.logical_and(orders['date'] <= x['date'], orders['group'] == x['group'])]                                     .agg({'date' : 'max', 'group' : 'max', 'transactionId' : pd.Series.nunique, 'visitorId' : pd.Series.nunique, 'revenue' : 'sum'}), axis=1)                                     .sort_values(by=['date','group'])
visitorsAggregated = datesGroups.apply(lambda x: visitors[np.logical_and(visitors['date'] <= x['date'], visitors['group'] == x['group'])]                                        .agg({'date' : 'max', 'group' : 'max', 'visitors' : 'sum'}), axis=1)                                        .sort_values(by=['date','group'])


# In[17]:


cumulativeData = ordersAggregated.merge(visitorsAggregated, left_on=['date', 'group'], right_on=['date', 'group'])
cumulativeData.columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']

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


# ### Выводы по данным

# Данные в целом выглядят корректными и содержат необходимые для анализа параметры. Однако, в датасете orders.csv были обнаружены выбросы (заказы с очень высокой стоимостью), которые были удалены из дальнейшего анализа. Кроме того, данные по датам в обоих датасетах необходимо было привести к типу datetime для более удобной работы с ними.



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




# График кумулятивного среднего чека показывает сильную нестабильность для обеих групп в начале теста. Затем средний чек группы A стабилизировался и оставался примерно на одном уровне, тогда как группа B также стабилизировалась после резкого скачка в середине теста. Группа B на конец теста имеет более высокий средний чек, но результаты по этому показателю менее стабильны, чем по кумулятивной выручке.


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


# ## Сырые данные

# Для расчёта статистической значимости различий в среднем чеке заказа между группами по "сырым" данным воспользуемся непараметрическим тестом Манна-Уитни, так как выборки не распределены нормально и имеют выбросы.
# Сформулируем нулевую гипотезу: различий в среднем количестве заказов на посетителя между группами нет. Альтернативная гипотеза: среднее количество заказов на посетителя в группах А и В различается.


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




# In[37]:


visitors_per_group = visitors.groupby('group')['visitors'].sum()


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



# P-значение равное 0.017 означает, что вероятность получения различий в количестве заказов на посетителя между группами А и В равной или более выраженной, чем те, что мы наблюдаем, составляет 1.7%. Обычно, при выбранном уровне значимости 5%, такой результат является статистически значимым, и мы можем отвергнуть нулевую гипотезу о том, что средние значения количества заказов на посетителя в группах А и В равны.
# 
# Следовательно, можно сделать вывод, что есть статистически значимые различия в количестве заказов на посетителя между группами А и В, причем среднее значение в одной из групп значительно отличается от среднего значения в другой группе. 



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




# Рассчитанное значение p-value равно 0.729. Это означает, что вероятность получить такое же или более экстремальное различие в среднем чеке на посетителя между группами случайно равна 0.729. Таким образом, нет статистически значимых различий между группами.
# 
# Нулевая гипотеза не была отвергнута, что означает, что различий в среднем чеке на посетителя между группами нет.
# 
# Относительное различие в среднем чеке на посетителя между группами составляет 25.9%. Это может быть интересным наблюдением, но, учитывая вычисленное значение p-value, мы не можем считать эту разницу статистически значимой.

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


# На уровне значимости 0.05 есть статистически значимые различия в количестве заказов на одного посетителя между группами A и B. Таким образом, мы отвергаем нулевую гипотезу о том, что средние значения количества заказов на одного посетителя в группах A и B равны.
# 
# Различие в среднем количестве заказов на одного посетителя между группами составляет 15.3%. Это означает, что группа B имеет более высокий уровень среднего количества заказов на одного посетителя по сравнению с группой A. 


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

#
# ## Выводы

# Исходя из результатов тестирования, не удалось обнаружить статистически значимых различий в среднем количестве заказов на посетителя между группами, но были обнаружены небольшие (2%) статистически значимые различия в среднем чеке заказа между группами.
# 
# При таком раскладе рекомендуется остановить тест и зафиксировать отсутствие различий в среднем количестве заказов на посетителя между группами, а также отсутствие статистически значимых различий в конверсии между группами.
# 
# Однако, стоит обратить внимание на статистически значимые различия в среднем чеке заказа между группами. Возможно, имеет смысл провести дополнительные исследования, чтобы выявить причины этого различия и проверить гипотезы о возможных изменениях пользовательского поведения. Если такие исследования окажутся успешными, можно будет принять решение о внедрении изменений на сайте для увеличения среднего чека. В этом случае, продолжение теста может оказаться полезным.

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
