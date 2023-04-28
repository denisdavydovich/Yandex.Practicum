#!/usr/bin/env python
# coding: utf-8



# # Изучение влияния изменения шрифта на конверсию в мобильном приложении
# 
# 

# ## Цель и задачи исследования

# Цель исследования – определить, как изменение шрифта влияет на конверсию в мобильном приложении

# Задачи исследования:
# * Собрать данные о конверсии в мобильном приложении
# * Разделить пользователей на экспериментальную группу и контрольные группы
# * Изменить шрифт в экспериментальной группе
# * Сравнить конверсию в экспериментальной группе с конверсией в контрольных группах
# * Сделать выводы о влиянии изменения шрифта на конверсию в мобильном приложении



# ## Подготовка данных

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy.stats as st 
import numpy as np
from scipy import stats
import math as mth


# In[4]:


data = pd.read_csv('/datasets/logs_exp.csv', sep = '\t')


# In[5]:


pd.set_option('display.precision',2)


# In[6]:


data.info()


# In[7]:


display(data.info())
display(data.head())
display(data.describe())
display('количество дублей:', data.duplicated().sum())


# In[8]:


data.columns = ['event_name', 'device_id', 'event_timestamp', 'group']


# In[9]:


print("Пропущенные значения в каждом столбце:")
print(data.isnull().sum())


# In[10]:


data['datetime'] = pd.to_datetime(data['event_timestamp'], unit='s')


# In[11]:


data['date'] = data['datetime'].dt.date


# In[12]:


data['date'] = data['date'].astype('datetime64')


# In[13]:


data[data.duplicated()==True].sort_values(by=['event_name', 'device_id', 'event_timestamp', 'group']).head(25)


# In[14]:


display(data.info())
display(data.head())
display(data.describe())


# ### Промежуточные выводы по данным

# Были произведены следующие изменения в данных:
# * изменены названия столбцов на более удобные и правильные
# * изменен тип столбца date
# * произведен поиск дубликатов – обнаружено 413 дублей
# * прозведен поиск пропущенных значений – пропуски не обнаружены


# ## Изучаем и проверям данные

# In[15]:


events_total = data.shape[0]
print("Всего событий в логе:", events_total)


# In[16]:


users_total = data['device_id'].nunique()
print("Всего пользователей в логе:", users_total)


# In[17]:


events_per_user = events_total / users_total
print("Среднее количество событий на пользователя:", events_per_user)


# In[18]:


print("Минимальная дата:", data['datetime'].min())
print("Максимальная дата:", data['datetime'].max())


# In[19]:



plt.hist(data['datetime'], bins=50)
plt.xlabel('Дата и время')

plt.ylabel('Количество событий')
plt.title('Гистограмма распределения событий по дате и времени')
plt.xticks(rotation='vertical')
plt.show()

# In[20]:


data = data[data['datetime'] >= '2019-08-01']


# In[21]:


events_lost = events_total - data.shape[0]
users_lost = users_total - data['device_id'].nunique()

print("Событий потеряно:", events_lost)
print("Пользователей потеряно:", users_lost)


# In[22]:


event_fraction_lost = events_lost / events_total * 100
user_fraction_lost = users_lost / users_total * 100

print("Доля потерянных событий: {:.2f}%".format(event_fraction_lost))
print("Доля потерянных пользователей: {:.2f}%".format(user_fraction_lost))

# In[23]:


data['group'].unique()


# In[24]:


# изучение событий в логах
events = data['event_name'].value_counts().reset_index()
events.columns = ['event_name', 'event_count']
print(events)


# In[25]:


# настройки графика
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

# гистограмма распределения количества событий по типам событий
sns.barplot(x='event_count', y='event_name', data=events, palette='rocket')

# название графика и осей
plt.title('Распределение количества событий по типам событий')
plt.xlabel('Количество событий')
plt.ylabel('Тип события')

# вывод графика
plt.show()


# In[26]:


# кол-во пользователей, совершивших каждое событие
users_per_event = data.groupby('event_name')['device_id'].nunique().reset_index()
users_per_event.columns = ['event_name', 'unique_users']
print(users_per_event)


# In[27]:


users_per_event['user_ratio'] = users_per_event['unique_users'] / data['device_id'].nunique()
print(users_per_event)


# In[28]:


# настройки графика
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

# столбчатая диаграмма количества уникальных пользователей по типам событий
sns.barplot(x='unique_users', y='event_name', data=users_per_event, palette='rocket')

# название графика и осей
plt.title('Количество уникальных пользователей по типам событий')
plt.xlabel('Количество пользователей')
plt.ylabel('Тип события')

# вывод графика
plt.show()


# ### Промежуточные выводы по преверке данных

#    *    На графике распределения количества событий по типам событий видно, что наибольшее количество событий было связано с просмотром главного экрана (MainScreenAppear) и нажатием кнопки "Купить" на этом экране (OffersScreenAppear). Количество событий, связанных с прохождением обучения (Tutorial), было наименьшим. Это может указывать на то, что многие пользователи уже знакомы с приложением или им не требуется дополнительное обучение
#    *    На графике количества уникальных пользователей видно, что больше всего уникальных пользователей совершали событие MainScreenAppear (показ главного экрана приложения), что может говорить о том, что оно является наиболее важным для пользователей. Далее идут события OffersScreenAppear (показ экрана с предложениями) и CartScreenAppear (показ экрана с корзиной). Событие PaymentScreenSuccessful (успешный платеж) имеет наименьшее количество уникальных пользователей, что может указывать на низкую конверсию в покупки
#    * На гистограмме распределения событий по дате и времени можно заметить, что данные охватывают период с 25 июля по 7 августа 2019 года. В целом, количество событий в этот период постепенно увеличивается, достигает пика в середине периода и затем снижается. Также можно выделить некоторые аномалии в виде небольших пиков и спадов, которые могут быть связаны с некоторыми внешними факторами, например, событиями в реальном мире или техническими проблемами
#    * В логах присутствуют потерянные данные. Всего потеряно 2826 событий и 17 уникальных пользователей. Важно обратить на это внимание при анализе результатов эксперимента
#    * Есть три группы для эксперимента: 246, 247, 248
#     


# ## Воронка событий


# In[29]:


data = data.drop(data[data['event_name'] == 'Tutorial'].index)


# In[30]:


# Создаем DataFrame с данными для воронки
funnel_data = data.loc[data['event_name'] != 'Обучение']     .groupby('event_name', as_index=False)     .agg({'device_id': 'nunique'})     .sort_values(by='device_id', ascending=False)

# Добавляем колонку с долей пользователей
funnel_data['ratio'] = funnel_data['device_id'] / funnel_data['device_id'].max()

# Создаем воронку
fig = go.Figure(
    go.Funnel(
        y=funnel_data['event_name'],
        x=funnel_data['device_id'],
        textinfo="value+percent previous",
        textposition="inside",
        marker=dict(color=funnel_data['device_id'],
                    colorscale='RdBu',
                    reversescale=True
                    )
    )
)

# Добавляем настройки графика
fig.update_layout(
    title='Воронка событий',
    height=700,
    font=dict(size=15)
)

# Показываем график
fig.show()


# In[31]:


# расчёт доли пользователей на каждом этапе воронки
funnel_data['user_ratio'] = funnel_data['device_id'] / funnel_data['device_id'].iloc[0]

print(funnel_data)


# In[32]:


# визуализация доли пользователей на каждом этапе
fig = make_subplots(rows=1, cols=2, 
                    specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Доля пользователей', 'Отношение пользователей']
                   )

fig.add_trace(
    go.Pie(labels=funnel_data['event_name'], values=funnel_data['user_ratio'], 
           name='Доля пользователей', hole=.3),
    1, 1
)

fig.add_trace(
    go.Pie(labels=funnel_data['event_name'][1:], values=funnel_data['user_ratio'].iloc[1:] / 
           funnel_data['user_ratio'].iloc[:-1], 
           name='Отношение пользователей', hole=.3),
    1, 2
)

fig.update_traces(textinfo='percent+label')
fig.update_layout(title_text='Воронка событий', title_x=0.5)
fig.show()


# In[33]:


funnel_users = (data.groupby('event_name')
                 .agg({'device_id': 'nunique'})
                 .sort_values(by='device_id', ascending=False)
               )
funnel_users.columns = ['users_count']
funnel_users


# In[34]:


conversion = (funnel_users.loc['PaymentScreenSuccessful', 'users_count'] / 
              funnel_users.loc['MainScreenAppear', 'users_count'])
print('{:.2%}'.format(conversion))


# ### Промежуточные выводы по воронке событий

# * Из графика воронки событий можно сделать следующие выводы:
#   * Всего было 5 типов событий, начиная от главного экрана до оплаты
#   * Большая часть пользователей, которые перешли на главный экран, переходили дальше к предложению о покупке, что говорит о хорошей конверсии с этого шага
#   * Однако, из тех, кто перешел на страницу предложения о покупке, далеко не все сделали покупку
#   * Менее половины пользователей, перешедших на страницу предложения о покупке, прошли этот шаг до конца и осуществили оплату
#   * Больше всего пользователей теряется на шаге предложения о покупке и при оплате
# * Из расчета доли пользователей можно сделать следующие выводы:
#   * Наибольшее количество уникальных пользователей было на первом этапе воронки (MainScreenAppear) - 7419 человек. 
#   * На последующих этапах количество пользователей постепенно уменьшается 
#   * На втором этапе (OffersScreenAppear) остаётся 4201 человек, на третьем (CartScreenAppear) - 1767 человек, и на последнем этапе (PaymentScreenSuccessful) - 454 человека
#   * Также была добавлена колонка user_ratio, в которой посчитана доля пользователей на каждом этапе относительно начального этапа (MainScreenAppear)
#   * На последнем этапе воронки (PaymentScreenSuccessful) оказалось всего 6.1% пользователей, которые начали взаимодействие с приложением 
#   * Это говорит о том, что значительная часть пользователей теряется на более ранних этапах
# * Из анализа воронки событий мы можем сделать следующие выводы:
#   * Больше всего пользователей приходит на главный экран (38% от общего числа), но только 62% из них переходят на экран предложений
#   * На экран предложений переходят 23% пользователей от общего числа
#   * Далее на экран платежей переходит только 48% от тех пользователей, которые были на экране предложений
#   * На экран успешной оплаты попадают уже 58% от общего числа пользователей
#   * Таким образом, воронка показывает, что больше всего пользователей теряется на этапе перехода с главного экрана на экран предложений, а также на этапе перехода с экрана предложений на экран платежей
#   * Также мы видим, что отношение пользователей на каждом последующем этапе воронки уменьшается
# * Из расчета соотношения количества пользователей:
#   * Получено значение конверсии в 47.70%
#   * Это означает, что менее половины всех пользователей, которые запустили приложение, дошли до этапа успешной оплаты

# ## Изучаем результаты эксперимента

# In[35]:


data.groupby('group')['device_id'].nunique()


# In[36]:


# запишем количество пользователей в таблицу
users_by_group = data.groupby('group')['device_id'].nunique()
users_by_group['A1+A2'] = users_by_group[246] + users_by_group[247]
users_by_group


# In[37]:


user_only = data.groupby('device_id')['group'].nunique()
user_only_sorted = user_only.sort_values()
print("Проверка на то, чтобы каждый пользователь состоял только в одной группе:")
print(user_only_sorted)

# In[38]:


# Код ревьюера
# Проверим пользователей, которые могли участвовать в двух или нескольких группах одновременно:
data.groupby('device_id').agg({'group':'nunique'}).query('group > 1') 


# In[39]:


event_group_test = data[data['event_name']!='Tutorial'].pivot_table(
    index='event_name', 
    columns='group', 
    values='device_id',
    aggfunc='nunique').sort_values(by=246, ascending=False)

event_group_test = event_group_test.reset_index()
event_group_test['A1+A2'] = event_group_test[246] + event_group_test[247]
event_group_test['all'] = event_group_test['A1+A2'] + event_group_test[248]

event_group_test['part_A1'] = (event_group_test[246] / users_by_group[247] * 100).round(1)
event_group_test['part_A2'] = (event_group_test[247] / users_by_group[247] * 100).round(1)
event_group_test['part_B'] = (event_group_test[248] / users_by_group[248] * 100).round(1)
event_group_test['part_A1+A2'] = ((event_group_test[246] + event_group_test[247]) /                                   (users_by_group[246] + users_by_group[247]) * 100).round(1)
event_group_test


# In[40]:


# Бар-графики
event_group_test.plot(x='event_name', y=[246, 247, 248], kind='bar', figsize=(10,6))
plt.xlabel('Название события')
plt.ylabel('Количество уникальных пользователей')
plt.title('Количество уникальных пользователей в каждой группе по событиям')

# График-линия
event_group_test.plot(x='event_name', y=['part_A1', 'part_A2', 'part_B'], kind='line', figsize=(10,6))
plt.xlabel('Название события')
plt.ylabel('Доля пользователей, %')
plt.title('Доля пользователей в каждой группе по событиям')
plt.legend(loc='upper left')
plt.show()


# In[41]:


users_in_groups = data.groupby('group').agg({'device_id': 'nunique'}).reset_index()
users_in_groups.columns = ['group', 'users_count']
print(users_in_groups)


# Гипотеза H0: Доля пользователей, совершивших самое популярное событие, одинакова в обеих контрольных группах.
# 
# Гипотеза H1: Доля пользователей, совершивших самое популярное событие, различна в контрольных группах.

# In[42]:


# Выберем данные для контрольных групп
control_group_1 = data[data['group'] == 246]
control_group_2 = data[data['group'] == 247]

# Посчитаем количество пользователей в каждой группе
n1 = control_group_1['device_id'].nunique()
n2 = control_group_2['device_id'].nunique()

# Выберем самое популярное событие
most_popular_event = data['event_name'].value_counts().index[0]

# Посчитаем количество пользователей, совершивших это событие в каждой группе
n1_event = control_group_1[control_group_1['event_name'] == most_popular_event]['device_id'].nunique()
n2_event = control_group_2[control_group_2['event_name'] == most_popular_event]['device_id'].nunique()

# Посчитаем долю пользователей, совершивших это событие в каждой группе
p1 = n1_event / n1
p2 = n2_event / n2

# Посчитаем статистику t-теста и p-value
p_combined = (n1_event + n2_event) / (n1 + n2)
z_value = (p1 - p2) / np.sqrt(p_combined * (1 - p_combined) * (1 / n1 + 1 / n2))
p_value = stats.norm.sf(abs(z_value)) * 2

# Выведем результаты
print('Количество пользователей в первой контрольной группе:', n1)
print('Количество пользователей во второй контрольной группе:', n2)
print('Количество пользователей, совершивших самое популярное событие в первой контрольной группе:', n1_event)
print('Количество пользователей, совершивших самое популярное событие во второй контрольной группе:', n2_event)
print('Доля пользователей, совершивших самое популярное событие в первой контрольной группе:', p1)
print('Доля пользователей, совершивших самое популярное событие во второй контрольной группе:', p2)
print('p-value:', p_value)

if p_value < 0.05:
    print('Отвергаем нулевую гипотезу: между долями есть значимая разница')
else:
    print('Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными')

# In[43]:


test_group = data[data['group'] == 248]


# In[44]:


n3 = test_group['device_id'].nunique()


# In[45]:


n3_event = test_group[test_group['event_name'] == most_popular_event]['device_id'].nunique()


# In[46]:


event_counts_test = test_group['event_name'].value_counts()


# In[47]:


test_group_unique_users = test_group[test_group['device_id'].isin(control_group_1['device_id']) &
                                      test_group['device_id'].isin(control_group_2['device_id'])]
if len(test_group_unique_users) > 0:
    print('В экспериментальной группе есть уникальные пользователи!')
else:
    print('В экспериментальной группе нет уникальных пользователей!')


# H0: Нет статистически значимого различия доли пользователей, совершивших определенное событие, между экспериментальной группой и объединенной контрольной группой (группы 1 и 3).
# H1: Есть статистически значимое различие доли пользователей, совершивших определенное событие, между экспериментальной группой и объединенной контрольной группой (группы 1 и 3).
# 
# H0: Нет статистически значимого различия доли пользователей, совершивших определенное событие, между экспериментальной группой и второй контрольной группой (группы 2 и 3).
# H1: Есть статистически значимое различие доли пользователей, совершивших определенное событие, между экспериментальной группой и второй контрольной группой (группы 2 и 3).
# 
# H0: Нет статистически значимого различия доли пользователей, совершивших определенное событие, между первой и второй контрольными группами (группы 1 и 2).
# H1: Есть статистически значимое различие доли пользователей, совершивших определенное событие, между первой и второй контрольными группами (группы 1 и 2).

# In[48]:


all_groups = data.pivot_table(index='event_name', columns='group',values='device_id',aggfunc='nunique')                       .sort_values(246,ascending=False)
all_groups['246+247'] = all_groups[246] + all_groups[247]
all_groups


# In[49]:


users = data.groupby('group')['device_id'].nunique()


# In[50]:


users= users.to_frame().reset_index()
users.loc[3] = ['246+247', 4997]


# In[51]:


users = users.set_index(users.columns[0])
users


# In[60]:


num_hypothesis = 16
alpha_bonf = 0.05 / num_hypothesis
print('Значение Бонферрони:',bonferroni)


# In[61]:


def z_test(exp1, exp2, event, alpha_bonf):
    p1_ev = all_groups.loc[event, exp1]
    p2_ev = all_groups.loc[event, exp2] 
    p1_us = users.loc[exp1, 'device_id'] 
    p2_us = users.loc[exp2, 'device_id'] 
    p1 = p1_ev / p1_us 
    p2 = p2_ev / p2_us 
    difference = p1 - p2
    p_combined = (p1_ev + p2_ev) / (p1_us + p2_us) 
    z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1 / p1_us + 1 / p2_us))
    distr = st.norm(0, 1)
    p_value = (1 - distr.cdf(abs(z_value))) * 2

    alpha = alpha_bonf / 2 # уровень значимости для двустороннего t-теста
    if (p_value < alpha):
        print('Проверка для  {} и {}, событие: {}, p-значение: {p_value:.5f} - различие статистически значимо'.format(exp1, exp2, event, p_value=p_value))
    else:
        print('Проверка для  {} и {}, событие: {}, p-значение: {p_value:.5f} - различие статистически не значимо'.format(exp1, exp2, event, p_value=p_value))


# In[62]:


for event in all_groups.index:
    z_test(246, 247, event, 0.05)
    print()


# In[55]:


for event in all_groups.index:
    z_test(246, 248, event, 0.05)
    print()


# In[56]:


for event in all_groups.index:
    z_test(247, 248, event, 0.05)
    print()


# In[57]:


for event in all_groups.index:
    z_test('246+247', 248, event, 0.05)
    print()

# ### Выводы по результатам эксперимента

#    * Из представленных экспериментов можно сделать выводы, что в данном исследовании не было обнаружено статистически значимых различий между контрольными группами
#    * В частности, не было обнаружено различий в доле пользователей, совершивших самое популярное событие 
#    * Это говорит о том, что изменения, которые были внесены в интерфейс приложения, не повлияли на поведение пользователей, а значит, эти изменения можно считать неэффективными 
#    * Однако, следует отметить, что на основании этого исследования нельзя делать окончательные выводы, поскольку оно может иметь ограниченную статистическую мощность, а также не учитывает многие другие факторы, которые могут повлиять на поведение пользователей
#    * Значения p-value высокие для всех трех сравнений, что говорит о том, что нет статистически значимых различий между группами 
#    * Таким образом, мы не можем отвергнуть нулевую гипотезу о том, что различий в поведении пользователей в группах нет

# ## Итоговые выводы

# В ходе данного проекта были проанализированы данные о пользователях мобильного приложения для продажи продуктов питания. В качестве целей были поставлены:
# 
# * Изучить воронку продаж и выявить этапы, на которых теряется больше всего пользователей
# * Исследовать результаты A/A/B-тестирования и определить, есть ли различия между контрольными и экспериментальной группами
#     
# В ходе анализа воронки продаж было установлено, что:
# * Наибольшие потери пользователей происходят на этапе перехода от главного экрана к экрану выбора товара
# * Возможно, это связано с тем, что на главном экране не достаточно информации о том, какие товары доступны в приложении
# * Рекомендуется улучшить информативность главного экрана, чтобы пользователи были более заинтересованы в продолжении работы с приложением
# 
# В ходе анализа результатов A/A/B-тестирования было установлено, что: 
# * Различий между контрольными группами не обнаружено 
# * В экспериментальной группе доля пользователей, дошедших до экрана оформления заказа, оказалась на 10% выше, чем в среднем по контрольным группам
# * При этом статистически значимых различий между экспериментальной и контрольными группами не обнаружено, что может быть связано с недостаточным размером выборки
# * Рекомендуется провести дополнительное тестирование с увеличенным размером выборки, чтобы подтвердить или опровергнуть результаты текущего тестирования




