#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid green 2px; padding: 20px">
#     
# <b>Денис, привет!</b> Мы рады тебя видеть на территории код-ревьюеров. Ты проделал большую работу над проектом, но давай познакомимся и сделаем его еще лучше! У нас тут своя атмосфера и несколько правил:
# 
# 
# 1. Меня зовут Александр Матвеевский. Я работаю код-ревьюером, моя основная цель — не указать на совершенные тобою ошибки, а поделиться своим опытом и помочь тебе стать аналитиком данных.
# 2. Общаемся на ты.
# 3. Если хочешь написать, спросить - не нужно стесняться. Только выбери свой цвет для комментария.  
# 4. Это учебный проект, тут можно не бояться сделать ошибку.  
# 5. У тебя неограниченное количество попыток для сдачи проекта.  
# 6. Let's Go!
# 
# ---
# 
# Я буду красить комментарии цветом, пожалуйста, не удаляй их:
# 
# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
# 
# Такой комментарий нужно исправить обязательно, он критически влияет на удачное выполнение проекта.
# </div>
#     
# ---
# 
# <div class="alert alert-block alert-warning">📝
#     
# 
# __Комментарий от ревьюера №1__
# 
# 
# Такой комментарий является рекомендацией или советом. Можешь использовать их на своё усмотрение.
# </div>
# 
# ---
# 
# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
# Такой комментарий  говорит о том, что было сделано что-то качественное и правильное =)
# </div>
#     
# ---
#     
# Предлагаю работать над проектом в диалоге: если ты что-то меняешь в проекте или отвечаешь на мои комментарии — пиши об этом. Мне будет легче отследить изменения, если ты выделишь свои комментарии:   
#     
# <div class="alert alert-info"> <b>Комментарии студента:</b> Например, вот так.</div>
#     
# Всё это поможет выполнить повторную проверку твоего проекта оперативнее. Если будут какие-нибудь вопросы по моим комментариям, пиши, будем разбираться вместе :)    
#     
# ---

# 

# 

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
# 
# 
# Отличная практика - расписывать цель и основные этапы своими словами (этот навык очень поможет на фильнальном проекте). Хорошо было бы добавить ход и цель исследования. Вот мой личный пример из 3 проекта: 
#     
# ![image.png](attachment:image.png)
# </div>

# 

# # Бизнес-показатели развлекательного приложения Procrastinate Pro+

# Заказчик
# 
# Приложение Procrastinate Pro+
# 
# Цель проекта
# 
# На основе данных, предоставленных компанией, необходимо провести анализ и ответить на вопросы:
# 
# откуда приходят пользователи и какими устройствами они пользуются,
# сколько стоит привлечение пользователей из различных рекламных каналов;
# сколько денег приносит каждый клиент,
# когда расходы на привлечение клиента окупаются,
# какие факторы мешают привлечению клиентов.ь

# Ход исследования
# 
# Исследование пройдёт в четыре этапа:
# 
# Обзор и предобработка данных;
# Исследовательский анализ данных;
# Анализ маркетинговых расходов;
# Оценка окупаемости рекламы.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №2__
# 
# Супер, старайся, пожалуйста, придерживаться этой стратегии в дальнейших проектах =)
# </div>

# ### Загрузите данные и подготовьте их к анализу

# Загрузите данные о визитах, заказах и рекламных расходах из CSV-файлов в переменные.
# 
# **Пути к файлам**
# 
# - визиты: `/datasets/visits_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/visits_info_short.csv);
# - заказы: `/datasets/orders_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/orders_info_short.csv);
# - расходы: `/datasets/costs_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/costs_info_short.csv).
# 
# Изучите данные и выполните предобработку. Есть ли в данных пропуски и дубликаты? Убедитесь, что типы данных во всех колонках соответствуют сохранённым в них значениям. Обратите внимание на столбцы с датой и временем.

# импорт

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# считываем файлы

# In[2]:


visits = pd.read_csv('/datasets/visits_info_short.csv')
orders = pd.read_csv('/datasets/orders_info_short.csv')
costs = pd.read_csv('/datasets/costs_info_short.csv')


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
#         
# Для подгрузки данных можно использовать конструкцию `try-except`, она поможет избежать потенциальных ошибок при загрузке данных, связанных, например, с некорректным указанием путей.
#         
# Подробнее о конструкции по ссылке:
#         
# https://pythonworld.ru/tipy-dannyx-v-python/isklyucheniya-v-python-konstrukciya-try-except-dlya-obrabotki-isklyuchenij.html
#         
# Либо же можно использовать стандартную библиотеку os:
#         
# https://pythonworld.ru/moduli/modul-os.html
# 
#     
# Несколько интересных статей кейсы использования конструкции:
#     
# https://www.programiz.com/python-programming/exception-handling
#     
# https://towardsdatascience.com/do-not-abuse-try-except-in-python-d9b8ee59e23b
#     
# https://www.techbeamers.com/use-try-except-python/
#         
# Как вариант в try можно указать корректные пути (в нашем случае глобальные) в except - некорректные (локальные). Можно также специфицровать тип ошибки, FileNotFoundError или задать кастомный тип ошибки (FilePathError, например)
#         
# Она полезна, если ты работаешь локально, а потом подгружаешь проект на платформу. Конструкция позволит не падать коду и локально, и на сервере ЯП, так как если не сработает один блок с путями, сработает другой.
#         
# Ну и вообще, в целом полезно про эту констуркцию знать, она универсальна и может быть использована в разных задачах.

# Смотрим информацию по каждому датасету

# In[3]:


visits.info()


# In[4]:


orders.info()


# In[5]:


costs.info()


# Выводим первые несколько строк датафреймовв

# In[6]:


visits.head(5)


# In[7]:


orders.head(5)


# In[8]:


costs.head(5)


# In[9]:


visits.isna().sum()


# In[10]:


orders.isna().sum()


# In[11]:


costs.isna().sum()


# Проверка дубликатов

# In[12]:


visits.duplicated().sum()


# In[13]:


orders.duplicated().sum()


# In[14]:


costs.duplicated().sum()


# In[15]:


visits.columns = [x.lower().replace(' ', '_') for x in visits.columns.values]


# In[16]:


orders.columns = [x.lower().replace(' ', '_') for x in orders.columns.values]


# In[17]:


costs.columns = [x.lower().replace(' ', '_') for x in costs.columns.values]


# In[18]:


visits.info()


# In[19]:


orders.info()


# In[20]:


costs.info()


# In[21]:


visits['session_end'] = pd.to_datetime(visits['session_end'], format = '%Y-%m-%d %H:%M:%S')
visits['session_start'] = pd.to_datetime(visits['session_start'], format = '%Y-%m-%d %H:%M:%S')


# In[22]:


orders['event_dt'] = pd.to_datetime(orders['event_dt'], format = '%Y-%m-%d %H:%M:%S')


# In[23]:


costs['dt'] = pd.to_datetime(costs['dt'], format="%Y-%m-%d")


# In[24]:


visits.describe()


# In[25]:


orders.describe()


# In[26]:


costs.describe()


# Ознакомился с таблицами. Пропусков и дубликатов нет. Были проблемы с типом данных, поменял на нужный формат

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
# 
# Отсутствует проверка на дубликаты и пропуски. Так же нету вывода о качестве исходных данных. Поправишь, пожалуйста?
# </div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №2__
# 
#     
# Хорошо, основные действия по предобработке сделаны
# </div>

# <div class="alert alert-block alert-warning">📝
#     
# 
# __Комментарий от ревьюера №1__
# 
# 
#     
# * на стадии загрузки и подготовки данных к исследовательскому анализу, советую посмотреть на данные более детально, чтобы избежать невынужденных ошибок в дальнейшем. Чем большем мы знаем о данных - тем более корректны и обоснованы выглядят наши выводы. Такие проверки много времени не занимают, но зато мы можем лучше контролировать данные и их анализ. Например, мы можем:
#     - проверить временной интервал на соответствие условию проекта, а также на возможные ошибки (например, проверить случаи, когда окончание сессии было раньше, и так далее);
#     - для численных данных посмотреть на их статистические показатели и проверить их на наличие каких-то ошибок или аномалий (например, нули или отрицательные значения там, где они не должны быть).

# ### Задайте функции для расчёта и анализа LTV, ROI, удержания и конверсии.
# 
# Разрешается использовать функции, с которыми вы познакомились в теоретических уроках.
# 
# Это функции для вычисления значений метрик:
# 
# - `get_profiles()` — для создания профилей пользователей,
# - `get_retention()` — для подсчёта Retention Rate,
# - `get_conversion()` — для подсчёта конверсии,
# - `get_ltv()` — для подсчёта LTV.
# 
# А также функции для построения графиков:
# 
# - `filter_data()` — для сглаживания данных,
# - `plot_retention()` — для построения графика Retention Rate,
# - `plot_conversion()` — для построения графика конверсии,
# - `plot_ltv_roi` — для визуализации LTV и ROI.

# In[27]:


def get_profiles(sessions, orders, ad_costs):

    # сортируем сессии по ID пользователя и дате привлечения
    # группируем по ID и находим параметры первых посещений
    profiles = (
        sessions.sort_values(by=['user_id', 'session_start'])
        .groupby('user_id')
        .agg(
            {
                'session_start': 'first',
                'channel': 'first',
                'device': 'first',
                'region': 'first',
            }
        )
         # время первого посещения назовём first_ts
        .rename(columns={'session_start': 'first_ts'})
        .reset_index()  # возвращаем user_id из индекса
    )

    # для когортного анализа определяем дату первого посещения
    # и первый день месяца, в который это посещение произошло
    profiles['dt'] = profiles['first_ts'].dt.date
    profiles['dt'] = pd.to_datetime(profiles['dt'], format="%Y-%m-%d")
    profiles['month'] = profiles['first_ts'].astype('datetime64[M]')

    # добавляем признак платящих пользователей
    profiles['payer'] = profiles['user_id'].isin(orders['user_id'].unique())

    # считаем количество уникальных пользователей
    # с одинаковыми источником и датой привлечения
    new_users = (
        profiles.groupby(['dt', 'channel'])
        .agg({'user_id': 'nunique'})
         # столбец с числом пользователей назовём unique_users
        .rename(columns={'user_id': 'unique_users'})
        .reset_index()  # возвращаем dt и channel из индексов
    )

    # объединяем траты на рекламу и число привлечённых пользователей
    # по дате и каналу привлечения
    ad_costs = ad_costs.merge(new_users, on=['dt', 'channel'], how='left')

    # делим рекламные расходы на число привлечённых пользователей
    # результаты сохраним в столбец acquisition_cost (CAC)
    ad_costs['acquisition_cost'] = ad_costs['costs'] / ad_costs['unique_users']

    # добавим стоимость привлечения в профили
    profiles = profiles.merge(
        ad_costs[['dt', 'channel', 'acquisition_cost']],
        on=['dt', 'channel'],
        how='left',
    )

    # органические пользователи не связаны с данными о рекламе,
    # поэтому в столбце acquisition_cost у них значения NaN
    # заменим их на ноль, ведь стоимость привлечения равна нулю
    profiles['acquisition_cost'] = profiles['acquisition_cost'].fillna(0)

    return profiles  # возвращаем профили с CAC


# In[28]:


def get_retention(
    profiles,
    sessions,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # добавляем столбец payer в передаваемый dimensions список
    dimensions = ['payer'] + dimensions

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # собираем «сырые» данные для расчёта удержания
    result_raw = result_raw.merge(
        sessions[['user_id', 'session_start']], on='user_id', how='left'
    )
    result_raw['lifetime'] = (
        result_raw['session_start'] - result_raw['first_ts']
    ).dt.days

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу удержания
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    # получаем таблицу динамики удержания
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    # возвращаем обе таблицы и сырые данные
    return result_raw, result_grouped, result_in_time


# In[29]:


def get_conversion(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # определяем дату и время первой покупки для каждого пользователя
    first_purchases = (
        purchases.sort_values(by=['user_id', 'event_dt'])
        .groupby('user_id')
        .agg({'event_dt': 'first'})
        .reset_index()
    )

    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        first_purchases[['user_id', 'event_dt']], on='user_id', how='left'
    )

    # рассчитываем лайфтайм для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days

    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']


# In[30]:


# функция для сглаживания фрейма

def filter_data(df, window):
    # для каждого столбца применяем скользящее среднее
    for column in df.columns.values:
        df[column] = df[column].rolling(window).mean() 
    return df


# In[31]:


def plot_retention(retention, retention_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 10))

    # исключаем размеры когорт и удержание первого дня
    retention = retention.drop(columns=['cohort_size', 0])
    # в таблице динамики оставляем только нужный лайфтайм
    retention_history = retention_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # если в индексах таблицы удержания только payer,
    # добавляем второй признак — cohort
    if retention.index.nlevels == 1:
        retention['cohort'] = 'All users'
        retention = retention.reset_index().set_index(['cohort', 'payer'])

    # в таблице графиков — два столбца и две строки, четыре ячейки
    # в первой строим кривые удержания платящих пользователей
    ax1 = plt.subplot(2, 2, 1)
    retention.query('payer == True').droplevel('payer').T.plot(
        grid=True, ax=ax1
    )
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание платящих пользователей')

    # во второй ячейке строим кривые удержания неплатящих
    # вертикальная ось — от графика из первой ячейки
    ax2 = plt.subplot(2, 2, 2, sharey=ax1)
    retention.query('payer == False').droplevel('payer').T.plot(
        grid=True, ax=ax2
    )
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание неплатящих пользователей')

    # в третьей ячейке — динамика удержания платящих
    ax3 = plt.subplot(2, 2, 3)
    # получаем названия столбцов для сводной таблицы
    columns = [
        name
        for name in retention_history.index.names
        if name not in ['dt', 'payer']
    ]
    # фильтруем данные и строим график
    filtered_data = retention_history.query('payer == True').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания платящих пользователей на {}-й день'.format(
            horizon
        )
    )

    # в чётвертой ячейке — динамика удержания неплатящих
    ax4 = plt.subplot(2, 2, 4, sharey=ax3)
    # фильтруем данные и строим график
    filtered_data = retention_history.query('payer == False').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax4)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания неплатящих пользователей на {}-й день'.format(
            horizon
        )
    )
    
    plt.tight_layout()
    plt.show()


# In[32]:


def get_conversion(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # определяем дату и время первой покупки для каждого пользователя
    first_purchases = (
        purchases.sort_values(by=['user_id', 'event_dt'])
        .groupby('user_id')
        .agg({'event_dt': 'first'})
        .reset_index()
    )

    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        first_purchases[['user_id', 'event_dt']], on='user_id', how='left'
    )

    # рассчитываем лайфтайм для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days

    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users' 
        dimensions = dimensions + ['cohort']

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        result = result.fillna(0).cumsum(axis = 1)
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # делим каждую «ячейку» в строке на размер когорты
        # и получаем conversion rate
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу конверсии
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    # для таблицы динамики конверсии убираем 'cohort' из dimensions
    if 'cohort' in dimensions: 
        dimensions = []

    # получаем таблицу динамики конверсии
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    # возвращаем обе таблицы и сырые данные
    return result_raw, result_grouped, result_in_time


# In[33]:


def plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon, window=7):

    # задаём сетку отрисовки графиков
    plt.figure(figsize=(20, 20))

    # из таблицы ltv исключаем размеры когорт
    ltv = ltv.drop(columns=['cohort_size'])
    # в таблице динамики ltv оставляем только нужный лайфтайм
    ltv_history = ltv_history.drop(columns=['cohort_size'])[[horizon - 1]]

    # стоимость привлечения запишем в отдельный фрейм
    cac_history = roi_history[['cac']]

    # из таблицы roi исключаем размеры когорт и cac
    roi = roi.drop(columns=['cohort_size', 'cac'])
    # в таблице динамики roi оставляем только нужный лайфтайм
    roi_history = roi_history.drop(columns=['cohort_size', 'cac'])[
        [horizon - 1]
    ]

    # первый график — кривые ltv
    ax1 = plt.subplot(3, 2, 1)
    ltv.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('LTV')

    # второй график — динамика ltv
    ax2 = plt.subplot(3, 2, 2, sharey=ax1)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in ltv_history.index.names if name not in ['dt']]
    filtered_data = ltv_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика LTV пользователей на {}-й день'.format(horizon))

    # третий график — динамика cac
    ax3 = plt.subplot(3, 2, 3, sharey=ax1)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in cac_history.index.names if name not in ['dt']]
    filtered_data = cac_history.pivot_table(
        index='dt', columns=columns, values='cac', aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика стоимости привлечения пользователей')

    # четвёртый график — кривые roi
    ax4 = plt.subplot(3, 2, 4)
    roi.T.plot(grid=True, ax=ax4)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('ROI')

    # пятый график — динамика roi
    ax5 = plt.subplot(3, 2, 5, sharey=ax4)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in roi_history.index.names if name not in ['dt']]
    filtered_data = roi_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax5)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.xlabel('Дата привлечения')
    plt.title('Динамика ROI пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show()


# In[34]:


def plot_conversion(conversion, conversion_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 5))

    # исключаем размеры когорт
    conversion = conversion.drop(columns=['cohort_size'])
    # в таблице динамики оставляем только нужный лайфтайм
    conversion_history = conversion_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # первый график — кривые конверсии
    ax1 = plt.subplot(1, 2, 1)
    conversion.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Конверсия пользователей')

    # второй график — динамика конверсии
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    columns = [		
        # столбцами сводной таблицы станут все столбцы индекса, кроме даты
        name for name in conversion_history.index.names if name not in ['dt']
    ]
    filtered_data = conversion_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика конверсии пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show()


# In[35]:


def get_ltv(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')
  
    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        purchases[['user_id', 'event_dt', 'revenue']], on='user_id', how='left'
    )
    
    # рассчитываем лайфтайм пользователя для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days
    
    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']

    # функция группировки по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        # строим «треугольную» таблицу выручки
        result = df.pivot_table(
            index=dims, columns='lifetime', values='revenue', aggfunc='sum'
        )
        # находим сумму выручки с накоплением
        result = result.fillna(0).cumsum(axis=1)
        # вычисляем размеры когорт
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        # объединяем размеры когорт и таблицу выручки
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # считаем LTV: делим каждую «ячейку» в строке на размер когорты
        result = result.div(result['cohort_size'], axis=0)
        # исключаем все лайфтаймы, превышающие горизонт анализа
        result = result[['cohort_size'] + list(range(horizon_days))]
        # восстанавливаем размеры когорт
        result['cohort_size'] = cohort_sizes

        # собираем датафрейм с данными пользователей и значениями CAC, 
        # добавляя параметры из dimensions
        cac = df[['user_id', 'acquisition_cost'] + dims].drop_duplicates()

        # считаем средний CAC по параметрам из dimensions
        cac = (
            cac.groupby(dims)
            .agg({'acquisition_cost': 'mean'})
            .rename(columns={'acquisition_cost': 'cac'})
        )

        # считаем ROI: делим LTV на CAC
        roi = result.div(cac['cac'], axis=0)

        # удаляем строки с бесконечным ROI
        roi = roi[~roi['cohort_size'].isin([np.inf])]

        # восстанавливаем размеры когорт в таблице ROI
        roi['cohort_size'] = cohort_sizes

        # добавляем CAC в таблицу ROI
        roi['cac'] = cac['cac']

        # в финальной таблице оставляем размеры когорт, CAC
        # и ROI в лайфтаймы, не превышающие горизонт анализа
        roi = roi[['cohort_size', 'cac'] + list(range(horizon_days))]

        # возвращаем таблицы LTV и ROI
        return result, roi

    # получаем таблицы LTV и ROI
    result_grouped, roi_grouped = group_by_dimensions(
        result_raw, dimensions, horizon_days
    )

    # для таблиц динамики убираем 'cohort' из dimensions
    if 'cohort' in dimensions:
        dimensions = []

    # получаем таблицы динамики LTV и ROI
    result_in_time, roi_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    return (
        result_raw,  # сырые данные
        result_grouped,  # таблица LTV
        result_in_time,  # таблица динамики LTV
        roi_grouped,  # таблица ROI
        roi_in_time,  # таблица динамики ROI
    )


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
#     
# Хорошо, все необходимые функции были заданы, можно приступать к расчета и анализу👀

# ### Исследовательский анализ данных
# 
# - Составьте профили пользователей. Определите минимальную и максимальную даты привлечения пользователей.
# - Выясните, из каких стран пользователи приходят в приложение и на какую страну приходится больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих из каждой страны.
# - Узнайте, какими устройствами пользуются клиенты и какие устройства предпочитают платящие пользователи. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого устройства.
# - Изучите рекламные источники привлечения и определите каналы, из которых пришло больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого канала привлечения.
# 
# После каждого пункта сформулируйте выводы.

# In[36]:


profiles = get_profiles(visits, orders, costs)
display(profiles.head(5)) 


# In[37]:


min_analysis_date = profiles['dt'].min()
observation_date = profiles['dt'].max()


# In[38]:


print(min_analysis_date, "\n", observation_date)


# <div class="alert alert-block alert-warning">📝
#     
# 
# __Комментарий от ревьюера №1__
# 
# Отсутствует вывод, иначе не совсем понятно, зачем мы смотрели на эти даты (например, соответствую ли даты ТЗ или нет).
# </div>

# In[39]:


region =  (profiles
           .pivot_table(
                        index='region',
                        columns='payer',
                        values='user_id',
                        aggfunc='count')
           .rename(columns={True: 'payer', False: 'not_payer'})
           .sort_values(by='payer', ascending=False)
          )


# In[40]:


fig, ax = plt.subplots(figsize=(18, 3))
region[['payer']].plot(kind='barh', stacked=True, ax=ax, alpha=0.4, color = 'maroon')
region[['not_payer']].plot(kind='barh', stacked=True, ax=ax, alpha=0.4, color = 'red')
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel('Количество ипользователей')
ax.set_ylabel(' ')
ax.set_title('Привлеченные пользователи в разрезе по странам',loc='left')
plt.show()

region['payer_share'] = (region.payer / (region.not_payer + region.payer) * 100).round(2)
region


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
# Здорово, что визуализируешь полученные данные. Это важный скилл для аналитиков, развивай его
# </div>

# In[41]:


device =  (profiles
           .pivot_table(
                        index='device',
                        columns='payer',
                        values='user_id',
                        aggfunc='count')
           .rename(columns={True: 'payer', False: 'not_payer'})
           .sort_values(by='payer', ascending=False)
          )


# In[42]:


fig, ax = plt.subplots(figsize=(18, 3))
device[['payer']].plot(kind='barh', stacked=True, ax=ax, alpha=0.4, color = 'maroon')
device[['not_payer']].plot(kind='barh', stacked=True, ax=ax, alpha=0.4, color = 'red')
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel('Количество пользователей')
ax.set_ylabel(' ')
ax.set_title('Привлеченные пользователи в разрезе устройств',loc='left')
plt.show()

device['payer_share'] = (device.payer / (device.not_payer + device.payer) * 100).round(2)
device


# In[43]:


channel =  (profiles
           .pivot_table(
                        index='channel',
                        columns='payer',
                        values='user_id',
                        aggfunc='count')
           .rename(columns={True: 'payer', False: 'not_payer'})
           .sort_values(by='payer', ascending=False)
          )


# In[44]:


fig, ax = plt.subplots(figsize=(18, 5))
channel[['payer']].plot(kind='barh', stacked=True, ax=ax, alpha=0.4, color = 'maroon')
channel[['not_payer']].plot(kind='barh', stacked=True, ax=ax, alpha=0.4, color = 'red')
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel('Количество пользователей')
ax.set_ylabel(' ')
ax.set_title('Привлеченные пользователи в разрезе каналов',loc='left')
plt.show()

channel['payer_share'] = (channel.payer / (channel.not_payer + channel.payer) * 100).round(2)
channel


# ### Промежуточные выводы исследовательского анализа

#     Минимальная дата привлечения пользователей: 2019-05-01
#     Максимальная дата привлечения пользователей: 2019-10-27
#     Даты соответствуют условиям задания.
#     В разбивке по странам видно в основном рост стоимости привлечения пользователей приходится на США, они же не показывают окупаемости на 14 день и на них же приходится отрицательный рост в динамике окупаемости. По остальным странам столь существенных изменений не наблюдается, все они показывают окупаемость
#    Самые большие затраты -  на рекламу в TipTop и FaceBoom, причем расходы на рекламу в TipTop многократно увеличились за анализируемый период, тогда как на остальные каналы траты почти не менялись. Также у TipTop выходит самая высокая цена за пользователя, почти в три раза больше чем у того же FaceBoom.
#    Клиенты (как платящие, так и не платящие) предпочитают iOS, они же не показывают окупаемости на 14 день, тогда как для привлечения владельцев компьютеров САС наименьший и наиболее высокий ROI, но в динамике видно что пользователи всех устройств в окупаемости показывают спад.
# 
# 
# 
# 

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
# 
# В целом в данном разделе был проведен хороший анализ данных - мы посмотрели на базовые значения конверсии пользователей в покупатели по регионам, устройствам и каналу привлечения, определили основной рынок. Получается, что больше всего приходит пользователей из США и они лучше других конвертируется. При этом большая часть пользователей заходит с мобильных устройств, это тоже стоит отметить. Также, мы можем сказать, что наибольшую конверсию имеют пользователи, которые пользуются Mac, а затем следуют пользователи iPhone, т.е. мы можем сказать, что в целом пользователи Apple имеют лучшую конверсию в покупателей. Возможно, тут есть плюсы ApplePay.

# ### Маркетинг 
# 
# 
# - Посчитайте общую сумму расходов на маркетинг.
# - Выясните, как траты распределены по рекламным источникам, то есть сколько денег потратили на каждый источник.
# - Постройте визуализацию динамики изменения расходов во времени (по неделям и месяцам) по каждому источнику. Постарайтесь отразить это на одном графике.
# - Узнайте, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника. Используйте профили пользователей.
# 
# Напишите промежуточные выводы.

# In[45]:


analysis_horizon = 14


# In[46]:


print(f'Общая сумма затрат на рекламу: {costs.costs.sum().round()}')


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
#         
# Здорово, что округлили

# In[47]:


costs.pivot_table(index='channel',values='costs',aggfunc='sum').sort_values(by='costs', ascending=False)


# In[48]:


fig, ax = plt.subplots(figsize=(16, 6))

(costs
 .pivot_table(
    index=costs.dt,
    values='costs',
    aggfunc='sum',
    columns='channel'
)
 .plot(ax=ax, stacked=True)
)

ax.set_title('Затраты на привлечение пользователей в разрезе каналов', loc='left')
ax.set_ylabel('затраты на привлечение в день')
ax.set_xlabel(' ')
plt.show()


# In[49]:


# построим график ежемесячных затрат по каналам

fig, ax = plt.subplots(figsize=(16, 6))

(costs
 .pivot_table(
    index=costs.dt.astype('datetime64[M]'),
    values='costs',
    aggfunc='sum',
    columns='channel'
)
 .plot(ax=ax, stacked=True)
)

ax.set_title('Затраты на привлечение пользователей в разрезе каналов', loc='left')
ax.set_ylabel('затраты на привлечение в месяц')
ax.set_xlabel(' ')
plt.show()


# In[50]:


# построим график еженедельных затрат по каналам

fig, ax = plt.subplots(figsize=(16, 6))

(costs
 .pivot_table(
    index=costs.dt.astype('datetime64[W]'),
    values='costs',
    aggfunc='sum',
    columns='channel'
)
 .plot(ax=ax, stacked=True)
)

ax.set_title('Затраты на привлечение пользователей в разрезе каналов', loc='left')
ax.set_ylabel('затраты на привлечение в неделю')
ax.set_xlabel(' ')
plt.show()


# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
# 
# Нет решения по одному из пунктов задания (из брифа) - визуализировать динамику расходов по каналам.
#     
# По  месяцам (с помощью метода `df[''].astype('datetime64[M]')` привести к месячной дате.). Добавишь, пожалуйста?
#     
# ---
#     
# А с помощью `[W]` - недельной
# </div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №2__
# 
# Отличные и наглядные графики 👍
#     
# Здорово, когда они подписаны. Так быстрее понять о чем идёт речь на нём.
#     
# </div>

# In[ ]:





# In[51]:


cac_person = (profiles
              .query('channel != "organic"')[['user_id', 'acquisition_cost']]
              .drop_duplicates()
              .agg({'acquisition_cost': 'mean'})
             )
cac_person


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
# Отличный инсайд для заказчика) Молодец =)
# </div>

# In[52]:


cac_channel = (profiles
       .pivot_table(index = 'channel',
                    values = 'acquisition_cost',
                    aggfunc='mean')
       .sort_values(by='acquisition_cost', ascending=False)
       .rename(columns={'acquisition_cost': 'cac'})
      )
cac_channel


# ### Промежуточные выводы в маркетинге
# 

# Общая сумма затрат на рекламу: 105497.0
# Дороже всего приложению обходится пользователь, которого привлекли просредством TipTop.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
# Наиболее дорогостоящим каналом привлечения пользователей является TipTop, однако количество и доля платящих пользователей, приходящих с этого источника, не так высоки. Это можно объяснить молодой аудиторией TipTop'a и, соответственно, не очень высокой их платежеспособностью.
#     
# </div>

# ### Оцените окупаемость рекламы
# 
# Используя графики LTV, ROI и CAC, проанализируйте окупаемость рекламы. Считайте, что на календаре 1 ноября 2019 года, а в бизнес-плане заложено, что пользователи должны окупаться не позднее чем через две недели после привлечения. Необходимость включения в анализ органических пользователей определите самостоятельно.
# 
# - Проанализируйте окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проверьте конверсию пользователей и динамику её изменения. То же самое сделайте с удержанием пользователей. Постройте и изучите графики конверсии и удержания.
# - Проанализируйте окупаемость рекламы с разбивкой по устройствам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по странам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Ответьте на такие вопросы:
#     - Окупается ли реклама, направленная на привлечение пользователей в целом?
#     - Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
#     - Чем могут быть вызваны проблемы окупаемости?
# 
# Напишите вывод, опишите возможные причины обнаруженных проблем и промежуточные рекомендации для рекламного отдела.

# In[53]:


profiles = profiles.query('channel != "organic"')


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №1__
# 
# Совершенно верно, поскольку мы за них ничего не платим,  а нам нужно изучить именно окупаемость рекламы. 
# </div>

# In[54]:


analysis_horizon = 14


# In[55]:


ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon)

# строим графики
plot_ltv_roi(ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon)


# Реклама к 14-му дню и далее не окупается;
# САС растет, значит рекламный бюджет увеличивается;
# Начиная с июня что-то идет не так: привлеченные клиенты перестают окупаться в двух-недельном лайфтайме

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
# 
# Лучше после каждого раздела, графика (или серии тестов) писать вывод по полученным данным с учетом поставленной бизнес задачи - так проще читать проект, поскольку будущим коллегам или заказчику не надо будет самим интерпретировать результаты каждого раздела, теста или графика.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №2__
#     
# Логика анализа верная, согласен с выводом. Наблюдаем, что динамика ROI за лайфтайм падает. При относительно стабильной динамике LTV, динамика САС растёт с мая по конец октября. Эту закономерность мы наблюдаем в динамике ROI, что при сильном увеличении САС, в равной степени падает динамика ROI пользователей.
# </div>
# 

# Каналы

# In[56]:


dimensions = ['channel']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon, window=14
)


# Затраты на рекламу в TipTop и FaceBoom вообще не окупаются
# 
# С затратами на рекламу TipTop увеличиваются каждый месяц

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №2__
# 
# Выводы?
# </div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №3__
#     
# Да, действительно, есть проблемы с каналом TipTop, видим значительный рост затрат на привлечение.
# </div>

# In[57]:


(
    profiles
    .pivot_table(index='channel',
                 columns='region',
                 aggfunc={'user_id': 'count'}
                )
    .sort_values(by=('user_id', 'United States'), 
                 ascending=False)
).div(
    profiles
    .pivot_table(columns='region',
                 aggfunc={'user_id': 'count'}
                )
    .values
).fillna(0).style.format('{:.2%}')


# In[58]:


dimensions = ['channel']

conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, analysis_horizon)


# Выше всего конверсия у пользователей, привлеченных посредством FaceBoom. Но вообще, динамики у всех пользователей в течение двухнедельного лайфтайма практически и нет.

# Страны

# In[59]:


dimensions = ['region']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon, window=14
)


# In[60]:


dimensions = ['region']

retention_raw, retention_grouped, retention_history = get_retention(
    profiles, visits, observation_date, analysis_horizon, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, analysis_horizon)


# несмотря на огромное количество привлеченных пользователей из США, толку от них для приложения нет, они быстро приходят и так же быстро уходят,
# 
# затраты на рекламу в Европе остаются неизменными и окупаются, а вот в США растут и перестали окупаться с июня (видимо, благодаря вливаниям рекламных денег в TipTop).

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №2__
# 
# Все верно. Нужно разбираться детальнее с рекламой в США, тем более, что это наш основной рынок. 

# In[61]:


dimensions = ['region']

conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, analysis_horizon)


# Конверсия пользователей США почти в два раза выше конверсии пользователей других стран.

# In[62]:


dimensions = ['device']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon, window=14
)


# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
# 
# Отсутствует окупаемость по девайсам. Добавь, пожалуйста
# </div>

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №2__
# 
# Ниже у тебя есть расчёты конверсиии удержания по девайсам, а окупаемость отсутствует. Добавь, пожалуйста
# </div>

# <div class="alert alert-block alert-warning">📝
#     
# 
# __Комментарий от ревьюера №3__
# 
# Неплохо бы отметить, что с окупаемостью проблемы по всем устройствам кроме PC
# </div>

# Девайсы

# In[63]:


dimensions = ['device']

conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, analysis_horizon)


# Все девайсы хорошо конверсируются, в лидерах устройства производста компании Apple, из общего потока немного выбиваются привлеченные пользователи, использующие PC: они привлекаются не так удачно

# In[65]:


dimensions = ['device']

retention_raw, retention_grouped, retention_history = get_retention(
    profiles, visits, observation_date, analysis_horizon, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, analysis_horizon)


# Среди всех устройств окупаемости достигает только ПК. Пользователи макбуков и айфонов удерживаются хуже всех, зато конвертируются лучше всех

# В разбивке по странам видно в основном рост стоимости привлечения пользователей приходится на США, они же не показывают окупаемости на 14 день и на них же приходится отрицательный рост в динамике окупаемости. По остальным странам столь существенных изменений не наблюдается, все они показывают окупаемость.
# Все устройства хорошо конверсируются, в лидерах устройства производста компании Apple, из общего потока немного выбиваются привлеченные пользователи, использующие PC: они привлекаются не так удачно.
# несмотря на огромное количество привлеченных пользователей из США, толку от них для приложения нет, они быстро приходят и так же быстро уходят,
# затраты на рекламу в Европе остаются неизменными и окупаются, а вот в США растут и перестали окупаться с июня (видимо, благодаря вливаниям рекламных денег в TipTop).
# А вот с удержанием платящих пользователей США хуже всех.
# Неплатящие пользователи не показываю каких-то отличий (ни тип устройства, ни регион на их удержание не влияют
# Выше всего конверсия у пользователей, привлеченных посредством FaceBoom. Но вообще, динамики у всех пользователей в течение двухнедельного лайфтайма практически и нет.

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
# 
# Анализ не полный, т.к. проблемы могут быть в удержании каналов/стран/устройств/, а без анализа конверсии и удержания, мы не сможем дать рекомендации маркетологам.
# </div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №3__
# 
# Хорошо, анализ окупаемости корректен, согласен с интерпретацией результатов. Проблемные источники определены верно.

# ### Напишите выводы
# 
# - Выделите причины неэффективности привлечения пользователей.
# - Сформулируйте рекомендации для отдела маркетинга.

# 

# В результате исследования было выявлено, что основной причиной финансовых проблем являются рекламные траты на привлечение пользователей посредством FaceBoom, TipTop в США и AdNonSence в Европе (топ3 САС):
# 
# высокая стоимость привлечения у этих каналов, с горизонтом событий в две недели она не окупается;
# 
# при этом, платящие пользователи FaceBoom и AdNonSence очень плохо удерживаются;
# 
# расходы на привлечение в TipTop за полгода выросли почти в три раза.

# Канал TipTop является проблемным. Он привлекает достаточное количество пользователей (10%), но у него самые высокие расходы на рекламу. Инвестиции в него не окупаются. Возможно, лучше будет использовать рекламный бюджет для продвижения в каналах WahooNetBanner и RocketSuperAds. Они привлекают примерно столько же платных пользователей при меньших расходах на рекламу.

# В качестве следуюших шагов необходимо:
#  отказаться от FaceBoom и перенаправить рекламные средства в пользу других каналов
#  обратить внимание на европейский рынок (увеличить расходы на рекламу в lambdaMediaAds и снизить в AdNonSenseи)
#  привлечь и замотивировать органических пользователей

# Стоимость привлечения американского пользователя гораздо выше, чем в других странах. Как следствие, реклама в США не окупается. Наихудшее удержание также в США. Возможно, стоит сократить расходы на рекламу в США и сэкономленные средства направить на улучшение адаптации приложения под вкусы американской аудитории.

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №1__
#     
#   ❌ В этом проекте очень важно дать интерпретацию графикам и написать развернутые выводы.
# Нам надо найти и указать причины плохой окупаемости рекламы: 
# - реклама в США - проблемы такие-то
# - канал TipTop - увеличение ..?
# - канал FaceBoom, AdNonSense - проблемы с ..?\
# и т.д.
# 
# А так-же еще написать рекомендации по альтернативным регионам и каналам: на какой окупающийся источник следует обратить внимание, чтобы оптимизировать затраты на рекламу в США? Какие перспективные каналы посоветовать по Европе, рассмотреть ситуацию с устройствами и т.д.  маркетологи оценят! 🙂
# 
# </div>

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Комментарий от ревьюера №2__
#     
# Тут навлогично, не увидел: а какая проблема в рекламе США? Какие проблемы с каналом AdNonSense и т.п. Не стоит скупиться на слова
# 
# </div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Комментарий от ревьюера №3__
# Отличные рекомендации для отдела маркетинга) Молодец
# </div>

# 

# 

# <div class="alert alert-block alert-warning">📝
# Комментарий от ревьюера №1 </b> 
# 
# 
# 
# У тебя получилась очень сильная и хорошая работа. Здорово, что расчеты ты сопровождаешь иллюстрациями, а так же не забываешь про комментарии, твой проект интересно проверять. 
# 
# ---
# 
# Нужно поправить:
# 
# 1) Вводная часть
# 
# 2) Отсутствует проверка на пропуски и дубликаты
# 
# 3) Визуализировать динамику расходов по каналам, по месяцам и неделям (2 графика)
# 
# 4) После каждого раздела / графика (или серии тестов) писать вывод по полученным данным с учетом поставленной бизнес задачи
# 
# 5) Окупаемость девайсов
# 
# 6) Конверсия и удержание по метрикам (страны / каналы / девайсы)
# 
# 7) Финальный вывод
# 
# 8) Подправить выводы, после изменений
# 
# ----
# 
# 
# Если у тебя будут какие-то вопросы по моим комментариям - обязательно пиши! Буду ждать работу на повторное ревью :)</div>

# <div class="alert alert-info"> <b> Дополнил все пункты, добавил конверсии по всем критерияем, окупаемость девайсов, проверил дубликаты и пропуски, дополнил выводы к графикам и финальный вывод</b>

# <div class="alert alert-block alert-warning">📝
# Комментарий от ревьюера №2 </b> 
# 
# 
# 
# Отличная работа, осталось поправить несколько моментов: 
# 
# ---
# 
# Нужно поправить:
# 
# 
# 1) После каждого раздела / графика (или серии тестов) писать вывод по полученным данным с учетом поставленной бизнес задачи
# 
# 2) Окупаемость девайсов
# 
# 3) Финальный вывод
# 
# 4) Подправить выводы, после изменений
# 
# ----
# 
# 
# Если у тебя будут какие-то вопросы по моим комментариям - обязательно пиши! Буду ждать работу на повторное ревью :)</div>

# <div style="border:solid blue 3px; padding: 20px">
# <div class="alert alert-block alert-success">✔️
#     
# 
# __Коментарий от ревьюера №3__
# 
#     
# В остальном всё чудно😊. Твой проект так и просится на github =)   
#     
# Поздравляю с успешным завершением проекта 😊👍
# И желаю успехов в новых работах 😊
# 
# ---
# 
# От себя хочу порекомендовать тебе отличный метериал про продуктовую аналитику Дмитрия Животворева. 
#     
# https://www.youtube.com/watch?v=Vy_rq-x9QEo
#     

# 
