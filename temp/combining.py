import stat_analysis

from IPython.display import display

# Отберём клиентов с покупательской активностью не менее 3 месяцев
active_clients = stat_analysis.market_money_active.index

# Переименуем столбцы для market_money
stat_analysis.market_money_pivot.columns = [f'выручка_{col}' for col in stat_analysis.market_money_pivot.columns]

# Создаем pivot-таблицу для времени
market_time_pivot = stat_analysis.transforms.market_time.pivot(index='id', columns='период', values='минут')

# Переименуем столбцы для market_time
market_time_pivot.columns = [f'время_{col}' for col in market_time_pivot.columns]

# Объединяем данные
merged_data = stat_analysis.transforms.market_file.set_index('id').join(stat_analysis.market_money_pivot, how='left').join(market_time_pivot, how='left')

# Оставляем только клиентов с активностью хотя бы в 3 месяца
merged_data = merged_data.loc[merged_data.index.isin(active_clients)]

# Печать первых 5 строк итоговой таблицы
display(merged_data.head())

print(stat_analysis.transforms.market_file.shape)
print(merged_data.shape)