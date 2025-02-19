import transforms

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Анализ количественных признаков
print("Статистический анализ market_file:")
print(transforms.market_file.drop(columns='id', errors='ignore').describe())

print("\nСтатистический анализ market_money:")
print(transforms.market_money.drop(columns='id', errors='ignore').describe())

print("\nСтатистический анализ market_time:")
print(transforms.market_time.drop(columns='id', errors='ignore').describe())

print("\nСтатистический анализ money:")
print(transforms.money.drop(columns='id', errors='ignore').describe())

# Функция для построения гистограмм
def plot_histograms(dataframe, title):
    # Отбираем только числовые столбцы, кроме 'id'
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).drop(columns='id', errors='ignore')
    
    # Построение гистограмм
    numeric_columns.hist(bins=20, figsize=(15, 10), edgecolor='black')
    plt.suptitle(title, fontsize=16)
    plt.show()

# Построение гистограмм для каждой таблицы
plot_histograms(transforms.market_file, "Распределение количественных признаков (market_file)")
plot_histograms(transforms.market_money, "Распределение количественных признаков (market_money)")
plot_histograms(transforms.market_time, "Распределение количественных признаков (market_time)")
plot_histograms(transforms.money, "Распределение количественных признаков (money)")

# Удаляем строки с выручкой больше 100,000
market_money_cleaned = transforms.market_money[transforms.market_money['выручка'] <= 100000]

# Проверяем, что выбросы удалены
print(market_money_cleaned.drop(columns='id', errors='ignore').head())
print(market_money_cleaned.drop(columns='id', errors='ignore').describe())

# Заменяем ошибочное значение
transforms.market_file['тип_сервиса'] = transforms.market_file['тип_сервиса'].replace('стандартт', 'стандарт')

# 2. Строим график для проверки
plt.figure(figsize=(14, 5))

ax = sns.countplot(data=transforms.market_file, y='тип_сервиса', order=transforms.market_file['тип_сервиса'].value_counts().index, palette="viridis")
plt.title('Тип сервиса')
plt.xlabel('Количество')
plt.ylabel('Тип сервиса')

# Добавление подписей данных
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 1, p.get_y() + p.get_height() / 2, f'{int(width)}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# Исправим опечатку
transforms.market_time['период'] = transforms.market_time['период'].replace('предыдцщий_месяц', 'предыдущий_месяц')
display(transforms.market_time['период'].value_counts())

market_money_pivot = transforms.market_money.pivot(index='id', columns='период', values='выручка')

# Отфильтруем клиентов с покупательской активностью не менее 3 месяцев.
market_money_active = market_money_pivot[
    (market_money_pivot['предыдущий_месяц'] > 0) &
    (market_money_pivot['препредыдущий_месяц'] > 0) &
    (market_money_pivot['текущий_месяц'] > 0)
]

display(market_money_active)