import pandas as pd
from IPython.display import display

# Загрузка данных
market_file = pd.read_csv('marketing/market_file.csv', delimiter=',', decimal=',')
market_money = pd.read_csv('marketing/market_money.csv', delimiter=',', decimal=',')
market_time = pd.read_csv('marketing/market_time.csv', delimiter=',', decimal=',')
money = pd.read_csv('marketing/money.csv', delimiter=';', decimal=',')

# Проверка первых 5 строк для каждого файла
print("market_file.csv:\n", market_file.head())
display(market_file.info())
print("\nmarket_money.csv:\n", market_money.head())
display(market_money.info())
print("\nmarket_time.csv:\n", market_time.head())
display(market_time.info())
print("\nmoney.csv:\n", money.head())
display(money.info())

# Функция для изменения названий столбцов: приведём к одному виду с "_" вместо пробелов и нижнему регистру
def clean_column_names(df):
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]

# Применяем к таблицам
clean_column_names(market_file)
clean_column_names(market_money)
clean_column_names(market_time)
clean_column_names(money)

# Проверка новых названий столбцов
print("Market File Columns:", market_file.columns)
print("Market Money Columns:", market_money.columns)
print("Market Time Columns:", market_time.columns)
print("Money Columns:", money.columns)

# Преобразование данных в числовой формат
market_file['маркет_актив_6_мес'] = pd.to_numeric(market_file['маркет_актив_6_мес'], errors='coerce')
market_file['акционные_покупки'] = pd.to_numeric(market_file['акционные_покупки'], errors='coerce')
market_money['выручка'] = pd.to_numeric(market_money['выручка'], errors='coerce')

# Проверка типов данных после преобразования
print("Проверка типов данных после преобразования:")
print("market_file:\n", market_file[['маркет_актив_6_мес', 'акционные_покупки']].dtypes)
print("\nmarket_money:\n", market_money['выручка'].dtypes)

# Проверка на наличие пропущенных значений после преобразования
print("\nПропущенные значения в market_file:")
print(market_file.isnull().sum())
print("\nПропущенные значения в market_money:")
print(market_money.isnull().sum())