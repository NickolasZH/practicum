import combining
import phik
import matplotlib.pyplot as plt
import seaborn as sns

# Выбор количественных признаков для анализа
quantitative_features = [
    'маркет_актив_6_мес', 'маркет_актив_тек_мес', 'длительность', 
    'акционные_покупки', 'средний_просмотр_категорий_за_визит', 
    'неоплаченные_продукты_штук_квартал', 'ошибка_сервиса', 'страниц_за_визит',
    'выручка_предыдущий_месяц', 'выручка_препредыдущий_месяц', 'выручка_текущий_месяц', 
    'время_предыдущий_месяц', 'время_текущий_месяц'
]

# Рассчитываем PhiK корреляцию между количественными признаками
phik_corr = combining.merged_data.phik_matrix(interval_cols=quantitative_features)

# Визуализация корреляционной матрицы PhiK
plt.figure(figsize=(12, 10))
sns.heatmap(phik_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.8})
plt.title("PhiK Корреляционная матрица")
plt.show()

# Фильтрация корреляций, исключая NaN и вывод только корреляций больше 0.9
high_corr = phik_corr.stack().reset_index()
high_corr.columns = ['feature_1', 'feature_2', 'phi_k_value']

# Оставляем только те строки, где корреляция больше 0.9, исключая саморефлексию (где feature_1 == feature_2)
high_corr = high_corr[(high_corr['phi_k_value'] > 0.9) & (high_corr['feature_1'] != high_corr['feature_2'])]

# Выводим результаты
print("Пары признаков с высокой корреляцией (больше 0.9):")
print(high_corr)