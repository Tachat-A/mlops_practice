import pytest
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Загружаем модель из файла
with open('model3.pkl', 'rb') as f:
    model = pickle.load(f)

# Загружаем датасет из файла
df1 = pd.read_csv('dataset1.csv')

def test_dataset1():
    # Предсказание модели
    approx = model.predict(df1['Feature1'].values.reshape(-1, 1))

    # Вычисление среднеквадратичной ошибки и R2-оценки
    mse = mean_squared_error(df1['Target'], approx)
    r2 = r2_score(df1['Target'], approx)

    # Проверка, что mse не превышает 2
    assert mse <= 2, f'Dataset 1: MSE is {mse}, что больше допустимого значения [2]'

    # Проверка, что r2 находится в диапазоне от 0 до 1
    assert 0 <= r2 <= 1, f'Dataset 1: R2 score is {r2}, что не находится в допустимом диапазоне [0, 1]'

test_dataset1()
