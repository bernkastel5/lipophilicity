"""
Решение по предсказанию липофильности (LogP) на основе SMILES.
Методология:
    1. Загрузка и предобработка данных (парсинг SMILES, санация, вычисление RDKit дескрипторов).
    2. Усиление модели:
       - Использование ансамблевого стэкинга, где базовыми моделями являются RandomForestRegressor,
         XGBRegressor и LGBMRegressor, а финальный мета-регрессор — Ridge.
       - Применяется 5-кратная кросс-валидация для оценки качества.
    3. Обучение финальной модели на полном тренировочном наборе.
    4. Генерация предсказаний для тестовой выборки с сохранением итогового файла, содержащего 'ID' и 'LogP'.

Используемые библиотеки: pandas, numpy, rdkit, scikit-learn, xgboost, lightgbm, joblib.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import StackingRegressor

import xgboost as xgb
import lightgbm as lgb

import joblib

# Отключаем вывод предупреждений RDKit (ошибки уровня 'rdApp.error')
RDLogger.DisableLog('rdApp.error')


def compute_descriptors(smiles):
    """
    Вычисляет набор молекулярных дескрипторов для заданного SMILES.
    Если парсинг, санация или вычисление хотя бы одного дескриптора не удаётся, возвращает None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        desc_dict = {}
        for name, func in Descriptors.descList:
            try:
                desc_val = func(mol)
                desc_dict[name] = desc_val
            except Exception:
                desc_dict[name] = np.nan
        if any(np.isnan(list(desc_dict.values()))):
            return None
        return desc_dict
    except KeyboardInterrupt:
        raise
    except Exception:
        return None


def generate_descriptor_dataframe(df, smiles_column='SMILES'):
    """
    Генерирует DataFrame с дескрипторами для каждого SMILES из входного DataFrame.
    Пропускает строки, для которых не удалось вычислить дескрипторы.
    Выводит количество обработанных/пропущенных молекул.
    """
    descriptors_list = []
    valid_indices = []
    skipped = 0
    for idx, smile in df[smiles_column].items():
        desc = compute_descriptors(smile)
        if desc is None:
            skipped += 1
            continue
        descriptors_list.append(desc)
        valid_indices.append(idx)
    total = len(df)
    print(f"Обработано молекул: {total - skipped} из {total}. Пропущено: {skipped}.")

    if not descriptors_list:
        return pd.DataFrame()
    desc_df = pd.DataFrame(descriptors_list, index=valid_indices)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    df_desc = desc_df.reset_index(drop=True)
    df_final = pd.concat([df_valid, df_desc], axis=1)
    return df_final


def train_model(train_df):
    """
    Обучает ансамблевую модель на основе стэкинга, используя RandomForest, XGBoost и LightGBM
    в качестве базовых моделей и Ridge в качестве финального регрессора.
    Проводит 5-кратную кросс-валидацию и выводит средние значения RMSE и R².
    Возвращает обученную модель и список признаков.
    """
    df_features = generate_descriptor_dataframe(train_df, smiles_column='SMILES')
    if 'LogP' not in df_features.columns:
        raise ValueError("В обучающем датасете отсутствует столбец 'LogP' с целевыми значениями.")

    # Отделяем признаки от целевой переменной (исключая столбцы 'SMILES' и 'LogP')
    features = df_features.drop(columns=['SMILES', 'LogP'])
    target = df_features['LogP']

    # Определяем базовые модели
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1))
    ]
    # Финальный регрессор
    final_estimator = Ridge(alpha=1.0)

    # Стэкинг-регрессор
    model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)

    # Выполнение кросс-валидации (5 фолдов)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(model, features, target, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_r2 = cross_val_score(model, features, target, cv=kf, scoring='r2', n_jobs=-1)

    print("5-Fold CV RMSE: {:.4f} ± {:.4f}".format(cv_rmse.mean(), cv_rmse.std()))
    print("5-Fold CV R²: {:.4f} ± {:.4f}".format(cv_r2.mean(), cv_r2.std()))

    # Обучение модели на полном наборе данных
    model.fit(features, target)

    return model, features.columns


def predict_test(model, feature_columns, test_df):
    """
    Вычисляет дескрипторы для SMILES тестового датасета, генерирует предсказания LogP и
    возвращает DataFrame с результатами, содержащий столбцы 'ID' и 'LogP'.
    """
    test_processed = generate_descriptor_dataframe(test_df, smiles_column='SMILES')
    if test_processed.empty:
        raise ValueError("Не удалось обработать ни одну молекулу из тестового датасета")
    X_test = test_processed[feature_columns]
    predictions = model.predict(X_test)

    submission = test_processed[['ID']].copy()
    submission['LogP'] = predictions
    return submission


def main():
    # Пути к файлам
    train_path = 'final_train_data80.csv'
    test_path = 'final_test_data80.csv'

    # Загрузка данных
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Обучение модели с кросс-валидацией и стэкингом
    model, feature_columns = train_model(train_df)

    # Сохранение обученной модели
    model_filename = 'model_logp.pkl'
    joblib.dump(model, model_filename)
    print("Модель сохранена в файле:", model_filename)

    # Генерация предсказаний для тестового набора
    submission_df = predict_test(model, feature_columns, test_df)

    submission_file = 'submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print("Файл с предсказаниями сохранен как:", submission_file)


if __name__ == '__main__':
    main()
