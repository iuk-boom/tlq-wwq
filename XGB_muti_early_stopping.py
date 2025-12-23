import time
import os
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ---------------------- 全局配置 ----------------------
os.environ['OMP_NUM_THREADS'] = '4'
start_time = time.time()
RANDOM_SEED = 217

# ---------------------- 1. 数据加载与预处理（不变） ----------------------
train_dataSet = pd.read_csv(r'D:\PycharmProjects\final\modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'D:\PycharmProjects\final\modified_数据集Time_Series662_detail.dat')

target_columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

train_dataSet['TIMESTAMP'] = pd.to_datetime(train_dataSet['TIMESTAMP'], format='mixed')
test_dataSet['TIMESTAMP'] = pd.to_datetime(test_dataSet['TIMESTAMP'], format='mixed')
train_dataSet.sort_values('TIMESTAMP', inplace=True)
test_dataSet.sort_values('TIMESTAMP', inplace=True)

train_data = train_dataSet.copy()
test_data = test_dataSet.copy()
train_data['is_train'] = 1
test_data['is_train'] = 0

train_data_with_time_features = train_data.copy()
test_data_with_time_features = test_data.copy()

# ---------------------- 2. 特征工程（不变，保留时间+噪声特征） ----------------------
def add_time_features(df, is_train=True, train_df=None):
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['minute'] = df['TIMESTAMP'].dt.minute
    df['second'] = df['TIMESTAMP'].dt.second

    lag_steps = [1, 2]
    for col in target_columns:
        for lag in lag_steps:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    window_size = 5
    for col in target_columns:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

    if is_train:
        df = df.fillna(method='ffill').fillna(0)
    else:
        for col in target_columns:
            for lag in lag_steps:
                mean_val = train_df[f'{col}_lag_{lag}'].mean()
                df[f'{col}_lag_{lag}'].fillna(mean_val, inplace=True)
            mean_val = train_df[f'{col}_rolling_mean'].mean()
            df[f'{col}_rolling_mean'].fillna(mean_val, inplace=True)
            mean_val = train_df[f'{col}_rolling_std'].mean()
            df[f'{col}_rolling_std'].fillna(mean_val, inplace=True)
    return df

train_data_with_time_features = add_time_features(train_data_with_time_features, is_train=True)
test_data_with_time_features = add_time_features(test_data_with_time_features, is_train=False,
                                                train_df=train_data_with_time_features)

full_data = pd.concat([train_data_with_time_features, test_data_with_time_features], ignore_index=True)

def add_additional_features(df):
    df['noise_mean'] = df[noise_columns].mean(axis=1)
    df['noise_std'] = df[noise_columns].std(axis=1)
    df['noise_max_min'] = df[noise_columns].max(axis=1) - df[noise_columns].min(axis=1)

    for i, col in enumerate(target_columns):
        df[f'{col}_error_ratio'] = df[col] / (df[noise_columns[i]].abs() + 1e-6)

    df['all_feature_mean'] = df[target_columns + noise_columns].mean(axis=1)
    return df

full_data = add_additional_features(full_data)

# ---------------------- 3. 数据准备（不变） ----------------------
train_data = full_data[full_data['is_train'] == 1].drop(['is_train', 'TIMESTAMP'], axis=1)
test_data = full_data[full_data['is_train'] == 0].drop(['is_train', 'TIMESTAMP'], axis=1)

feature_cols = noise_columns + [
    'noise_mean', 'noise_std', 'noise_max_min',
    'T_SONIC_error_ratio', 'CO2_density_error_ratio', 'CO2_density_fast_tmpr_error_ratio',
    'H2O_density_error_ratio', 'H2O_sig_strgth_error_ratio', 'CO2_sig_strgth_error_ratio',
    'all_feature_mean', 'hour', 'minute', 'second'
]
for col in target_columns:
    feature_cols.extend([f'{col}_lag_1', f'{col}_lag_2', f'{col}_rolling_mean', f'{col}_rolling_std'])

X_train = train_data[feature_cols].values.astype(np.float32)
X_test = test_data[feature_cols].values.astype(np.float32)
y_train = train_data[target_columns].values.astype(np.float32)
y_test = test_data[target_columns].values.astype(np.float32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
)

# ---------------------- 4. 贝叶斯优化调参（核心调整：降低正则化，放开模型） ----------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800, step=100),  # 增加迭代次数，让模型学够
        'max_depth': trial.suggest_int('max_depth', 4, 7),  # 树深放宽到7，提升拟合能力
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06, log=True),  # 稍降低学习率，稳步学习
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # 提高样本抽样比例，保留更多信息
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),  # 提高特征抽样比例
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True),  # 大幅降低L1正则（之前10-80太狠）
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),  # 大幅降低L2正则
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),  # 降低子节点权重，允许模型学细节
        'objective': 'reg:squarederror',
        'seed': RANDOM_SEED,
        'n_jobs': 4,
        'tree_method': 'hist',
        'verbose': -1
    }

    model = XGBRegressor(**params)
    model.fit(X_train_split, y_train_split[:, 0])

    y_val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val[:, 0], y_val_pred)
    return val_mae

study = optuna.create_study(direction='minimize', study_name='xgb_mae_optimization_fix_error')
study.optimize(objective, n_trials=10, timeout=900)

best_params = study.best_params
print("贝叶斯优化最优参数：")
print(best_params)

# ---------------------- 5. 训练6个单输出模型（不变） ----------------------
best_params.update({
    'objective': 'reg:squarederror',
    'seed': RANDOM_SEED,
    'n_jobs': 4,
    'tree_method': 'hist',
    'verbose': 10
})

models = []
y_predict = np.zeros_like(y_test)

print("\n开始训练6个单输出模型...")
for i, target_col in enumerate(target_columns):
    print(f"\n=== 训练第 {i + 1} 个模型（目标特征：{target_col}）===")
    model = XGBRegressor(**best_params)
    model.fit(X_train_scaled, y_train[:, i])
    models.append(model)
    y_predict[:, i] = model.predict(X_test_scaled)

# ---------------------- 6. 结果评估与保存（不变） ----------------------
overall_mean_error = np.mean(np.abs(y_test - y_predict))
print(f"\n==================== 结果 ====================")
print(f"最终模型总体平均误差: {overall_mean_error:.4f}")

results = []
for true_val, pred_val in zip(y_test, y_predict):
    error = np.abs(true_val - pred_val)
    results.append([
        ' '.join(map(str, true_val)),
        ' '.join(map(str, pred_val)),
        ' '.join(map(str, error))
    ])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_XGB_fix_error_optimized.csv", index=False)
print(f"结果已保存至: result_XGB_fix_error_optimized1.csv")

end_time = time.time()
total_time = end_time - start_time
print(f"总耗时：{total_time:.3f}秒（{total_time / 60:.2f}分钟）")
print("=============================================")