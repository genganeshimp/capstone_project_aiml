import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,  cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
reg_df = pd.read_csv('data/modelling_table.csv')

reg_df = reg_df[FEATURES + ['delivery_hours']].copy()

X = reg_df[FEATURES]
y = reg_df['delivery_hours']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

train_rows = X_train.shape[0]
val_rows = X_val.shape[0]
test_rows = X_test.shape[0]
train_scaled_max_abs_mean = np.max(np.abs(X_train_scaled.mean(axis=0)))
test_scaled_max_abs_mean = np.max(np.abs(X_test_scaled.mean(axis=0)))

bmetric_df = pd.DataFrame({
    'metric': ['train_rows', 'val_rows', 'test_rows', 'train_scaled_max_abs_mean', 'test_scaled_max_abs_mean'],
    'value': [train_rows, val_rows, test_rows, train_scaled_max_abs_mean, test_scaled_max_abs_mean]
}).round(4)

bmetric_df.to_csv('outputs/part_b_checks.csv', index=False)

#3. Baseline
y_train_mean = y_train.mean()
y_val_pred_baseline = [y_train_mean] * len(y_val)
baseline_mae = mean_absolute_error(y_val, y_val_pred_baseline)
baseline_mse = mean_squared_error(y_val, y_val_pred_baseline)
baseline_rmse = root_mean_squared_error(y_val, y_val_pred_baseline)
baseline_r2 = r2_score(y_val, y_val_pred_baseline)

#4.
dlst = [1, 2, 3, 4, 6, 8, 12]
score_list = []
for d in dlst:
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)
    score_list.append({'max_depth': d, 'train_rmse': train_rmse, 'val_rmse': val_rmse})

score_df = pd.DataFrame(score_list).round(4)
score_df.to_csv('outputs/part_b_tree_depth.csv', index=False)
best_depth = score_df.loc[score_df['val_rmse'].idxmin(), 'max_depth']

#5. Four candidate models
candidate_lst=[]
candidate_lst.append({'model': 'mean_baseline', 'split': 'val', 
                       'MAE': baseline_mae, 
                       'MSE': baseline_mse, 
                       'RMSE': baseline_rmse, 
                       'R2': baseline_r2})
lr_pipe = Pipeline([('scaler', StandardScaler()),
                    ('model', LinearRegression())])
models_list = [
    ('linear_regression', lr_pipe),
    ('decision_tree', DecisionTreeRegressor(max_depth=best_depth, random_state=42)),
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))]

for model_name, model in models_list:
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    candidate_lst.append({'model': model_name, 'split': 'val', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})

candidate_df = pd.DataFrame(candidate_lst).round(4)
candidate_df.to_csv('outputs/part_b_metrics.csv', index=False)

assert np.allclose(
    candidate_df['RMSE'],
    np.sqrt(candidate_df['MSE']),
    atol=0.001
)

# 6.Learning-rate experiment
lr_lst =  []
for lr in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    lr_lst.append({'learning_rate': lr, 'val_rmse': val_rmse, 'val_r2': val_r2})

lr_df = pd.DataFrame(lr_lst).round(4)
lr_df.to_csv('outputs/part_b_gb_learning_rate.csv', index=False)

#  Learning-rate experiment. Fit GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42) 
# for lr in 0.01, 0.1, 1.0 and write outputs/part_b_gb_learning_rate.csv (learning_rate, val_rmse, val_r2).

#Cross-validated comparison and final model.
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

final_model_lst = candidate_df.sort_values(by='RMSE').iloc[0:2]['model'].tolist()
cv_lst = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for model_nm, model in models_list:
    if model_nm in final_model_lst:
        cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='r2')
        cv_lst.append({'model': model_nm, 'mean_r2': np.mean(cv_scores), 'std_r2': np.std(cv_scores)})

cv_df = pd.DataFrame(cv_lst).round(4)
cv_df.to_csv('outputs/part_b_cv.csv', index=False)

#FINAL MODEL
final_model = candidate_df.loc[candidate_df[candidate_df['split'] == 'val']['RMSE'].idxmin()]['model']
for model_nm, model in models_list:
    if model_nm == final_model:
        # model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        candidate_lst.append({'model': model_nm, 'split': 'test', 'MAE': test_mae, 'MSE': test_mse, 'RMSE': test_rmse, 'R2': test_r2})

candidate_df = pd.DataFrame(candidate_lst).round(4)
candidate_df.to_csv('outputs/part_b_metrics.csv', index=False)
