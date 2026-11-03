import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
reg_df = pd.read_csv('data/modelling_table.csv')

reg_df = reg_df[FEATURES + ['is_delayed']].copy()

X = reg_df[FEATURES]
y = reg_df['is_delayed']

# 1.Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

#2. Blind baseline and three classifiers
logistic_pipe  = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(random_state=42))
])

depth = [2, 3, 4, 5, 6, 8, 10]
dclf_dict = {}
for i in depth:
    y_dclf_pred = DecisionTreeClassifier(criterion="gini", max_depth=i, random_state=42).fit(X_train, y_train).predict(X_val)
    f1 = f1_score(y_val, y_dclf_pred)
    dclf_dict[i] = f1

max_value = max(dclf_dict.values())
d = min([k for k, v in dclf_dict.items() if v == max_value])
clf_models_dict = {'blind_all_on_time': 'baseline', 
                   'logistic_regression': logistic_pipe,
                   'decision_tree_depth_'+str(d): DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42),
                   'random_forest_default': RandomForestClassifier(n_estimators=100, random_state=42)}
df_lst = []
for model_name, model in clf_models_dict.items():
    if model != 'baseline':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)       
    else:
        y_pred = np.zeros_like(y_val)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0) 
    df_lst.append({'model': model_name, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

clf_df = pd.DataFrame(df_lst).round(4)
clf_df.to_csv('outputs/part_c_models.csv', index=False)

#3. Which columns carry signal?
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
model_importances = model.feature_importances_
selector = SelectKBest(f_classif, k="all")
selector.fit(X_train, y_train)

feature_df = pd.DataFrame({"feature":FEATURES,"f_score":selector.scores_, "rf_importance":model.feature_importances_})
feature_df['f_rank'] = feature_df['f_score'].rank(ascending=False, method='min')
feature_df['is_noise'] = feature_df['feature'].isin(NOISE_COLUMNS)

feature_df = feature_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].round(4)

feature_df.to_csv("outputs/part_c_feature_scores.csv",index=False)
# feature_df.sort_values(by='f_rank', inplace=True)
k_values = [5, 7, 10, 12, 17]

selectk_lst = []
for k in k_values:
    pipe = Pipeline([
        ('selectk', SelectKBest(f_classif, k=k)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    selectk_lst.append({'k': k, 'val_f1': val_f1})

selectk_df = pd.DataFrame(selectk_lst).round(4)
selectk_df.to_csv("outputs/part_c_selectk.csv",index=False)
final_k = 7

#4. Tuning. 
selector = SelectKBest(f_classif, k=final_k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [1, 5, 10]
}
tuning_results = []
tune_dict = {
            "grid_search":GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5),
            "randomized_search":RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42)
            }

for search_name, search in tune_dict.items():
    search.fit(X_train_selected, y_train)
    tuning_results.append({
        'search': search_name,
        'candidates': len(search.cv_results_['params']),
        'cv_folds': search.cv,
        'total_fits': len(search.cv_results_['params']) * search.cv,
        'best_params': search.best_params_,
        'best_cv_accuracy': search.best_score_,
        'val_f1': f1_score(y_val, search.predict(X_val_selected))})
    

default_forest_same_7_features = RandomForestClassifier(n_estimators=100,min_samples_leaf=1 ,random_state=42)
default_forest_same_7_features.fit(X_train_selected, y_train)
best_params= {
    'n_estimators': 100,
    'min_samples_leaf': 1,
    'max_depth': None
}
tuning_results.append({
    'search': 'default_forest_same_7_features',
    'candidates': 1,
    'cv_folds': 0,
    'total_fits': 1,
    'best_params': best_params,
    'best_cv_accuracy': '',
    'val_f1': f1_score(y_val, default_forest_same_7_features.predict(X_val_selected))})

tuning_df = pd.DataFrame(tuning_results).round(4)
tuning_df.to_csv("outputs/part_c_tuning.csv",index=False)


#5. Choosing the final configuration.

final_tuned_df = pd.read_csv("outputs/part_c_tuning.csv")

conditions = [
    final_tuned_df['search'] == 'default_forest_same_7_features',
    final_tuned_df['search'] == 'grid_search',
    final_tuned_df['search'] == 'randomized_search'
]
choices = [3, 2, 1]
final_tuned_df['model_id'] = np.select(
    conditions,
    choices
)

final_tuned_df = final_tuned_df.sort_values(by=['val_f1', 'model_id'], ascending = [False, False])
final_tuned_df['f1_rank'] = final_tuned_df['val_f1'].rank(ascending=False, method='min')
max_model = final_tuned_df.loc[final_tuned_df['val_f1'].idxmax()]


best_params = max_model['best_params']
if isinstance(best_params, str):
    best_params = ast.literal_eval(best_params)

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=final_k)),
    ("model", RandomForestClassifier(random_state=42, **best_params))
])

final_model.fit(X_train, y_train)

#6. Alert threshold. 

y_val_proba = final_model.predict_proba(X_val)[:, 1]
threshold_results = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = (y_val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    threshold_results.append({'threshold': t, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

threshold_df = pd.DataFrame(threshold_results).round(4)
threshold_df.to_csv("outputs/part_c_thresholds.csv",index=False)

threshold = threshold_df[threshold_df['recall'] >= 0.8]['threshold']
alert_threshold = max(threshold) if not threshold.empty else 0.1


#7. One test evaluation. 
# Score the final model on the test rows at the chosen threshold, exactly once, 
# and write outputs/part_c_final_test.csv (threshold, TN, FP, FN, TP, accuracy, precision, recall, f1). 
# Write outputs/part_c_final_config.csv, one row, columns selector_k, n_estimators, max_depth, min_samples_leaf, threshold, 
# chosen_from (max_depth is unlimited when there is no limit; chosen_from is the name of the winning candidate in part_c_tuning.csv).

y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= alert_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)

final_test_df = pd.DataFrame([{'threshold': alert_threshold, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 
                               'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}]).round(4)

final_test_df.to_csv("outputs/part_c_final_test.csv",index=False)

final_config_df = pd.DataFrame([{'selector_k': final_k,
                                'n_estimators': best_params['n_estimators'],
                                'max_depth': best_params['max_depth'] if best_params['max_depth'] is not None else 'unlimited',
                                'min_samples_leaf': best_params['min_samples_leaf'],
                                'threshold': alert_threshold,
                                'chosen_from': max_model['search']}])
final_config_df.to_csv("outputs/part_c_final_config.csv",index=False)