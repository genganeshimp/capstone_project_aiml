import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import clean_shipments, FEATURES

os.makedirs('models', exist_ok=True)

# Final configuration chosen in Part C
config = pd.read_csv('outputs/part_c_final_config.csv').iloc[0]
max_depth = None if str(config['max_depth']) == 'unlimited' else int(config['max_depth'])
threshold = float(config['threshold'])

# Rebuild Part C's stratified split
clf_df = pd.read_csv('data/modelling_table.csv')
X = clf_df[FEATURES]
y = clf_df['is_delayed']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config['selector_k']))),
    ("model", RandomForestClassifier(random_state=42,
                                     n_estimators=int(config['n_estimators']),
                                     max_depth=max_depth,
                                     min_samples_leaf=int(config['min_samples_leaf'])))
])
final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, 'models/delay_model.pkl')
pd.DataFrame({'threshold': [threshold]}).to_csv('models/delay_threshold.csv', index=False)
pd.DataFrame({'feature': FEATURES}).to_csv('models/feature_order.csv', index=False)

_, medians = clean_shipments(pd.read_csv('data/raw_shipments.csv'), medians=None)
pd.DataFrame({'column': list(medians.keys()), 'median': list(medians.values())}).round(4)\
  .to_csv('models/impute_values.csv', index=False)

# Reload model and compare
reloaded_model = joblib.load('models/delay_model.pkl')
proba_in_memory = final_model.predict_proba(X_test)[:, 1]
proba_reloaded = reloaded_model.predict_proba(X_test)[:, 1]

reload_df = pd.DataFrame({
    'metric': ['test_rows_scored', 'reload_max_abs_probability_diff', 'threshold'],
    'value': [len(X_test), abs(proba_in_memory - proba_reloaded).max(), threshold]
}).round(4)
reload_df.to_csv('outputs/part_d_reload_check.csv', index=False)
