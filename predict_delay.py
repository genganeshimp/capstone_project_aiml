import numpy as np
import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features

new_df = pd.read_csv('data/new_shipments.csv')
hubs_df = pd.read_csv('data/hubs.csv')
model = joblib.load('models/delay_model.pkl')
threshold = float(pd.read_csv('models/delay_threshold.csv')['threshold'].iloc[0])
feature_order = pd.read_csv('models/feature_order.csv')['feature'].tolist()
impute_df = pd.read_csv('models/impute_values.csv')
medians = dict(zip(impute_df['column'], impute_df['median'].astype(float)))
segments_df = pd.read_csv('outputs/hub_segments.csv')

clean_df, _ = clean_shipments(new_df, medians=medians)
features_df = build_features(clean_df, hubs_df)

X_new = features_df[feature_order]
features_df['delay_probability'] = np.round(model.predict_proba(X_new)[:, 1], 4)
features_df['flagged'] = np.where(features_df['delay_probability'] >= threshold, 1, 0)

features_df = features_df.merge(segments_df[['hub_id', 'segment']], on='hub_id', how='left')
features_df['action'] = np.select(
    [(features_df['flagged'] == 1) & (features_df['segment'] == 'Escalate'),
     features_df['flagged'] == 1],
    ['Escalate to hub manager', 'Proactive customer call'],
    default='No action')

cols = ['shipment_id', 'hub_id', 'segment', 'delay_probability', 'flagged', 'action']
features_df[cols].to_csv('outputs/new_shipment_actions.csv', index=False)
