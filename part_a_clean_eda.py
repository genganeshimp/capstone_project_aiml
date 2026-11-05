import pandas as pd
import numpy as np
import os
from src.shipment_prep import clean_shipments, build_features, FEATURES

OTHER = ['delivery_hours', 'is_delayed']

#4. Read Data
# print(os.getcwd())
raw_df = pd.read_csv('data/raw_shipments.csv')
hubs_df = pd.read_csv('data/hubs.csv')

clean_df, medians = clean_shipments(raw_df, medians=None)
modelling_table = build_features(clean_df, hubs_df)

modelling_table['is_delayed'] = np.where(modelling_table['delivery_hours'] > modelling_table['promised_hours'], 1, 0)

modelling_table.to_csv('data/modelling_table.csv', index=False)

modelling_df = pd.read_csv('data/modelling_table.csv')

#5.Checks file
dedup_df = raw_df.drop_duplicates()
metric_dict = {}
metric_dict['raw_rows'] =  raw_df.shape[0]
metric_dict['duplicate_rows_removed'] = raw_df.duplicated().sum()
metric_dict['rows_after_dedup'] = metric_dict['raw_rows'] - metric_dict['duplicate_rows_removed']
metric_dict['missing_distance_km_before'] = dedup_df['distance_km'].isnull().sum()
metric_dict['missing_parcel_weight_before'] = dedup_df['parcel_weight_kg'].isnull().sum()
metric_dict['sentinel_weight_rows'] = dedup_df[dedup_df['parcel_weight_kg'] > 100].shape[0]
metric_dict['missing_declared_value_before'] = dedup_df['declared_value'].isnull().sum()
metric_dict['missing_payment_mode_before'] = dedup_df['payment_mode'].isnull().sum()
metric_dict['service_type_spellings_before'] = dedup_df['service_type'].nunique()
metric_dict['service_type_spellings_after'] = modelling_df['service_type'].nunique()
metric_dict['declared_value_mean_inr'] = modelling_df['declared_value_inr'].mean()
metric_dict['log_declared_value_mean'] =  modelling_df['log_declared_value'].mean()
metric_dict['median_distance_km'] = medians['distance_km']
metric_dict['median_parcel_weight_kg'] = modelling_df['parcel_weight_kg'].median()
metric_dict['median_declared_value_inr'] = modelling_df['declared_value_inr'].median()
metric_dict['rows_after_hub_merge'] = modelling_df.shape[0]
metric_dict['nulls_in_features_and_target'] = modelling_df[FEATURES + OTHER].isnull().sum().sum()
metric_dict['delayed_count'] = modelling_df['is_delayed'].sum()
metric_dict['delay_rate'] = modelling_df['is_delayed'].mean()
q1 = modelling_df['delivery_hours'].quantile(0.25)
q3 = modelling_df['delivery_hours'].quantile(0.75)
metric_dict['hours_q1'] = q1
metric_dict['hours_q3'] = q3
metric_dict['hours_lower_bound'] = q1 - 1.5 * (q3 - q1)
metric_dict['hours_upper_bound'] = q3 + 1.5 * (q3 - q1)
metric_dict['hours_outlier_count'] = modelling_df[(modelling_df['delivery_hours'] < metric_dict['hours_lower_bound']) | (modelling_df['delivery_hours'] > metric_dict['hours_upper_bound'])]['delivery_hours'].size
metric_dict['declared_value_median_inr'] = modelling_df['declared_value_inr'].median()
metric_dict['log_declared_value_median'] =  modelling_df['log_declared_value'].median()

metric_df = pd.DataFrame(list(metric_dict.items()), columns=['metric','value']).round(4)
metric_df.to_csv('outputs/part_a_checks.csv', index=False)

#6.Delay evidence table
service_df = modelling_df.groupby('service_type')\
                         .agg(shipments= ('shipment_id','count'),\
                          delayed= ('is_delayed', 'sum'),\
                          mean_hours= ('delivery_hours', 'mean')).reset_index()
service_df = service_df.rename(columns={'service_type': 'group'})
service_df['dimension'] = 'service_type'

hub_tier_df = modelling_df.groupby('hub_tier')\
                            .agg(shipments= ('shipment_id','count'),\
                            delayed= ('is_delayed', 'sum'),\
                            mean_hours= ('delivery_hours', 'mean')).reset_index()
hub_tier_df = hub_tier_df.rename(columns={'hub_tier': 'group'})
hub_tier_df['dimension'] = 'hub_tier'

bins = [0, 200, 400, 600, 2000]
labels = ['0-200', '201-400', '401-600', '601+']
modelling_df['dist_band'] = pd.cut(
    modelling_df['distance_km'],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True
)
dist_df = modelling_df.groupby('dist_band', observed=False)\
                      .agg(shipments= ('shipment_id','count'),\
                       delayed= ('is_delayed', 'sum'),\
                       mean_hours= ('delivery_hours', 'mean')).reset_index()
dist_df = dist_df.rename(columns={'dist_band': 'group'})
dist_df['dimension'] = 'distance_band'
modelling_df.drop(columns=['dist_band'], inplace=True)

det_df = pd.concat([service_df, hub_tier_df, dist_df])
det_df['delay_rate'] = np.round(det_df['delayed'] / det_df['shipments'],4)
cols = ['dimension', 'group', 'shipments', 'delayed', 'delay_rate', 'mean_hours']
det_df[cols].round(4).to_csv('outputs/part_a_delay_by_group.csv', index=False)

#7. Correlations
corr_matrix = modelling_df.corr(numeric_only=True).round(4)
corr_matrix.to_csv('outputs/part_a_correlations.csv')