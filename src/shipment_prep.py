import numpy as np
import pandas as pd

def clean_shipments(raw, medians=None):
    clean_df = raw.copy()
    clean_df.drop_duplicates(inplace=True)
    clean_df['service_type'] = clean_df['service_type'].str.strip().str.title()
    clean_df['payment_mode'] = clean_df['payment_mode'].fillna('Unknown')
    clean_df['declared_value_inr'] = clean_df['declared_value'].str.replace(',','').str.replace('₹','')
    clean_df['declared_value_inr'] = pd.to_numeric(clean_df['declared_value_inr'], errors='coerce')
    # clean_df.drop('declared_value', axis=1, inplace=True)
    clean_df.drop(columns=['declared_value'],  inplace=True)
    clean_df['dispatch_time'] = pd.to_datetime(clean_df['dispatch_time'], errors='coerce')
    # clean_df.drop('declared_value', axis=1, inplace=True)
    clean_df['parcel_weight_kg'] = clean_df['parcel_weight_kg'].where(clean_df['parcel_weight_kg'] <= 100, np.nan)
    distance_km_median =  clean_df[clean_df['distance_km'].notnull()]['distance_km'].median().round(2)
    # print(distance_km_median)
    # print(clean_df['distance_km'].median().round(2))
    parcel_weight_kg_median =  clean_df[clean_df['parcel_weight_kg'].notnull()]['parcel_weight_kg'].median().round(2)
    declared_value_inr_median =  clean_df[clean_df['declared_value_inr'].notnull()]['declared_value_inr'].median().round(2)

    clean_df['distance_km'] = clean_df['distance_km'].fillna(distance_km_median)
    clean_df['parcel_weight_kg'] = clean_df['parcel_weight_kg'].fillna(parcel_weight_kg_median)
    clean_df['declared_value_inr'] = clean_df['declared_value_inr'].fillna(declared_value_inr_median)

    medians = {'distance_km':distance_km_median.tolist(), 
                'parcel_weight_kg':parcel_weight_kg_median, 
                'declared_value_inr':declared_value_inr_median}
    return  (clean_df, medians)

def build_features(clean, hubs):
    clean_raw_hubs = pd.merge(clean, hubs, on = 'hub_id')
    PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

    clean_raw_hubs['promised_hours'] = clean_raw_hubs['service_type'].map(PROMISED_HOURS).astype(int)
    clean_raw_hubs['is_cod'] = np.where(clean_raw_hubs['payment_mode'] == 'COD', 1, 0)
    clean_raw_hubs['is_weekend'] = np.where(clean_raw_hubs['dispatch_time'].dt.day_name().isin(['Saturday','Sunday']), 1, 0)
    clean_raw_hubs['dispatch_hour'] = clean_raw_hubs['dispatch_time'].dt.hour
    clean_raw_hubs['log_declared_value'] = np.log(clean_raw_hubs['declared_value_inr'])

    return clean_raw_hubs