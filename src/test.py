import numpy as np
import pandas as pd
import os

print(os.getcwd())


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
clean_df = pd.read_csv('../data/raw_shipments.csv')
print(clean_df.head())

clean_shipments(clean_df, medians=None)