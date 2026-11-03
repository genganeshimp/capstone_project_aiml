import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1.Hub profile
hub_df = pd.read_csv('data/modelling_table.csv')

hub_profile_df = hub_df.groupby('hub_id').agg(shipments = ('shipment_id','count'),\
                             mean_delivery_hours = ('dispatch_hour','mean'),\
                             delay_rate = ('is_delayed', 'mean'),\
                             mean_distance_km = ('distance_km', 'mean'),\
                             mean_n_stops = ('n_stops', 'mean'))
                            
hub_profile_df = hub_profile_df.reset_index().sort_values(by='hub_id').round(4)
hub_profile_df.to_csv('outputs/hub_profile.csv', index=False)

# 2.K-Means segmentation. 

khub_df = pd.read_csv('outputs/hub_profile.csv')
khub_df = khub_df.sort_values(by='hub_id').reset_index(drop=True)
features = [
    'mean_delivery_hours',
    'delay_rate',
    'mean_distance_km',
    'mean_n_stops'
]

X = khub_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile_df['cluster'] = kmeans.fit_predict(X_scaled)
