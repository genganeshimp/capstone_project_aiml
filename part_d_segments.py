import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1.Hub profile
hub_df = pd.read_csv('data/modelling_table.csv')

hub_profile_df = hub_df.groupby('hub_id').agg(shipments = ('shipment_id','count'),\
                             mean_delivery_hours = ('delivery_hours','mean'),\
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

# 3.Separation evidence
pair_dist = np.sqrt(((X_scaled[:, None, :] - X_scaled[None, :, :]) ** 2).sum(axis=2))
iu = np.triu_indices(len(X_scaled), k=1)   # 276 unordered hub pairs
sep_lst = []
for K in [2, 3, 4, 5]:
    labels_k = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X_scaled)
    same = labels_k[iu[0]] == labels_k[iu[1]]
    across_over_within = pair_dist[iu][~same].mean() / pair_dist[iu][same].mean()
    sizes = sorted(np.bincount(labels_k).tolist(), reverse=True)
    sep_lst.append({'K': K, 'across_over_within': across_over_within,
                    'cluster_sizes': '-'.join(str(s) for s in sizes)})

sep_df = pd.DataFrame(sep_lst).round(4)
sep_df.to_csv('outputs/part_d_separation.csv', index=False)

# 4.Segment names and tier check
cluster_rate = hub_profile_df.groupby('cluster')['delay_rate'].mean().sort_values()
segment_names = dict(zip(cluster_rate.index, ['Standard', 'Watch', 'Escalate']))
hub_profile_df['segment'] = hub_profile_df['cluster'].map(segment_names)

hubs_df = pd.read_csv('data/hubs.csv')
hub_segments_df = hub_profile_df.merge(hubs_df[['hub_id', 'hub_city', 'hub_tier']], on='hub_id', how='left')
hub_segments_df[['hub_id', 'segment', 'hub_city', 'hub_tier']].to_csv('outputs/hub_segments.csv', index=False)

segment_profile_df = hub_profile_df.groupby('segment')\
                                   .agg(hubs = ('hub_id', 'count'),\
                                        mean_delivery_hours = ('mean_delivery_hours', 'mean'),\
                                        delay_rate = ('delay_rate', 'mean'),\
                                        mean_distance_km = ('mean_distance_km', 'mean'),\
                                        mean_n_stops = ('mean_n_stops', 'mean')).reset_index()
segment_profile_df['segment'] = pd.Categorical(segment_profile_df['segment'], ['Standard', 'Watch', 'Escalate'], ordered=True)
segment_profile_df = segment_profile_df.sort_values('segment').round(4)
segment_profile_df.to_csv('outputs/segment_profile.csv', index=False)

tier_segment = {1: 'Standard', 2: 'Watch', 3: 'Escalate'}
mismatch_df = hub_segments_df[hub_segments_df['segment'] != hub_segments_df['hub_tier'].map(tier_segment)]
mismatch_df[['hub_id', 'segment', 'hub_city', 'hub_tier']].to_csv('outputs/tier_vs_segment_mismatch.csv', index=False)
