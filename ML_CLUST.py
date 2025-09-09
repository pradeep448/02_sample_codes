# https://github.com/pradeep448/machine-learning/blob/main/Clustering/kmeans%20clustering%20-%20predicting%20top%20countries%20for%20HELP%20International%20NGO%20campaign/model_training/ML%20clustering%20country%20data%20final%20done.ipynb

# KMeans with Elbow and Silhouette in scikit-learn
# ------------------------------------------------
# This script:
# 1) Generates toy blob data
# 2) Computes inertia over k=1..9 for the Elbow method
# 3) Computes silhouette score over k=2..9
# 4) Picks k by max silhouette (simple heuristic) and fits final model

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import numpy as np

# 1) Generate synthetic 2D data with 4 true centers (for demo)
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)

# 2) Elbow method: inertia (within-cluster sum of squares) across k
k_values = range(1, 10)
inertias = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)  # k-means++ init by default
    km.fit(X)                                   # fit only; labels not needed for inertia
    inertias.append(km.inertia_)                # lower is better; look for an "elbow" in the curve

# 3) Silhouette scores: only valid for k >= 2
ks_for_sil = range(2, 10)
sil_scores = []
for k in ks_for_sil:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)                  # fit and get labels in one step
    sil = silhouette_score(X, labels)           # mean silhouette over all samples, range ~ [-1, 1]
    sil_scores.append(sil)

# 4) Choose k via simple heuristic (max silhouette). In practice, cross-check with Elbow.
best_k_by_sil = ks_for_sil[int(np.argmax(sil_scores))]

# 5) Fit final model using chosen k
final_kmeans = KMeans(n_clusters=best_k_by_sil, random_state=42)
final_labels = final_kmeans.fit_predict(X)
final_centers = final_kmeans.cluster_centers_
final_inertia = final_kmeans.inertia_
final_sil = silhouette_score(X, final_labels)

# 6) Report (replace prints with logging as needed)
print("Elbow (k vs inertia):")
for k, inertia in zip(k_values, inertias):
    print(f"  k={k}: inertia={inertia:.2f}")

print("\nSilhouette (k vs score):")
for k, s in zip(ks_for_sil, sil_scores):
    print(f"  k={k}: silhouette={s:.3f}")

print(f"\nChosen k (by silhouette): {best_k_by_sil}")
print(f"Final inertia: {final_inertia:.2f}")
print(f"Final silhouette: {final_sil:.3f}")
print(f"Cluster centers shape: {final_centers.shape}")








# # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# sns.set()

# # %%
# dataset_org = pd.read_csv('')
# dataset_org.head()

# # %%
# dataset_org.shape

# # %%
# dataset_org.info()

# # %%
# dataset_org.describe()

# # %%
# dataset_org.isna().sum().sum()

# # %%
# dataset_org.duplicated().sum()

# # %%
# dataset=dataset_org.drop('country',axis=1)

# # %%
# dataset.head()

# # %%
# sns.pairplot(dataset)

# # %%
# plt.figure(figsize=(10,10))
# sns.heatmap(dataset.corr(),cmap='Blues',annot=True,fmt='.2f',
#             square=True,yticklabels=dataset.columns,xticklabels=dataset.columns)

# # %%
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# dataset_sc=sc.fit_transform(dataset)

# # %%
# from sklearn.decomposition import PCA
# pca=PCA(2)
# pc=pca.fit_transform(dataset_sc)

# # %%
# plt.figure(figsize=(7,7))
# sns.heatmap(pd.DataFrame(np.c_[dataset_sc,pc]).corr().iloc[:9,9:],cmap='Blues',annot=True,
#             square=True,yticklabels=dataset.columns,xticklabels=['pc_0','pc_1'])

# # %%
# from sklearn.cluster import KMeans
# wcss=[]
# for i in range(1,11):
#     kmeans=KMeans(i)
#     kmeans.fit(pc)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,11),wcss)

# # %%
# import kneed
# kl=kneed.KneeLocator(range(1,11),wcss,curve="convex", direction="decreasing")
# kl.elbow

# # %%
# # validation
# from sklearn.metrics import silhouette_score
# sil_coeff=[]
# for i in range(2,11):
#     kmeans_sil=KMeans(i)
#     kmeans_sil.fit(pc)
#     score=silhouette_score(pc,kmeans_sil.labels_)
#     sil_coeff.append(score)
# plt.plot(range(2,11),sil_coeff)

# # %%
# from sklearn.pipeline import Pipeline
# pipe = Pipeline([
#     ('sc_p',StandardScaler()),
#     ('pca_p',PCA(2)),
#     ('kmeans_fin_p',KMeans(3))
# ])
# pipe.fit(dataset)
# u_labels=np.unique(pipe[-1].labels_)

# for i in u_labels:
#     plt.scatter(pc[pipe[-1].labels_==i,0],pc[pipe[-1].labels_==i,1],label=f'{i}')
# plt.scatter(pipe[-1].cluster_centers_[:,0],pipe[-1].cluster_centers_[:,1],color='black',
#             marker='o',s=300)
# plt.legend()
# plt.xlabel('pc_0: child_mort, income, inflation, life_expec, total_fer, gdpp')
# plt.ylabel('pc_1: exports, imports')

# # %%
# '''insights from above plot
# cluster 0: 
# cluster 1: 
# cluster 2: 
# '''

# # %%
# sns.countplot(pipe[-1].labels_)

# # %%
# # countries to be selected
# list(dataset_org.iloc[pipe[-1].labels_==0,0])

# # %%
# # for pipe.predict(), closest cluster is assigned to test point

# # %%



