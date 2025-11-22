from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
x,y=make_blobs(n_samples=500,centers=4,random_state=42)
kmeans=KMeans(n_clusters=4,random_state=42)
kmeans.fit(x,y) 
predected_labels=kmeans.predict(x)
plt.scatter(x[:,0],x[:,1],c=predected_labels)
plt.title("KMeans cluster")
plt.show()