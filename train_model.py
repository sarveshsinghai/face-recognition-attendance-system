import pickle
from sklearn.neighbors import KNeighborsClassifier

with open("data/faces_data.pkl", "rb") as f:
    faces = pickle.load(f)

with open("data/names.pkl", "rb") as f:
    labels = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(faces, labels)

with open("data/model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Model trained successfully.")