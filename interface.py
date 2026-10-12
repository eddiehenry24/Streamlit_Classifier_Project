import streamlit as st
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA


st.title("My Machine Learning Classifier App")

st.write("""
### Explore different classifiers
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine")
)

classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)


# Get the dataset
def get_dataset(name):
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)

st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(set(y)))


# Create the model parameter controls
def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        leaf_size = st.sidebar.slider("Leaf Size", 1, 100, 30)

        params["K"] = K
        params["leaf_size"] = leaf_size

    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        degree = st.sidebar.slider("Degree", 1, 10, 3)

        params["C"] = C
        params["degree"] = degree

    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimators = st.sidebar.slider(
            "Number of Estimators", 1, 100
        )
        min_samples_split = st.sidebar.slider(
            "Min Samples Split", 2, 20, 2
        )

        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["min_samples_split"] = min_samples_split

    return params


params = add_parameter_ui(classifier_name)


# Create the selected classifier
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(
            n_neighbors=params["K"],
            leaf_size=params["leaf_size"]
        )

    elif clf_name == "SVM":
        clf = SVC(
            C=params["C"],
            kernel="poly",
            degree=params["degree"]
        )

    else:
        clf = RandomForestClassifier(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            min_samples_split=params["min_samples_split"],
            random_state=42
        )

    return clf


clf = get_classifier(classifier_name, params)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train the classifier
clf.fit(X_train, y_train)


# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("Classifier accuracy:", accuracy)

st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))


# Create a PCA plot
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

st.write("PCA Plot:")

fig, ax = plt.subplots()
scatter = ax.scatter(x1, x2, c=y, cmap="viridis")
ax.legend(*scatter.legend_elements(), title="Classes")

st.pyplot(fig)
plt.close(fig)