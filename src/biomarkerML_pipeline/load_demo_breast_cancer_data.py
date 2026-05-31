
from sklearn.datasets import load_breast_cancer

def load_demo_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    return X, y


