import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(path_or_file):
    """Loads data from a path or a file-like object (for Streamlit uploader)."""
    return pd.read_csv(path_or_file)

def train_model(df, features, n_clusters=3):
    """Trains a K-Means model on specified features."""
    X = df[features]
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Simple naming logic based on feature means
    cluster_labels = {i: f"Segment {i+1}" for i in range(n_clusters)}
    df['Segment'] = df['Cluster'].map(cluster_labels)
    
    return df, kmeans, scaler, cluster_labels

def get_elbow_data(df, features, max_k=10):
    """Computes inertia for different values of K for the Elbow Method."""
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    ks = list(range(1, max_k + 1))
    
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        
    return ks, inertias

def predict_segment(model, scaler, cluster_labels, input_data):
    """Predicts the segment for a new data point."""
    # input_data should be a list/array matching the features used for training
    data_scaled = scaler.transform([input_data])
    cluster = model.predict(data_scaled)[0]
    
    return cluster_labels[cluster]