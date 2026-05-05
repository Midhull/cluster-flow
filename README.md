# 🧠 ClusterFlow: AI Customer Intelligence

**ClusterFlow** is a professional-grade customer segmentation tool built with Python, Scikit-Learn, and Streamlit. It enables businesses to discover hidden patterns in their customer base using unsupervised machine learning (K-Means Clustering).

![Dashboard](https://raw.githubusercontent.com/Midhull/cluster-flow/main/dashboard.png)

## 🚀 Key Features

- **Dynamic Data Upload**: Support for custom CSV datasets.
- **Intelligent Feature Selection**: Choose any numeric columns for clustering.
- **Interactive Visualization**: High-fidelity, zoomable charts powered by Plotly.
- **Optimal K Detection**: Integrated Elbow Method to find the most meaningful number of clusters.
- **Segment Predictor**: Real-time classification for new customer entries.
- **Data Export**: Download processed datasets with assigned segments for marketing campaigns.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (with Custom CSS for Glassmorphism)
- **Analytics**: Scikit-Learn, Pandas, NumPy
- **Visuals**: Plotly Express, Plotly Graph Objects

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd kmns
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

## 📊 How to Use

1. **Upload**: Drop your customer CSV into the sidebar uploader.
2. **Configure**: Select the features you want to analyze (e.g., *Annual Income*, *Spending Score*).
3. **Analyze**: Use the **Elbow Method** chart to determine the best value for **K**.
4. **Interact**: Explore the clusters in the interactive scatter plot.
5. **Predict**: Use the prediction section to classify new potential customers.
6. **Export**: Click "Download Segmented Data" to use the results in your CRM or marketing tools.

## 🚀 Live Dashboard
👉 [Open ClusterFlow App](https://cluster-flowgit-ngvtwthrjixov7cb8shkpg.streamlit.app/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
