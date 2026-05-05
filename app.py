import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import load_data, train_model, predict_segment, get_elbow_data

# --- Page Config ---
st.set_page_config(
    page_title="ClusterFlow | Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        border: 1px solid #4F46E5;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #E2E8F0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 ClusterFlow")
    st.markdown("### AI-Powered Segmentation")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=["csv"])
    
    if uploaded_file is None:
        st.info("💡 Using sample data: `customers.csv`")
        data_source = "customers.csv"
    else:
        data_source = uploaded_file

    df = load_data(data_source)
    
    st.divider()
    st.markdown("### Model Configuration")
    
    # Feature selection
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    features = st.multiselect(
        "Select Features for Clustering",
        options=numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )
    
    k_val = st.slider("Number of Clusters (K)", 2, 10, 3)
    
    st.divider()
    if st.button("🚀 Run Analysis"):
        st.session_state['run_analysis'] = True

# --- Main Content ---
st.title("🧠 Customer Intelligence Dashboard")
st.markdown("Extract meaningful patterns from your customer base using advanced K-Means clustering.")

if len(features) < 2:
    st.warning("⚠️ Please select at least two numeric features in the sidebar.")
    st.stop()

# Run training
df_segmented, model, scaler, cluster_labels = train_model(df, features, n_clusters=k_val)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Spatial Distribution")
    
    if len(features) >= 2:
        fig = px.scatter(
            df_segmented, 
            x=features[0], 
            y=features[1], 
            color='Segment',
            hover_data=df.columns,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Elbow Method (Optimal K)")
    ks, inertias = get_elbow_data(df, features)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=ks, y=inertias, mode='lines+markers', line=dict(color='#4F46E5')))
    fig_elbow.update_layout(
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Data Explorer
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔍 Segmented Explorer")
st.dataframe(df_segmented, use_container_width=True)
csv = df_segmented.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Segmented Data",
    data=csv,
    file_name='segmented_customers.csv',
    mime='text/csv',
)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
st.divider()
st.subheader("🔮 Intelligent Predictor")
st.markdown("Predict the segment for a new potential customer based on the current model.")

p_cols = st.columns(len(features))
input_values = []
for i, feat in enumerate(features):
    with p_cols[i]:
        val = st.number_input(f"Enter {feat}", value=float(df[feat].mean()))
        input_values.append(val)

if st.button("Predict Segment"):
    result = predict_segment(model, scaler, cluster_labels, input_values)
    st.balloons()
    st.success(f"### Result: This customer belongs to **{result}**")