import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import re

# Konfigurasi halaman
st.set_page_config(
    page_title="Hospital Review Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .hospital-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .preprocessing-step {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HasilSentimenAllRS.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

# Sidebar untuk navigasi
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">🏥 Hospital Analytics</h2>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "📋 Pilih Menu",
    ["🏠 Home", "🔧 Preprocessing", "🎯 Clustering", "😊 Sentiment Analysis"]
)

# Menu Home
if menu == "🏠 Home":
    st.markdown('<h1 class="main-header">🏥 Hospital Review Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Reviews</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = df['rating'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Rating</h3>
            <h2>{avg_rating:.1f} ⭐</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_hospitals = df['location'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Hospitals</h3>
            <h2>{total_hospitals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_locations = df['location'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Locations</h3>
            <h2>{total_locations}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hospital Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Hospital Overview")
        hospital_stats = df.groupby('location').agg({
            'rating': 'mean',
            'review': 'count'
        }).round(2)
        
        for hospital in hospital_stats.index:
            avg_rating = hospital_stats.loc[hospital, 'rating']
            review_count = hospital_stats.loc[hospital, 'review']
            stars = "⭐" * int(avg_rating)
            
            st.markdown(f"""
            <div class="hospital-card">
                {hospital} - {avg_rating}/5 {stars}<br>
                <small>{review_count} reviews</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Rating Distribution")
        rating_dist = df['rating'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_dist.index,
            y=rating_dist.values,
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_dist.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            title="Distribution of Ratings",
            xaxis_title="Rating",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Reviews by Hospital")
        hospital_counts = df['location'].value_counts()
        
        fig = px.pie(
            values=hospital_counts.values,
            names=hospital_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Review Distribution by Hospital")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📍 Reviews by Location")
        location_counts = df['location'].value_counts()
        
        fig = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            color=location_counts.values,
            color_continuous_scale='blues'
        )
        fig.update_layout(
            title="Review Distribution by Location",
            xaxis_title="Number of Reviews",
            yaxis_title="Location"
        )
        st.plotly_chart(fig, use_container_width=True)

# Menu Preprocessing
elif menu == "🔧 Preprocessing":
    st.markdown('<h1 class="main-header">🔧 Data Preprocessing</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Proses preprocessing dilakukan untuk membersihkan dan mempersiapkan data text review 
    agar dapat dianalisis dengan lebih baik. Berikut adalah tahapan preprocessing yang dilakukan:
    """)
    
    # Preprocessing Steps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="preprocessing-step">
            <h4>1. 🧹 Data Cleaning</h4>
            <p>Menghilangkan karakter khusus, tanda baca, dan normalize text</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="preprocessing-step">
            <h4>2. 🔤 Case Folding</h4>
            <p>Mengubah semua huruf menjadi lowercase untuk konsistensi</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="preprocessing-step">
            <h4>3. 💬 Slang Words Normalization</h4>
            <p>Mengubah kata-kata slang menjadi kata formal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="preprocessing-step">
            <h4>4. ✂ Tokenizing</h4>
            <p>Memecah text menjadi token/kata-kata individual</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="preprocessing-step">
            <h4>5. 🚫 Stop Words Removal</h4>
            <p>Menghilangkan kata-kata yang tidak bermakna (dan, atau, yang, dll)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="preprocessing-step">
            <h4>6. 🌱 Stemming</h4>
            <p>Mengubah kata ke bentuk dasarnya (running → run)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample preprocessing comparison
    st.subheader("📝 Contoh Hasil Preprocessing")
    
    sample_idx = st.selectbox("Pilih sampel review:", range(min(10, len(df))))
    
    if sample_idx < len(df):
        sample = df.iloc[sample_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Original Review:*")
            st.info(sample['review'])
            
            st.markdown("*After Cleaning:*")
            st.success(sample['clean_data'])
            
            st.markdown("*After Case Folding:*")
            st.success(sample['casefold_data'])
        
        with col2:
            st.markdown("*After Tokenizing:*")
            st.success(str(sample['tokenizing_data']))
            
            st.markdown("*After Stop Words Removal:*")
            st.success(sample['stopwords_data'])
            
            st.markdown("*After Stemming:*")
            st.success(sample['stemming_data'])
    
    # Preprocessing Statistics
    st.subheader("📊 Statistik Preprocessing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_words_before = df['review'].str.split().str.len().mean()
        st.metric("Avg Words (Original)", f"{avg_words_before:.1f}")
    
    with col2:
        avg_words_after = df['stopwords_data'].str.split().str.len().mean()
        st.metric("Avg Words (After Preprocessing)", f"{avg_words_after:.1f}")
    
    with col3:
        reduction = ((avg_words_before - avg_words_after) / avg_words_before) * 100
        st.metric("Word Reduction", f"{reduction:.1f}%")

# Menu Clustering
elif menu == "🎯 Clustering":
    st.markdown('<h1 class="main-header">🎯 Clustering Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Clustering digunakan untuk mengelompokkan review berdasarkan kesamaan karakteristik. 
    Kami menggunakan algoritma clustering untuk mengidentifikasi pola dalam data review.
    """)
    
    # Cluster Overview
    col1, col2, col3 = st.columns(3)
    
    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_names = {0: 'Positive Reviews', 1: 'Neutral Reviews', 2: 'Negative Reviews'}
    
    for i, (cluster, count) in enumerate(cluster_counts.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cluster {cluster}</h3>
                <h4>{cluster_names[cluster]}</h4>
                <h2>{count}</h2>
                <p>{count/len(df)*100:.1f}% of total</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Clustering Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Distribution")
        
        fig = px.pie(
            values=cluster_counts.values,
            names=[cluster_names[i] for i in cluster_counts.index],
            color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Clusters by Hospital")
        
        cluster_hospital = pd.crosstab(df['location'], df['cluster'])
        
        fig = px.bar(
            cluster_hospital,
            color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c']
        )
        fig.update_layout(
            title="Cluster Distribution by Hospital",
            xaxis_title="Hospital",
            yaxis_title="Number of Reviews"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Analysis
    st.subheader("🔍 Cluster Analysis")
    
    selected_cluster = st.selectbox(
        "Pilih cluster untuk analisis detail:",
        options=list(cluster_names.keys()),
        format_func=lambda x: f"Cluster {x}: {cluster_names[x]}"
    )
    
    cluster_data = df[df['cluster'] == selected_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"*Statistik Cluster {selected_cluster}:*")
        st.write(f"- Total reviews: {len(cluster_data)}")
        st.write(f"- Average rating: {cluster_data['rating'].mean():.2f}")
        st.write(f"- Most common hospital: {cluster_data['location'].mode().iloc[0]}")
        st.write(f"- Most common location: {cluster_data['location'].mode().iloc[0]}")
    
    with col2:
        st.markdown("*Sample Reviews:*")
        sample_reviews = cluster_data['review'].sample(min(3, len(cluster_data)))
        for i, review in enumerate(sample_reviews, 1):
            st.write(f"{i}. {review}")

# Menu Sentiment Analysis
elif menu == "😊 Sentiment Analysis":
    st.markdown('<h1 class="main-header">😊 Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Analisis sentimen untuk memahami emosi dan opini dalam review rumah sakit. 
    Sentimen diklasifikasikan menjadi Positive, Neutral, dan Negative.
    """)
    
    # Sentiment Overview
    sentiment_counts = df['predicted_sentiment'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_count = sentiment_counts.get('Positive', 0)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);">
            <h3>😊 Positive</h3>
            <h2>{positive_count}</h2>
            <p>{positive_count/len(df)*100:.1f}% of reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        neutral_count = sentiment_counts.get('Neutral', 0)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);">
            <h3>😐 Neutral</h3>
            <h2>{neutral_count}</h2>
            <p>{neutral_count/len(df)*100:.1f}% of reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        negative_count = sentiment_counts.get('Negative', 0)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
            <h3>😞 Negative</h3>
            <h2>{negative_count}</h2>
            <p>{negative_count/len(df)*100:.1f}% of reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f39c12', 
                'Negative': '#e74c3c'
            }
        )
        fig.update_layout(
            title="Overall Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Sentiment by Hospital")
        
        sentiment_hospital = pd.crosstab(df['location'], df['predicted_sentiment'])
        
        fig = px.bar(
            sentiment_hospital,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f39c12',
                'Negative': '#e74c3c'
            }
        )
        fig.update_layout(
            title="Sentiment Distribution by Hospital",
            xaxis_title="Hospital",
            yaxis_title="Number of Reviews"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment vs Rating Analysis
    st.subheader("⭐ Sentiment vs Rating Analysis")
    
    sentiment_rating = df.groupby(['predicted_sentiment', 'rating']).size().unstack(fill_value=0)
    
    fig = px.imshow(
        sentiment_rating.values,
        labels=dict(x="Rating", y="Sentiment", color="Count"),
        x=[str(i) for i in sentiment_rating.columns],
        y=sentiment_rating.index,
        color_continuous_scale='YlOrRd'
    )
    fig.update_layout(title="Heatmap: Sentiment vs Rating")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Sentiment Analysis
    st.subheader("🔍 Detailed Sentiment Analysis")
    
    selected_hospital = st.selectbox("Pilih rumah sakit:", ['All'] + list(df['location'].unique()))
    
    if selected_hospital != 'All':
        filtered_df = df[df['location'] == selected_hospital]
    else:
        filtered_df = df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*Sentiment Statistics:*")
        sentiment_stats = filtered_df['predicted_sentiment'].value_counts()
        for sentiment, count in sentiment_stats.items():
            percentage = count/len(filtered_df)*100
            if sentiment == 'Positive':
                st.markdown(f'<p class="sentiment-positive">😊 {sentiment}: {count} ({percentage:.1f}%)</p>', unsafe_allow_html=True)
            elif sentiment == 'Negative':
                st.markdown(f'<p class="sentiment-negative">😞 {sentiment}: {count} ({percentage:.1f}%)</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="sentiment-neutral">😐 {sentiment}: {count} ({percentage:.1f}%)</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("*Average Rating by Sentiment:*")
        avg_rating_sentiment = filtered_df.groupby('predicted_sentiment')['rating'].mean()
        for sentiment, avg_rating in avg_rating_sentiment.items():
            stars = "⭐" * int(avg_rating)
            st.write(f"{sentiment}: {avg_rating:.2f} {stars}")
    
    # Sample Reviews by Sentiment
    st.subheader("📝 Sample Reviews by Sentiment")
    
    selected_sentiment = st.selectbox(
        "Pilih sentiment untuk melihat contoh review:",
        ['Positive', 'Neutral', 'Negative']
    )
    
    sentiment_reviews = filtered_df[filtered_df['predicted_sentiment'] == selected_sentiment]
    if len(sentiment_reviews) > 0:
        sample_reviews = sentiment_reviews['review'].sample(min(5, len(sentiment_reviews)))
        for i, review in enumerate(sample_reviews, 1):
            st.write(f"{i}. {review}")
    else:
        st.info(f"Tidak ada review dengan sentiment {selected_sentiment} untuk filter yang dipilih.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏥 Hospital Review Analytics Dashboard | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)