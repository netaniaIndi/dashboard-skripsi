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

# Custom CSS untuk styling dengan warna yang lebih menarik
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
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.4);
    }
    
    .hospital-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2c3e50;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .hospital-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(168, 237, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sentiment-positive {
        color: #00b894;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0, 184, 148, 0.2);
    }
    
    .sentiment-negative {
        color: #e17055;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(225, 112, 85, 0.2);
    }
    
    .sentiment-neutral {
        color: #fdcb6e;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(253, 203, 110, 0.2);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
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

# Fungsi untuk membuat visualisasi clustering
def create_cluster_distribution(df):
    """Membuat chart distribusi cluster per RS"""
    cluster_counts = df.groupby(['location', 'cluster']).size().reset_index(name='count')
    
    fig = px.bar(
        cluster_counts,
        x='location',
        y='count',
        color='cluster',
        labels={'location': 'Rumah Sakit', 'count': 'Jumlah Ulasan', 'cluster': 'Cluster'},
        color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

def create_cluster_sentiment_heatmap(df):
    """Membuat heatmap cluster vs sentimen"""
    cluster_sentiment = pd.crosstab(df['cluster'], df['predicted_sentiment'])
    
    fig = px.imshow(
        cluster_sentiment.values,
        x=cluster_sentiment.columns,
        y=cluster_sentiment.index,
        title="Heatmap: Cluster vs Sentimen",
        labels={'x': 'Sentimen', 'y': 'Cluster', 'color': 'Jumlah'},
        color_continuous_scale='Sunset'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

def create_cluster_rating_correlation(df):
    """Membuat box plot korelasi cluster dan rating"""
    fig = px.box(
        df,
        x='cluster',
        y='rating',
        title="Korelasi Rating dan Cluster",
        labels={'cluster': 'Cluster', 'rating': 'Rating'},
        color='cluster',
        color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

# Load data
df = load_data()

# Sidebar untuk navigasi
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">🏥 Analisis Rumah Sakit</h2>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "📋 Pilih Menu",
    ["🏠 Home", "🎯 Clustering", "😊 Analisis Sentimen"]
)

# Menu Home
if menu == "🏠 Home":
    st.markdown('<h1 class="main-header">🏥 Dashboard Analisis Rumah Sakit</h1>', unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);">
            <h3>Total Ulasan</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_hospitals = df['location'].nunique()
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%);">
            <h3>Rumah Sakit</h3>
            <h2>{total_hospitals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_locations = df['location'].nunique()
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);">
            <h3>Lokasi</h3>
            <h2>{total_locations}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hospital Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Rating Rumah Sakit")
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
                <small>{review_count} ulasan</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Distribusi Rating")
        rating_dist = df['rating'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_dist.index,
            y=rating_dist.values,
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_dist.values,
            color_continuous_scale='Sunset'
        )
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Jumlah Ulasan",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Charts
    st.markdown('<h3 class="main-header">Ulasan per Rumah Sakit</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        hospital_counts = df['location'].value_counts()
        
        fig = px.pie(
            values=hospital_counts.values,
            names=hospital_counts.index,
            color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#ff9ff3', '#54a0ff']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        location_counts = df['location'].value_counts()
        
        fig = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            color=location_counts.values,
            color_continuous_scale='Turbo'
        )
        fig.update_layout(
            xaxis_title="Jumlah Ulasan",
            yaxis_title="Rumah Sakit",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)

# Menu Clustering
elif menu == "🎯 Clustering":
    st.markdown('<h1 class="main-header">🎯 Clustering </h1>', unsafe_allow_html=True)
    
    st.sidebar.header("🔍 Filter Clustering")    
    selected_hospitals_cluster = st.sidebar.multiselect(
        "Pilih Rumah Sakit:",
        options=df['location'].unique(),
        default=df['location'].unique()
    )
    
    # Filter data untuk clustering
    filtered_df_cluster = df[df['location'].isin(selected_hospitals_cluster)]
    
    # Cluster Overview Metrics
    col1, col2 = st.columns(2)
    
    cluster_counts = filtered_df_cluster['cluster'].value_counts().sort_index()
    total_clusters = len(cluster_counts)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);">
            <h3>Total Review</h3>
            <h2>{len(filtered_df_cluster):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);">
            <h3>Total Cluster</h3>
            <h2>{total_clusters}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Clustering Visualizations
    st.header("📈 Visualisasi Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Hasil Distribusi Cluster Keseluruhan")
        
        cluster_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
        
        fig = px.pie(
            values=cluster_counts.values,
            names=[cluster_names.get(i, f'Cluster {i}') for i in cluster_counts.index],
            color_discrete_sequence=['#00b894', '#fdcb6e', '#e17055', '#74b9ff', '#a29bfe', '#fd79a8']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Distribusi Cluster per Rumah Sakit")
        
        if len(filtered_df_cluster) > 0:
            fig = create_cluster_distribution(filtered_df_cluster)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Cluster Analysis Table
    st.header("📋 Detail Cluster per Rumah Sakit")

    if 'predicted_sentiment' in filtered_df_cluster.columns:
        cluster_summary = filtered_df_cluster.groupby(['location', 'cluster']).agg({
            'predicted_sentiment': ['count']
        }).round(2)
        cluster_summary.columns = ['Jumlah Ulasan']
        cluster_summary = cluster_summary.reset_index()
        st.dataframe(cluster_summary, use_container_width=True)
    else:
        cluster_summary = filtered_df_cluster.groupby(['location', 'cluster']).agg({
            'rating': ['count']
        }).round(2)
        cluster_summary.columns = ['Jumlah Ulasan']
        cluster_summary = cluster_summary.reset_index()
        st.dataframe(cluster_summary, use_container_width=True)
    
    # Detailed Analysis and Insights
    st.header("🔍 Analisis dan Insight Clustering")
    
    # Generate insights per hospital
    st.markdown("### 🏥 **Analisis Clustering per Rumah Sakit:**")
    
    for hospital in sorted(filtered_df_cluster['location'].unique()):
        hospital_data = filtered_df_cluster[filtered_df_cluster['location'] == hospital]
        
        # Statistik clustering
        total_reviews = len(hospital_data)
        avg_rating = hospital_data['rating'].mean()
        cluster_counts_hospital = hospital_data['cluster'].value_counts().sort_index()
        dominant_cluster = hospital_data['cluster'].mode().iloc[0] if len(hospital_data) > 0 else 'N/A'
        num_clusters = hospital_data['cluster'].nunique()
        
        # Rating per cluster
        rating_per_cluster = hospital_data.groupby('cluster')['rating'].mean().round(2)
        
        st.markdown(f"""
        **{hospital}:**
        - 📊 **Total Ulasan**: {total_reviews} ulasan
        - ⭐ **Rating Rata-rata**: {avg_rating:.2f}/5.0
        - 🔄 **Jumlah Cluster**: {num_clusters} cluster
        - 📈 **Distribusi Cluster**: {dict(cluster_counts_hospital)}
        """)
        
        # Ringkasan Analisis Per Cluster
        st.subheader("Hasil Analisis Tiap Cluster")
        hospital_cluster_stats = filtered_df_cluster.groupby('location').agg({
            'cluster': 'nunique',
            'rating': 'mean'
        }).round(2)
        hospital_cluster_stats.columns = ['Jumlah Cluster', 'Rating Rata-rata']
        st.dataframe(hospital_cluster_stats, use_container_width=True)
        
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cluster Analysis per Selected Cluster
    st.subheader("🔍 Analisis Detail per Cluster")
    
    selected_cluster = st.selectbox(
        "Pilih cluster untuk analisis detail:",
        options=sorted(filtered_df_cluster['cluster'].unique()),
        format_func=lambda x: f"Cluster {x}"
    )
    
    cluster_data = filtered_df_cluster[filtered_df_cluster['cluster'] == selected_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Statistik Cluster {selected_cluster}:**")
        st.write(f"- Total reviews: {len(cluster_data)}")
        st.write(f"- Rating range: {cluster_data['rating'].min():.1f} - {cluster_data['rating'].max():.1f}")
        if 'predicted_sentiment' in cluster_data.columns:
            sentiment_dist = cluster_data['predicted_sentiment'].value_counts()
            for sentiment, count in sentiment_dist.items():
                st.write(f"- {sentiment}: {count} ({count/len(cluster_data)*100:.1f}%)")
        
        # Hospital distribution in cluster
        hospital_dist = cluster_data['location'].value_counts()
        st.write("**Distribusi per Rumah Sakit:**")
        for hospital, count in hospital_dist.items():
            st.write(f"- {hospital}: {count} ulasan")
    
    with col2:
        st.markdown("**Sample Reviews:**")
        sample_reviews = cluster_data['review'].sample(min(3, len(cluster_data)))
        for i, review in enumerate(sample_reviews, 1):
            st.write(f"{i}. {review}")

# Menu Sentiment Analysis
elif menu == "😊 Analisis Sentimen":
    st.markdown('<h1 class="main-header">😊 Analisis Sentimen</h1>', unsafe_allow_html=True)
  
    # Sentiment Overview
    sentiment_counts = df['predicted_sentiment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        positive_count = sentiment_counts.get('positif', 0)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%);">
            <h3>😊 Positif</h3>
            <h2>{positive_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        negative_count = sentiment_counts.get('negatif', 0)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #e17055 0%, #d63031 100%);">
            <h3>😞 Negatif</h3>
            <h2>{negative_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Sentimen")
        
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={
                'positif': '#00b894',
                'negatif': '#e17055'
            }
        )
        fig.update_layout(
            xaxis_title="Sentimen",
            yaxis_title="Jumlah Data",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Distribusi Sentimen per Rumah Sakit")
        
        sentiment_hospital = pd.crosstab(df['location'], df['predicted_sentiment'])
        
        fig = px.bar(
            sentiment_hospital,
            color_discrete_map={
                'positif': '#00b894',
                'negatif': '#e17055'
            }
        )
        fig.update_layout(
            xaxis_title="Rumah Sakit",
            yaxis_title="Jumlah Data",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Sentiment Analysis
    st.subheader("🔍 Detail Analisis Sentimen")
    
    selected_hospital = st.selectbox("Pilih rumah sakit:", ['Semua'] + list(df['location'].unique()))
    
    if selected_hospital != 'Semua':
        filtered_df = df[df['location'] == selected_hospital]
    else:
        filtered_df = df
    
    st.markdown("**Statistik Sentimen:**")
    sentiment_stats = filtered_df['predicted_sentiment'].value_counts()
    for sentiment, count in sentiment_stats.items():
        percentage = count/len(filtered_df)*100
        if sentiment == 'positif':
            st.markdown(f'<p class="sentiment-positive">😊 {sentiment}: {count} ({percentage:.1f}%)</p>', unsafe_allow_html=True)
        elif sentiment == 'negatif':
            st.markdown(f'<p class="sentiment-negative">😞 {sentiment}: {count} ({percentage:.1f}%)</p>', unsafe_allow_html=True)

    # Sample Reviews by Sentiment
    st.subheader("📝 Contoh Ulasan Berdasarkan Sentimen")
    
    selected_sentiment = st.selectbox(
        "Pilih sentiment untuk melihat contoh review:",
        ['positif', 'negatif']
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
    <p>🏥 Dashboard Analisis RS | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)