import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import re

# Konfigurasi halaman
st.set_page_config(
    page_title="CSERA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling dengan warna yang lebih menarik dan menghilangkan anchor links
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
    
    /* Menghilangkan anchor links dari headers */
    .element-container h1 a,
    .element-container h2 a,
    .element-container h3 a,
    .element-container h4 a,
    .element-container h5 a,
    .element-container h6 a {
        display: none !important;
    }
    
    /* Mencegah pointer cursor pada headers */
    .element-container h1,
    .element-container h2,
    .element-container h3,
    .element-container h4,
    .element-container h5,
    .element-container h6 {
        cursor: default !important;
    }
    
    /* Custom header styles tanpa link */
    .custom-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2E86AB;
        margin: 1rem 0;
        padding: 0;
        cursor: default !important;
    }
    
    .custom-subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
        padding: 0;
        cursor: default !important;
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
# def create_cluster_distribution(df):
#     """Membuat chart distribusi cluster per RS"""
#     cluster_counts = df.groupby(['location', 'cluster']).size().reset_index(name='count')
    
#     fig = px.bar(
#         cluster_counts,
#         x='location',
#         y='count',
#         color='cluster',
#         labels={'location': 'Rumah Sakit', 'count': 'Jumlah Ulasan', 'cluster': 'Cluster'},
#         color_discrete_sequence=[
#             '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0'
#         ]
#     )

#     fig.update_layout(
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         font=dict(color='#FF5722')
#     )
#     return fig


def create_cluster_distribution(df):
    """Membuat chart distribusi cluster per RS dengan warna yang berbeda untuk setiap cluster"""
    cluster_counts = df.groupby(['location', 'cluster']).size().reset_index(name='count')
    
    # Convert cluster to string to ensure proper color mapping
    cluster_counts['cluster'] = cluster_counts['cluster'].astype(str)
    
    fig = px.bar(
        cluster_counts,
        x='location',
        y='count',
        color='cluster',
        labels={'location': 'Rumah Sakit', 'count': 'Jumlah Ulasan', 'cluster': 'Cluster'},
        color_discrete_sequence=['#e74c3c', '#2ecc71', '#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#e91e63']
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        xaxis_title="Rumah Sakit",
        yaxis_title="Jumlah Ulasan",  
        legend_title="Cluster",
        showlegend=True,
        xaxis_tickangle=45,
        margin=dict(r=100)
    )
    
    return fig

# Load data
df = load_data()

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar untuk navigasi
# st.sidebar.markdown("""
# <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); border-radius: 10px; margin-bottom: 1rem;">
#     <h2 style="color: white; margin: 0;">🏥 Analisis Rumah Sakit</h2>
# </div>
# """, unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
        <img src="data:image/png;base64,{base64.b64encode(open('logo csera.png', 'rb').read()).decode()}" style="max-width: 100%;">
    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown('<div class="custom-subheader">📋 Menu Navigasi</div>', unsafe_allow_html=True)

# Navigation buttons
if st.sidebar.button("🏠 Home", help="Halaman utama dashboard"):
    st.session_state.current_page = "Home"

if st.sidebar.button("🎯 Clustering", help="Analisis clustering data"):
    st.session_state.current_page = "Clustering"

if st.sidebar.button("😊 Analisis Sentimen", help="Analisis sentimen ulasan"):
    st.session_state.current_page = "Sentiment"

# Get current page
current_page = st.session_state.current_page

# Menu Home
if current_page == "Home":
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
        st.markdown('<div class="custom-subheader">🏥 Rating Rumah Sakit</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="custom-subheader">📊 Distribusi Rating</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="custom-header">Ulasan per Rumah Sakit</div>', unsafe_allow_html=True)
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
elif current_page == "Clustering":
    st.markdown('<h1 class="main-header">🎯 Clustering </h1>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="custom-subheader">🔍 Filter Clustering</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="custom-header">📈 Visualisasi Clustering</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="custom-subheader">📊 Hasil Distribusi Cluster Keseluruhan</div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="custom-subheader">🏥 Distribusi Cluster per Rumah Sakit</div>', unsafe_allow_html=True)
        
        if len(filtered_df_cluster) > 0:
            fig = create_cluster_distribution(filtered_df_cluster)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Cluster Analysis Table
    st.markdown('<div class="custom-header">📋 Detail Cluster per Rumah Sakit</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="custom-header">🔍 Analisis dan Insight Clustering</div>', unsafe_allow_html=True)
    
    # Generate insights per hospital
    st.markdown('<div class="custom-subheader">🏥 Analisis Clustering per Rumah Sakit:</div>', unsafe_allow_html=True)
    
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
        
        st.markdown("---")
    
    # Cluster Analysis per Selected Cluster
    st.markdown('<div class="custom-subheader">🔍 Analisis Detail per Cluster</div>', unsafe_allow_html=True)
    
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
elif current_page == "Sentiment":
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
        st.markdown('<div class="custom-subheader">📊 Distribusi Sentimen</div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="custom-subheader">🏥 Distribusi Sentimen per Rumah Sakit</div>', unsafe_allow_html=True)
        
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
    st.markdown('<div class="custom-subheader">🔍 Detail Analisis Sentimen</div>', unsafe_allow_html=True)
    
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
    
    # Top 10 Keywords Table
    st.markdown('<div class="custom-subheader">🔤 Analisis Sentimen per Rumah Sakit</div>', unsafe_allow_html=True)
    
    # Mapping file names to hospital names
    hospital_files = {
        'RST': 'Top10KataRST.csv',
        'RSV': 'Top10KataRSV.csv', 
        'RSW': 'Top10KataRSW.csv',
        'RSX': 'Top10KataRSX.csv',
        'RSY': 'Top10KataRSY.csv',
        'RSZ': 'Top10KataRSZ.csv'
    }
    
    # Debug: Check which files exist
    import os
    # Create tabs for each hospital
    tabs = st.tabs(list(hospital_files.keys()))

    def create_full_html_table(df):
        """Create HTML table without scroll - full height"""
        html = """
        <style>
        .full-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 14px;
            margin: 20px 0;
        }
        .full-table th, .full-table td {
            border: 1px solid #ddd;
            padding: 12px 8px;
            text-align: left;
            vertical-align: top;
            word-wrap: break-word;
            word-break: break-word;
            white-space: normal;
        }
        .full-table th {
            font-weight: bold;
            text-align: center;
        }
        .cluster-col { 
            width: 10%; 
            text-align: center;
            font-weight: bold;
        }
        .kata-col { 
            width: 20%; 
            font-weight: 500;
        }
        .analisis-col { 
            width: 50%; 
            line-height: 1.5;
        }
        .sentimen-col { 
            width: 20%; 
            text-align: center;
            font-weight: bold;
        }
        </style>
        
        <table class="full-table">
        <thead>
            <tr>
        """
        
        # Add headers
        col_classes = ['cluster-col', 'kata-col', 'analisis-col', 'sentimen-col']
        for i, col in enumerate(df.columns):
            class_name = col_classes[i] if i < len(col_classes) else ''
            html += f'<th class="{class_name}">{col}</th>'
        
        html += "</tr></thead><tbody>"
        
        # Add rows
        for _, row in df.iterrows():
            html += "<tr>"
            for i, value in enumerate(row):
                class_name = col_classes[i] if i < len(col_classes) else ''
                # Handle NaN values
                display_value = str(value) if pd.notna(value) else ""
                
                # Add special styling for sentiment column
                if i == 3 and 'positif' in display_value.lower():
                    class_name += ' sentimen-positif'
                elif i == 3 and 'negatif' in display_value.lower():
                    class_name += ' sentimen-negatif'
                
                html += f'<td class="{class_name}">{display_value}</td>'
            html += "</tr>"
        
        html += "</tbody></table>"
        return html

    for i, (hospital_code, filename) in enumerate(hospital_files.items()):
        with tabs[i]:
            try:
                # Check if file exists
                if not os.path.exists(filename):
                    st.error(f"File {filename} tidak ditemukan di direktori saat ini.")
                    st.info("Pastikan file CSV berada di direktori yang sama dengan script Python Anda.")
                    continue
                
                # Read the CSV file with semicolon separator
                top_words_df = pd.read_csv(filename, sep=';', encoding='utf-8')
                
                st.subheader(f"Data untuk {hospital_code}")
                
                
                # Display as full HTML table
                html_table = create_full_html_table(top_words_df)
                st.markdown(html_table, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error membaca file {filename}: {str(e)}")
                continue
    
    # Sample Reviews by Sentiment
    st.markdown('<div class="custom-subheader">📝 Contoh Ulasan Berdasarkan Sentimen</div>', unsafe_allow_html=True)
    
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