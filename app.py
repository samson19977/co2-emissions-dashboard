import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# Configure Streamlit page with dramatic new title
st.set_page_config(
    page_title="🌍 AFRICA'S CLIMATE & POPULATION REVEALED: The Future in Data",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with new color scheme
st.markdown("""
<style>
    :root {
        --primary: #3498db;
        --danger: #e74c3c;
        --success: #2ecc71;
        --warning: #f39c12;
        --population: #9b59b6;
        --climate: #1abc9c;
    }
    
    .header-container {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .header-container h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #fff 0%, #1abc9c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        margin-bottom: 1.5rem;
        border-top: 5px solid var(--primary);
        transition: all 0.4s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.2);
    }
    
    .population-card {
        border-top: 5px solid var(--population);
    }
    
    .climate-card {
        border-top: 5px solid var(--climate);
    }
    
    .alert-card {
        background: #fff8f8;
        border-left: 5px solid var(--danger);
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9f5ff 100%);
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 2rem;
        border-left: 5px solid var(--success);
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }
    
    .threshold-indicator {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .threshold-safe { background-color: var(--success); }
    .threshold-warning { background-color: var(--warning); }
    .threshold-danger { background-color: var(--danger); }
    
    .population-badge {
        background-color: var(--population);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 8px;
    }
    
    .climate-badge {
        background-color: var(--climate);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 8px;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px !important;
        padding: 8px 12px !important;
    }
    
    .stSlider div[data-testid="stTickBar"] {
        margin-top: 15px;
    }
    
    .stSlider div[data-testid="stTickBar"] div {
        background: var(--primary) !important;
    }
    
    .stSlider div[role="slider"] {
        background: var(--primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data with enhanced caching and population metrics
@st.cache_data
def load_data():
    try:
        url = "https://huggingface.co/spaces/NSamson1/Early-Warning-Airquality/raw/main/co2_Emission_Africa.csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        
        # Define sector columns and convert to numeric
        sector_cols = ['Transportation (Mt)', 'Manufacturing/Construction (Mt)',
                      'Land-Use Change and Forestry (Mt)', 'Industrial Processes (Mt)',
                      'Energy (Mt)', 'Electricity/Heat (Mt)']
        
        for col in sector_cols + ['Total CO2 Emission including LUCF (Mt)', 'GDP PER CAPITA (USD)', 'Population']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Advanced imputation by sub-region median
            df[col] = df.groupby(['Sub-Region', 'Year'])[col].transform(
                lambda x: x.fillna(x.median()))
        
        # Calculate derived metrics
        df['Emissions per Capita'] = df['Total CO2 Emission including LUCF (Mt)'] / (df['Population'] / 1e6)
        df['Emissions Intensity'] = df['Total CO2 Emission including LUCF (Mt)'] / df['GDP PER CAPITA (USD)']
        df['Population Growth Rate'] = df.groupby('Country')['Population'].pct_change() * 100
        
        # Calculate population density (assuming area remains constant)
        df['Population Density'] = df.groupby('Country')['Population'].transform(
            lambda x: x / x.max() * 100)  # Normalized density
        
        return df.dropna(subset=['Country', 'Year'] + sector_cols)
    
    except Exception as e:
        st.error(f"🚨 Data loading failed: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

# Define sector thresholds (hypothetical values for demonstration)
SECTOR_THRESHOLDS = {
    'Transportation (Mt)': {'warning': 10, 'danger': 20},
    'Manufacturing/Construction (Mt)': {'warning': 15, 'danger': 30},
    'Energy (Mt)': {'warning': 25, 'danger': 50},
    'Electricity/Heat (Mt)': {'warning': 10, 'danger': 20}
}

# =============================================
# Header Section with dramatic new title
# =============================================
st.markdown("""
<div class="header-container">
    <h1>AFRICA'S CLIMATE & POPULATION REVEALED</h1>
    <p style="font-size:1.2rem; color:rgba(255,255,255,0.9);">The hidden connections between demographic change and climate impact</p>
    <div style="margin-top:1rem;">
        <span class="population-badge">POPULATION DYNAMICS</span>
        <span class="climate-badge">CLIMATE IMPACT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================
# Continent-Wide Insights - Now with Population
# =============================================
st.header("🌍 Continental Overview: People & Planet")

# Key metrics row - now with population metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_2020 = df[df['Year'] == 2020]['Total CO2 Emission including LUCF (Mt)'].sum()
    st.metric("Total 2020 Emissions", f"{total_2020:,.0f} Mt", 
              help="Sum across all African nations")
with col2:
    pop_2020 = df[df['Year'] == 2020]['Population'].sum() / 1e6
    st.metric("2020 Population", f"{pop_2020:,.1f} Million", 
              delta="+2.5% annual growth", 
              help="Total population across Africa")
with col3:
    avg_growth = df.groupby('Country')['Total CO2 Emission including LUCF (Mt)'].apply(
        lambda x: x.pct_change().mean()
    ).mean() * 100
    st.metric("Avg Emissions Growth", f"{avg_growth:.1f}%")
with col4:
    pop_growth = df.groupby('Country')['Population'].apply(
        lambda x: x.pct_change().mean()
    ).mean() * 100
    st.metric("Avg Population Growth", f"{pop_growth:.1f}%")

# NEW: Population-Emissions Scatter Plot with Time Animation
st.subheader("🚀 The Population-Emissions Paradox")
st.markdown("How population growth correlates with emissions growth across African nations")

fig = px.scatter(df, 
                 x='Population', 
                 y='Total CO2 Emission including LUCF (Mt)',
                 size='GDP PER CAPITA (USD)',
                 color='Sub-Region',
                 hover_name='Country',
                 animation_frame='Year',
                 animation_group='Country',
                 log_x=True,
                 size_max=45,
                 range_x=[1e5,1e8],
                 range_y=[0,500],
                 template='plotly_dark',
                 height=600)

fig.update_layout(
    title='Population vs CO2 Emissions Over Time',
    xaxis_title='Population (log scale)',
    yaxis_title='CO2 Emissions (Mt)',
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# NEW: Population Growth vs Emissions Growth Parallel Analysis
st.subheader("📊 Dual Analysis: Population & Emissions Trends")

# Create two columns for parallel analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧑‍🤝‍🧑 Population Growth Leaders")
    pop_growth_df = df.groupby('Country')['Population'].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
    ).nlargest(10).reset_index(name='Growth %')
    
    fig = px.bar(pop_growth_df, 
                 x='Country',
                 y='Growth %',
                 color='Growth %',
                 color_continuous_scale='purples',
                 title="Top 10 Population Growth Countries (2000-2020)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🏭 Emissions Growth Leaders")
    emissions_growth_df = df.groupby('Country')['Total CO2 Emission including LUCF (Mt)'].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
    ).nlargest(10).reset_index(name='Growth %')
    
    fig = px.bar(emissions_growth_df, 
                 x='Country',
                 y='Growth %',
                 color='Growth %',
                 color_continuous_scale='tealrose',
                 title="Top 10 Emissions Growth Countries (2000-2020)")
    st.plotly_chart(fig, use_container_width=True)

# NEW: Stacked Bar Chart Showing Population and Emissions Growth Side by Side
st.subheader("🧩 The Growth Puzzle: Population vs Emissions")

# Prepare data for comparison
comparison_df = df.groupby(['Year']).agg({
    'Population': 'sum',
    'Total CO2 Emission including LUCF (Mt)': 'sum'
}).reset_index()

# Normalize to percentage growth from 2000
comparison_df['Population Growth'] = (comparison_df['Population'] / comparison_df['Population'].iloc[0] - 1) * 100
comparison_df['Emissions Growth'] = (comparison_df['Total CO2 Emission including LUCF (Mt)'] / 
                                    comparison_df['Total CO2 Emission including LUCF (Mt)'].iloc[0] - 1) * 100

fig = go.Figure()

fig.add_trace(go.Bar(
    x=comparison_df['Year'],
    y=comparison_df['Population Growth'],
    name='Population Growth',
    marker_color='#9b59b6',
    opacity=0.8
))

fig.add_trace(go.Bar(
    x=comparison_df['Year'],
    y=comparison_df['Emissions Growth'],
    name='Emissions Growth',
    marker_color='#1abc9c',
    opacity=0.8
))

fig.update_layout(
    barmode='group',
    title='Comparing Population and Emissions Growth (2000 Baseline)',
    yaxis_title='Growth Percentage (%)',
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# =============================================
# Country-Specific Analysis - Enhanced with Population
# =============================================
st.header("🔍 Deep Dive: Country-Level Insights")

# Country selector with region filter
regions = sorted(df['Sub-Region'].unique())
selected_region = st.sidebar.selectbox("Filter by Region", ['All'] + regions, key='region_filter')
if selected_region != 'All':
    countries = sorted(df[df['Sub-Region'] == selected_region]['Country'].unique())
else:
    countries = sorted(df['Country'].unique())
    
selected_country = st.sidebar.selectbox("Select Country", countries, key='country_select')

# Year range selector
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    key='year_slider'
)

# Filter data
country_data = df[(df['Country'] == selected_country) & 
               (df['Year'] >= selected_years[0]) & 
               (df['Year'] <= selected_years[1])].sort_values('Year')

if not country_data.empty:
    latest_year = country_data['Year'].max()
    latest_data = country_data[country_data['Year'] == latest_year].iloc[0]
    
    # Country header with key metrics - now with population
    st.markdown(f"## {selected_country} Climate & Population Profile")
    
    # NEW: Dual metric cards for population and climate
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card population-card">
            <h3>Population</h3>
            <p style="font-size: 2rem; color: #9b59b6;">{latest_data['Population']/1e6:,.1f}M</p>
            <p>Growth: {country_data['Population Growth Rate'].iloc[-1]:.1f}% (2020)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card climate-card">
            <h3>CO2 Emissions</h3>
            <p style="font-size: 2rem; color: #1abc9c;">{latest_data['Total CO2 Emission including LUCF (Mt)']:,.1f} Mt</p>
            <p>{latest_data['Emissions per Capita']:.2f} t/person</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Safely calculate population growth since 2000
        try:
            pop_2000 = country_data[country_data['Year'] == 2000]['Population'].values[0]
            pop_growth = ((latest_data['Population'] - pop_2000) / pop_2000) * 100
            growth_text = f"+{pop_growth:.1f}%"
        except (IndexError, KeyError):
            growth_text = "N/A"
            
        st.markdown(f"""
        <div class="metric-card">
            <h3>Population Growth</h3>
            <p style="font-size: 2rem; color: #9b59b6;">{growth_text}</p>
            <p>Since 2000</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Safely calculate emissions growth since 2000
        try:
            emissions_2000 = country_data[country_data['Year'] == 2000]['Total CO2 Emission including LUCF (Mt)'].values[0]
            emissions_growth = ((latest_data['Total CO2 Emission including LUCF (Mt)'] - emissions_2000) / emissions_2000) * 100
            growth_text = f"+{emissions_growth:.1f}%"
        except (IndexError, KeyError):
            growth_text = "N/A"
            
        st.markdown(f"""
        <div class="metric-card">
            <h3>Emissions Growth</h3>
            <p style="font-size: 2rem; color: #1abc9c;">{growth_text}</p>
            <p>Since 2000</p>
        </div>
        """, unsafe_allow_html=True)

    # NEW: Side-by-side population and emissions trends
    st.subheader("📈 Dual Trends: Population & Emissions Over Time")
    
    fig = go.Figure()
    
    # Add population trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=country_data['Year'],
        y=country_data['Population']/1e6,
        name='Population (Millions)',
        line=dict(color='#9b59b6', width=3),
        yaxis='y'
    ))
    
    # Add emissions trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=country_data['Year'],
        y=country_data['Total CO2 Emission including LUCF (Mt)'],
        name='CO2 Emissions (Mt)',
        line=dict(color='#1abc9c', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f"{selected_country}: Population vs Emissions",
        xaxis_title='Year',
        yaxis=dict(
            title='Population (Millions)',
            title_font=dict(color='#9b59b6'),
            tickfont=dict(color='#9b59b6')
        ),
        yaxis2=dict(
            title='CO2 Emissions (Mt)',
            title_font=dict(color='#1abc9c'),
            tickfont=dict(color='#1abc9c'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Sectoral analysis with pie chart and threshold warnings
    st.subheader("🏭 Sectoral Breakdown with Threshold Monitoring")
    
    sector_cols = ['Transportation (Mt)', 'Manufacturing/Construction (Mt)', 
                  'Energy (Mt)', 'Electricity/Heat (Mt)']
    
    latest_sectors = country_data[country_data['Year'] == latest_year][sector_cols].iloc[0]
    
    # Create two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced 3D Pie chart showing sector contributions
        fig = go.Figure(go.Pie(
            labels=[s.replace(' (Mt)', '') for s in sector_cols],
            values=latest_sectors.values,
            hole=0.5,
            pull=[0.1 if i == latest_sectors.idxmax() else 0 for i in sector_cols],
            marker_colors=['#FFA07A', '#20B2AA', '#9370DB', '#FFD700'],
            textinfo='percent+label+value',
            textposition='inside',
            insidetextorientation='radial'
        ))
        
        fig.update_layout(
            title=f"<b>Sector Contribution in {latest_year}</b>",
            title_font=dict(size=18),
            showlegend=False,
            height=400,
            margin=dict(t=80, b=0, l=0, r=0),
            annotations=[dict(
                text=f"Total: {latest_sectors.sum():.1f} Mt",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Sector Threshold Status")
        # Threshold cards with labels - organized in a grid
        cols = st.columns(2)
        for idx, (sector, value) in enumerate(latest_sectors.items()):
            if sector in SECTOR_THRESHOLDS:
                if value > SECTOR_THRESHOLDS[sector]['danger']:
                    status = "danger"
                    alert = "🚨 Dangerous levels"
                    color = "#FF6B6B"
                elif value > SECTOR_THRESHOLDS[sector]['warning']:
                    status = "warning"
                    alert = "⚠️ Warning levels"
                    color = "#FFD166"
                else:
                    status = "safe"
                    alert = "✅ Within safe limits"
                    color = "#06D6A0"
                
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div style="background: white; border-radius: 15px; padding: 1rem; 
                                margin-bottom: 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                                border-left: 5px solid {color};">
                        <h3 style="margin-top: 0; color: {color};">{sector.replace(' (Mt)', '')}</h3>
                        <p style="font-size: 1.5rem; margin-bottom: 0.5rem;">{value:,.1f} Mt</p>
                        <p style="margin-bottom: 0.5rem;"><span style="background-color: {color}; 
                            width: 12px; height: 12px; border-radius: 50%; display: inline-block;"></span> {alert}</p>
                        <div style="font-size: 0.8rem; color: #666;">
                            <span style="color: #FFD166;">Warning: {SECTOR_THRESHOLDS[sector]['warning']} Mt</span><br>
                            <span style="color: #FF6B6B;">Danger: {SECTOR_THRESHOLDS[sector]['danger']} Mt</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Time series for each sector
    st.subheader("📊 Sector Trends Over Time")
    sector = st.selectbox("Select sector to analyze", sector_cols, key='sector_select')
    
    fig = px.line(country_data, 
                  x='Year', 
                  y=sector,
                  title=f"{sector.replace(' (Mt)', '')} Trend in {selected_country}",
                  markers=True)
    
    # Add threshold lines if defined
    if sector in SECTOR_THRESHOLDS:
        fig.add_hline(y=SECTOR_THRESHOLDS[sector]['warning'], 
                      line_dash="dot", 
                      annotation_text=f"Warning: {SECTOR_THRESHOLDS[sector]['warning']} Mt", 
                      line_color="orange",
                      annotation_position="bottom right")
        fig.add_hline(y=SECTOR_THRESHOLDS[sector]['danger'], 
                      line_dash="dash", 
                      annotation_text=f"Danger: {SECTOR_THRESHOLDS[sector]['danger']} Mt", 
                      line_color="red",
                      annotation_position="top right")
    
    st.plotly_chart(fig, use_container_width=True)

    # =============================================
    # Policy Insights Section - Enhanced with Population
    # =============================================
    st.header("💡 Actionable Insights for Policymakers")
    
    # Generate dynamic recommendations based on data
    latest = country_data[country_data['Year'] == latest_year].iloc[0]
    recommendations = []
    
    # Population-based recommendations
    pop_growth = country_data['Population Growth Rate'].mean()
    if pop_growth > 2.5:
        recommendations.append(
            "👶 **High Population Growth**: Invest in family planning education and women's empowerment programs "
            f"(current growth: {pop_growth:.1f}%)"
        )
    
    # Transportation recommendations
    if latest['Transportation (Mt)'] > SECTOR_THRESHOLDS['Transportation (Mt)']['warning']:
        recommendations.append(
            "🚗 **Transportation**: Implement electric vehicle incentives and improve public transit "
            f"(current: {latest['Transportation (Mt)']:,.1f} Mt)"
        )
    
    # Energy recommendations
    if latest['Energy (Mt)'] > SECTOR_THRESHOLDS['Energy (Mt)']['warning']:
        recommendations.append(
            "⚡ **Energy**: Accelerate renewable energy adoption and phase out coal plants "
            f"(current: {latest['Energy (Mt)']:,.1f} Mt)"
        )
    
    # Urbanization recommendations
    if latest['Population Density'] > 70:  # Assuming normalized density
        recommendations.append(
            "🏙️ **Urbanization**: Develop green cities with sustainable infrastructure and public spaces"
        )
    
    if recommendations:
        st.markdown(f"""
        <div class="insight-card">
            <h3>Priority Actions for {selected_country}</h3>
            <ul>
                {"".join([f"<li>{rec}</li>" for rec in recommendations])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("All indicators are within sustainable thresholds. Maintain current policies.")

# =============================================
# Footer
# =============================================
st.markdown(f"""
<div style="text-align: center; margin-top: 4rem; color: #7f8c8d; font-size: 0.9rem;">
    <p>AFRICA CLIMATE & POPULATION INSIGHTS | Developed for sustainable development</p>
    <p> Developed by researcher Samson Niyizurugero, with support from AGNES (African Group of Negotiators Experts) and AIMS Rwanda<p>
    <p>Data sources: Global Carbon Project, World Bank, UN Population Division | Updated: {datetime.now().strftime("%B %d, %Y")}</p>
</div>
""", unsafe_allow_html=True)
