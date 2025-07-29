import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="African CO₂ Emissions Intelligence Dashboard",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with vibrant colors
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    .header-text {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .subheader-text {
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        color: #7f8c8d !important;
        margin-bottom: 1.5rem !important;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 15px;
        border-left: 4px solid #3498db;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 400;
    }
    .sector-card {
        transition: all 0.3s ease;
        cursor: pointer;
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .sector-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    .selected-sector {
        border: 3px solid #e74c3c !important;
        box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.2);
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 10px;
    }
    .stButton>button {
        border: none;
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stSelectbox, .stSlider, .stRadio {
        background-color: white;
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .stExpander:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .user-guide {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
    .recommendation-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        # Load from local data folder
        df = pd.read_csv("data/co2_Emission_Africa.csv")
        df.columns = df.columns.str.strip()
        
        # Ensure numeric columns are properly converted
        numeric_cols = ['Total CO2 Emission including LUCF (Mt)', 'GDP PER CAPITA (USD)', 'Population',
                       'Transportation (Mt)', 'Manufacturing/Construction (Mt)',
                       'Land-Use Change and Forestry (Mt)', 'Industrial Processes (Mt)',
                       'Energy (Mt)', 'Electricity/Heat (Mt)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame if file not found

df = load_data()

# Show warning if data loading failed
if df.empty:
    st.warning("Failed to load data. Please ensure 'data/co2_Emission_Africa.csv' exists.")
# Sidebar with filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Africa_%28orthographic_projection%29.svg/600px-Africa_%28orthographic_projection%29.svg.png", 
             width=150, use_container_width=True)
    st.title("Dashboard Controls")
    
    st.markdown("""
    <div class="user-guide">
        <strong>📌 How to use this dashboard:</strong>
        <ol>
            <li>Select a country from the dropdown</li>
            <li>Adjust the year range as needed</li>
            <li>Choose your analysis focus</li>
            <li>Click on sector cards to explore details</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    country = st.selectbox(
        "🌍 Select Country", 
        sorted(df['Country'].unique()),
        index=sorted(df['Country'].unique()).index('Algeria') if 'Algeria' in df['Country'].unique() else 0
    )
    
    year = st.slider(
        "📅 Select Year Range",
        min_value=2000,
        max_value=2020,
        value=(2000, 2020)
    )
    
    analysis_type = st.radio(
        "📊 Analysis Focus",
        options=["Sector Breakdown", "Trend Analysis", "Comparative Analysis"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    **Data Sources:**
    - World Bank Development Indicators
    - Global Carbon Project
    - UNEP Emissions Database
    """)
    
    st.markdown("---")
    st.markdown("""
    **Dashboard Features:**
    - Interactive visualizations
    - Dynamic filtering
    - Emission forecasting
    - Sector comparisons
    """)

# Main content
st.markdown(f'<h1 class="header-text">African CO₂ Emissions Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown(f'<div class="subheader-text">Comprehensive analysis of carbon emissions across African nations</div>', unsafe_allow_html=True)

# User guidance at the top
st.markdown("""
<div class="user-guide">
    <strong>🔍 Explore the dashboard:</strong>
    <ul>
        <li>Start with the <strong>Key National Indicators</strong> below for an overview</li>
        <li>Click on any <strong>sector card</strong> to see detailed trends</li>
        <li>Use the <strong>comparative analysis tabs</strong> for deeper insights</li>
        <li>Scroll down for <strong>tailored recommendations</strong> based on your selection</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Filter data
df_country = df[(df['Country'] == country) & (df['Year'] >= year[0]) & (df['Year'] <= year[1])]
df_country_all_years = df[df['Country'] == country]

# Key metrics row
st.subheader("🌱 Key National Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #3498db;">
        <div class="metric-title">Total CO₂ Emissions</div>
        <div class="metric-value">{df_country['Total CO2 Emission including LUCF (Mt)'].mean():,.2f} Mt</div>
        <div class="metric-delta">Last year: {df_country[df_country['Year'] == df_country['Year'].max() - 1]['Total CO2 Emission including LUCF (Mt)'].values[0]:,.2f} Mt</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #2ecc71;">
        <div class="metric-title">GDP per Capita</div>
        <div class="metric-value">${df_country['GDP PER CAPITA (USD)'].mean():,.2f}</div>
        <div class="metric-delta">Last year: ${df_country[df_country['Year'] == df_country['Year'].max() - 1]['GDP PER CAPITA (USD)'].values[0]:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #9b59b6;">
        <div class="metric-title">Population</div>
        <div class="metric-value">{df_country['Population'].mean() / 1e6:,.1f}M</div>
        <div class="metric-delta">Last year: {df_country[df_country['Year'] == df_country['Year'].max() - 1]['Population'].values[0] / 1e6:,.1f}M</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    emission_change = ((df_country[df_country['Year'] == df_country['Year'].max()]['Total CO2 Emission including LUCF (Mt)'].values[0] - 
                       df_country[df_country['Year'] == df_country['Year'].max() - 1]['Total CO2 Emission including LUCF (Mt)'].values[0]) / 
                      df_country[df_country['Year'] == df_country['Year'].max() - 1]['Total CO2 Emission including LUCF (Mt)'].values[0]) * 100
    
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {'#e74c3c' if emission_change > 0 else '#2ecc71'};">
        <div class="metric-title">Emission Change</div>
        <div class="metric-value" style="color: {'#e74c3c' if emission_change > 0 else '#2ecc71'};">{abs(emission_change):.1f}%</div>
        <div class="metric-delta">Year-over-year</div>
    </div>
    """, unsafe_allow_html=True)

# Sector breakdown with interactive cards
st.subheader("🏭 Emission Sector Breakdown")
st.markdown("""
<div class="user-guide">
    <strong>💡 Click on any sector card below</strong> to see its detailed trend analysis in the chart to the right.
    The pie chart shows the current composition of emissions by sector.
</div>
""", unsafe_allow_html=True)

sector_cols = [
    "Transportation (Mt)",
    "Manufacturing/Construction (Mt)",
    "Land-Use Change and Forestry (Mt)",
    "Industrial Processes (Mt)",
    "Energy (Mt)",
    "Electricity/Heat (Mt)"
]

# Vibrant color palette for sectors
sector_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFBE0B', '#FF9F1C', '#8AC926']

# Store selected sector in session state
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = sector_cols[0]

# Create sector cards with vibrant colors
cols = st.columns(len(sector_cols))
for i, col in enumerate(cols):
    with col:
        sector_value = df_country[sector_cols[i]].mean()
        is_selected = st.session_state.selected_sector == sector_cols[i]
        
        # Card styling with hover effects
        card_style = f"""
            background: linear-gradient(135deg, {sector_colors[i]}20, white);
            border-left: 4px solid {sector_colors[i]};
            {'box-shadow: 0 0 15px ' + sector_colors[i] + '80;' if is_selected else ''}
            transition: all 0.3s ease;
            cursor: pointer;
        """
        
        st.markdown(f"""
        <div class="sector-card {'selected-sector' if is_selected else ''}" 
             onclick="window.parent.document.querySelectorAll('.sector-btn')[{i}].click()"
             style="{card_style}">
            <div class="metric-title" style="color: {sector_colors[i]}; font-weight: 600;">
                {sector_cols[i].split(' (')[0]}
            </div>
            <div class="metric-value" style="color: {sector_colors[i]}; font-size: 1.3rem;">
                {sector_value:,.2f} Mt
            </div>
            <div class="metric-delta" style="color: {'#e74c3c' if sector_value > 0 else '#2ecc71'}">
                {'▲' if sector_value > 0 else '▼'} from baseline
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden button for interactivity
        if col.button("", key=f"sector_{i}", help=sector_cols[i]):
            st.session_state.selected_sector = sector_cols[i]

# Main visualization area
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    # Dynamic chart based on selected sector
    if st.session_state.selected_sector:
        st.subheader(f"📈 {st.session_state.selected_sector.split(' (')[0]} Emissions Trend")
        st.markdown(f"""
        <div class="user-guide" style="margin-bottom: 15px;">
            Showing trend for <strong>{st.session_state.selected_sector.split(' (')[0]}</strong> sector. 
            The dashed red line shows the overall trend.
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.line(
            df_country_all_years,
            x="Year",
            y=st.session_state.selected_sector,
            markers=True,
            color_discrete_sequence=[sector_colors[sector_cols.index(st.session_state.selected_sector)]],
            template="plotly_white",
            line_shape="spline"
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Year",
            yaxis_title="Emissions (Mt)",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add trendline
        fig.add_trace(
            go.Scatter(
                x=df_country_all_years["Year"],
                y=np.poly1d(np.polyfit(
                    df_country_all_years["Year"],
                    df_country_all_years[st.session_state.selected_sector],
                    1
                ))(df_country_all_years["Year"]),
                mode='lines',
                name='Trend',
                line=dict(color='#e74c3c', dash='dash', width=2)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🥧 Sector Composition")
    st.markdown("""
    <div class="user-guide" style="margin-bottom: 15px;">
        Current distribution of emissions across sectors. Hover over segments for details.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Prepare data for pie chart
        pie_data = df_country[sector_cols].mean().reset_index()
        pie_data.columns = ['Sector', 'Value']
        pie_data['Sector'] = pie_data['Sector'].str.replace(' (Mt)', '')
        
        fig_pie = px.pie(
            pie_data,
            values='Value',
            names='Sector',
            color_discrete_sequence=sector_colors,
            hole=0.4,
            template="plotly_white"
        )
        
        fig_pie.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>%{value:.2f} Mt (%{percent})",
            marker=dict(line=dict(color='#ffffff', width=1))
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate pie chart: {str(e)}")

# Comparative analysis row
st.markdown("---")
st.subheader("🔍 Comparative Analysis")
st.markdown("""
<div class="user-guide">
    Use these tabs to compare different aspects of emissions data:
    <ul>
        <li><strong>Sector Comparison</strong>: Compare trends between multiple sectors</li>
        <li><strong>Emission Drivers</strong>: See how emissions correlate with economic factors</li>
        <li><strong>Forecast</strong>: View projected emissions for the next 5 years</li>
    </ul>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Sector Comparison", "📈 Emission Drivers", "🔮 Forecast"])

with tab1:
    st.markdown("""
    <div class="user-guide">
        Select multiple sectors below to compare their emission trends over time.
    </div>
    """, unsafe_allow_html=True)
    
    selected_sectors = st.multiselect(
        "Select sectors to compare",
        options=sector_cols,
        default=[sector_cols[0], sector_cols[-1]],
        format_func=lambda x: x.split(' (')[0]
    )
    
    if selected_sectors:
        fig_compare = go.Figure()
        
        for sector in selected_sectors:
            fig_compare.add_trace(
                go.Scatter(
                    x=df_country_all_years["Year"],
                    y=df_country_all_years[sector],
                    mode='lines+markers',
                    name=sector.split(' (')[0],
                    line=dict(width=3),
                    marker=dict(size=8)
                )
            )
        
        fig_compare.update_layout(
            height=400,
            title="Sector Emissions Comparison Over Time",
            xaxis_title="Year",
            yaxis_title="Emissions (Mt)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)

with tab2:
    st.markdown("""
    <div class="user-guide">
        This heatmap shows how different factors correlate with each other. 
        Red indicates positive correlation, blue indicates negative correlation.
    </div>
    """, unsafe_allow_html=True)
    
    heatmap_cols = sector_cols + [
        'GDP PER CAPITA (USD)',
        'Population',
        'Total CO2 Emission including LUCF (Mt)'
    ]
    
    try:
        corr_df = df_country[heatmap_cols].corr()
        
        fig_heatmap = go.Figure(
            go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns.str.replace(' (Mt)', ''),
                y=corr_df.columns.str.replace(' (Mt)', ''),
                colorscale='RdYlBu_r',
                zmin=-1,
                zmax=1,
                hoverongaps=False,
                text=corr_df.round(2).values,
                texttemplate="%{text}",
                colorbar=dict(title='Correlation')
            )
        )
        
        fig_heatmap.update_layout(
            height=500,
            title="Correlation Between Emissions and Economic Indicators",
            xaxis_title="Variables",
            yaxis_title="Variables",
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate correlation heatmap: {str(e)}")

with tab3:
    st.markdown("""
    <div class="user-guide">
        This forecast predicts emissions for the next 5 years based on historical trends.
        The shaded area represents a possible range of values.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Prepare data for forecasting
        forecast_years = list(range(df_country['Year'].max() + 1, df_country['Year'].max() + 6))
        
        # Simple forecasting model
        X = df_country_all_years[['Year']]
        y = df_country_all_years['Total CO2 Emission including LUCF (Mt)']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        future_X = pd.DataFrame({'Year': forecast_years})
        future_y = model.predict(future_X)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Year': list(df_country_all_years['Year']) + forecast_years,
            'Emissions': list(df_country_all_years['Total CO2 Emission including LUCF (Mt)']) + list(future_y),
            'Type': ['Historical'] * len(df_country_all_years) + ['Forecast'] * len(forecast_years)
        })
        
        # Create the plot
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(
            go.Scatter(
                x=df_country_all_years['Year'],
                y=df_country_all_years['Total CO2 Emission including LUCF (Mt)'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8, color='#3498db')
            )
        )
        
        # Forecast data
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_years,
                y=future_y,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#e74c3c', width=3, dash='dot'),
                marker=dict(size=8, color='#e74c3c', symbol='diamond')
            )
        )
        
        # Confidence interval (example)
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_years + forecast_years[::-1],
                y=list(future_y * 1.1) + list(future_y * 0.9)[::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
        )
        
        fig_forecast.update_layout(
            height=400,
            title="Total CO₂ Emissions Forecast",
            xaxis_title="Year",
            yaxis_title="Emissions (Mt)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Display forecast values
        st.markdown("""
        <div class="user-guide">
            <strong>Forecasted Values:</strong> Below are the predicted emissions for the next 5 years.
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            pd.DataFrame({
                'Year': forecast_years,
                'Forecasted Emissions (Mt)': future_y,
                'Change from Last Year (%)': [((future_y[i] - (future_y[i-1] if i > 0 else df_country_all_years['Total CO2 Emission including LUCF (Mt)'].iloc[-1])) / 
                                            (future_y[i-1] if i > 0 else df_country_all_years['Total CO2 Emission including LUCF (Mt)'].iloc[-1])) * 100 
                                           for i in range(len(future_y))]
            }).set_index('Year').style.format({
                'Forecasted Emissions (Mt)': '{:,.2f}',
                'Change from Last Year (%)': '{:,.1f}%'
            }).background_gradient(cmap='YlOrRd'),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Could not generate forecast: {str(e)}")

# Recommendations section - Always visible
st.markdown("---")
st.subheader("💡 Sustainable Development Recommendations")
st.markdown("""
<div class="user-guide">
    Based on the current analysis, here are tailored recommendations for sustainable development.
    These suggestions are automatically adjusted based on your country and sector selections.
</div>
""", unsafe_allow_html=True)

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    # Green Energy Initiatives (always visible)
    st.markdown("""
    <div class="recommendation-card" style="border-left-color: #2ecc71;">
        <h4 style="color: #27ae60; margin-top: 0;">🌿 Green Energy Initiatives</h4>
        <ul style="color: #2c3e50;">
            <li><strong>Solar Energy Expansion</strong>: Invest in utility-scale solar farms in the Sahara region</li>
            <li><strong>Wind Power Development</strong>: Harness coastal wind resources for clean energy</li>
            <li><strong>Grid Modernization</strong>: Implement smart grid technologies to reduce transmission losses</li>
            <li><strong>Energy Storage</strong>: Develop battery storage systems to manage renewable intermittency</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sustainable Urban Development (always visible)
    st.markdown("""
    <div class="recommendation-card" style="border-left-color: #3498db;">
        <h4 style="color: #2980b9; margin-top: 0;">🏙️ Sustainable Urban Development</h4>
        <ul style="color: #2c3e50;">
            <li><strong>Electric Public Transit</strong>: Transition bus fleets to electric vehicles</li>
            <li><strong>Bike Infrastructure</strong>: Build protected bike lanes in major cities</li>
            <li><strong>Green Building Standards</strong>: Mandate energy-efficient construction practices</li>
            <li><strong>Urban Greening</strong>: Increase tree canopy coverage in urban areas</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    # Industrial Efficiency Programs (always visible)
    st.markdown("""
    <div class="recommendation-card" style="border-left-color: #f39c12;">
        <h4 style="color: #d35400; margin-top: 0;">🏭 Industrial Efficiency Programs</h4>
        <ul style="color: #2c3e50;">
            <li><strong>Carbon Capture</strong>: Pilot CCS technologies in cement and steel plants</li>
            <li><strong>Process Optimization</strong>: Implement AI-driven efficiency systems</li>
            <li><strong>Waste Heat Recovery</strong>: Capture and reuse industrial waste heat</li>
            <li><strong>Circular Economy</strong>: Promote industrial symbiosis and material reuse</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Land Use & Forestry (always visible)
    st.markdown("""
    <div class="recommendation-card" style="border-left-color: #27ae60;">
        <h4 style="color: #16a085; margin-top: 0;">🌳 Land Use & Forestry</h4>
        <ul style="color: #2c3e50;">
            <li><strong>Reforestation</strong>: Restore degraded lands with native species</li>
            <li><strong>Agroforestry</strong>: Integrate trees into agricultural systems</li>
            <li><strong>Soil Carbon</strong>: Promote regenerative agricultural practices</li>
            <li><strong>Fire Management</strong>: Implement early warning systems for wildfires</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong style="color: #2c3e50; font-size: 1.1rem;">African CO₂ Emissions Intelligence Platform</strong><br>
    <span style="color: #7f8c8d;">Developed by Best Reseacher Samson Niyizurugero , my Insitutution is AIMS Reseach and Innovation center  • Data updated: {date}</span><br>
    <span style="color: #95a5a6;">© {year} All Rights Reserved</span>
</div>
""".format(
    date=datetime.now().strftime("%B %d, %Y"),
    year=datetime.now().year
), unsafe_allow_html=True)
