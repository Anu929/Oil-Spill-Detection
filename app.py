import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Configure page settings
st.set_page_config(
    page_title="Oil Spill Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("Oil Spills global data.csv")
    df = df.drop(columns=["Entity", "Code"])
    df.columns = ["Year", "Large_Spills", "Medium_Spills"]
    df["Large_Spills"] = pd.to_numeric(df["Large_Spills"], errors='coerce')
    df["Medium_Spills"] = pd.to_numeric(df["Medium_Spills"], errors='coerce')
    return df

df = load_data()

# Train ML models for both large and medium spills
@st.cache_resource
def train_model(df):
    model_large = LinearRegression()
    model_large.fit(df[["Year"]], df["Large_Spills"])
    
    model_medium = LinearRegression()
    model_medium.fit(df[["Year"]], df["Medium_Spills"])
    
    return model_large, model_medium

model_large, model_medium = train_model(df)

# Main Header
st.markdown("""
    <div class="main-header">
        <h1>🌊 Marine Oil Spill Prediction Dashboard</h1>
        <p>AI-Powered Forecasting of Global Oil Spill Trends</p>
    </div>
""", unsafe_allow_html=True)

# Introduction section
with st.expander("📚 About This Dashboard", expanded=True):
    st.markdown("""
    This dashboard uses **Machine Learning** to predict the number of large oil spills 
    that will occur in future years based on historical data from 1970 to 2023.
    
    **Key Features:**
    - 📊 Visualize historical oil spill trends over 50+ years
    - 🔮 Get AI-powered predictions for 2025-2034
    - 📈 Interactive charts for deep insights
    - 💡 Understand spill patterns and trends
    """)

# Information box
st.markdown("""
    <div class="info-box">
    <strong>💡 How It Works:</strong> Our AI model analyzes decades of historical data to identify patterns 
    and predict future trends. Select any year below to see the predicted number of large oil spills 
    for that year.
    </div>
""", unsafe_allow_html=True)

# Prediction section
st.markdown("---")
st.subheader("🔮 Make a Prediction")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    year = st.slider(
        "Select a year to predict:",
        min_value=2025,
        max_value=2034,
        value=2025,
        step=1,
        help="Choose any year from 2025 to 2034"
    )

# Make prediction
prediction_large = model_large.predict(pd.DataFrame({"Year": [year]}))[0]
prediction_medium = model_medium.predict(pd.DataFrame({"Year": [year]}))[0]
prediction_large_rounded = int(max(0, round(prediction_large)))
prediction_medium_rounded = int(max(0, round(prediction_medium)))

# Display predictions in nice boxes side by side
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(f"""
        <div class="metric-box">
            <h4>Large Spills</h4>
            <h2 style="color: #e74c3c; font-size: 2.5em;">{prediction_large_rounded}</h2>
            <p style="color: #666; font-size: 0.9em;">incidents in {year}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);">
            <p style="color: #666; font-size: 0.85em; margin-bottom: 0.5rem;">Prediction for</p>
            <h3 style="color: #667eea; margin: 0; font-size: 1.5em;">{year}</h3>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-box">
            <h4>Medium Spills</h4>
            <h2 style="color: #3498db; font-size: 2.5em;">{prediction_medium_rounded}</h2>
            <p style="color: #666; font-size: 0.9em;">incidents in {year}</p>
        </div>
    """, unsafe_allow_html=True)

# Historical data table
st.markdown("---")
st.subheader("📋 Historical Data Overview")

data_summary = df.tail(10).copy()
data_summary.columns = ["Year", "Large Spills", "Medium Spills"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Latest Year Data", int(df["Year"].max()), f"{int(df['Large_Spills'].iloc[-1])} large spills")
with col2:
    avg_recent = df['Large_Spills'].tail(10).mean()
    st.metric("📈 Average (Last 10 Years)", f"{avg_recent:.1f}", "large spills/year")
with col3:
    st.metric("📉 Total Historical Data", f"{len(df)} years", "1970-2023")

st.dataframe(
    data_summary,
    use_container_width=True,
    hide_index=True
)

# Interactive Visualization section
st.markdown("---")
st.subheader("📈 Interactive Historical Trends & Future Predictions")

st.markdown("""
    <div class="info-box">
    <strong>What You're Looking At:</strong> Hover over the chart to see exact values. 
    The red line shows large oil spills, blue shows medium spills. The dashed line shows 
    our AI predictions for future years.
    </div>
""", unsafe_allow_html=True)

# Create Plotly interactive chart
fig = go.Figure()

# Historical large spills
fig.add_trace(go.Scatter(
    x=df["Year"],
    y=df["Large_Spills"],
    mode='lines+markers',
    name='Large Spills (Historical)',
    line=dict(color='#e74c3c', width=3),
    marker=dict(size=6),
    hovertemplate='<b>Year: %{x}</b><br>Large Spills: %{y}<extra></extra>'
))

# Historical medium spills
fig.add_trace(go.Scatter(
    x=df["Year"],
    y=df["Medium_Spills"],
    mode='lines+markers',
    name='Medium Spills (Historical)',
    line=dict(color='#3498db', width=3),
    marker=dict(size=6),
    hovertemplate='<b>Year: %{x}</b><br>Medium Spills: %{y}<extra></extra>'
))

# Future predictions
future_years = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034])
future_predictions_large = model_large.predict(pd.DataFrame({"Year": future_years}))
future_predictions_medium = model_medium.predict(pd.DataFrame({"Year": future_years}))
future_predictions_large = np.maximum(future_predictions_large, 0)
future_predictions_medium = np.maximum(future_predictions_medium, 0)

fig.add_trace(go.Scatter(
    x=future_years,
    y=future_predictions_large,
    mode='lines+markers',
    name='Large Spills (Predicted)',
    line=dict(color='#e74c3c', width=2.5, dash='dash'),
    marker=dict(size=6, symbol='diamond'),
    hovertemplate='<b>Year: %{x}</b><br>Predicted: %{y:.1f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=future_years,
    y=future_predictions_medium,
    mode='lines+markers',
    name='Medium Spills (Predicted)',
    line=dict(color='#3498db', width=2.5, dash='dash'),
    marker=dict(size=6, symbol='diamond'),
    hovertemplate='<b>Year: %{x}</b><br>Predicted: %{y:.1f}<extra></extra>'
))

# Update layout
fig.update_layout(
    title='Oil Spill Trends: Historical Data & AI Predictions',
    xaxis_title='Year',
    yaxis_title='Number of Incidents',
    hovermode='x unified',
    plot_bgcolor='#f8f9fa',
    paper_bgcolor='white',
    font=dict(size=12),
    height=500,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

st.plotly_chart(fig, use_container_width=True)

# Prediction table for all years
st.markdown("---")
st.subheader("🔮 Predictions for All Years (2025-2034)")

prediction_data = []
for yr in range(2025, 2035):
    pred_large = model_large.predict(pd.DataFrame({"Year": [yr]}))[0]
    pred_medium = model_medium.predict(pd.DataFrame({"Year": [yr]}))[0]
    pred_large_rounded = int(max(0, round(pred_large)))
    pred_medium_rounded = int(max(0, round(pred_medium)))
    prediction_data.append({
        "Year": yr, 
        "Large Spill Incidents": pred_large_rounded,
        "Medium Spill Incidents": pred_medium_rounded,
        "Total Incidents": pred_large_rounded + pred_medium_rounded
    })

pred_df = pd.DataFrame(prediction_data)
st.dataframe(pred_df, use_container_width=True, hide_index=True)

# Insights section
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    avg_large = df["Large_Spills"].mean()
    avg_medium = df["Medium_Spills"].mean()
    st.markdown(f"""
        <div class="info-box">
        <strong>📊 Historical Averages:</strong><br>
        • Large Spills: <strong>{avg_large:.1f}</strong> per year<br>
        • Medium Spills: <strong>{avg_medium:.1f}</strong> per year
        </div>
    """, unsafe_allow_html=True)

with col2:
    latest_large = int(df["Large_Spills"].iloc[-1])
    latest_medium = int(df["Medium_Spills"].iloc[-1])
    st.markdown(f"""
        <div class="info-box">
        <strong>🔍 Latest Year ({int(df['Year'].iloc[-1])}):</strong><br>
        • Large Spills: <strong>{latest_large}</strong><br>
        • Medium Spills: <strong>{latest_medium}</strong>
        </div>
    """, unsafe_allow_html=True)

# Interactive chart for exploring data
st.markdown("---")
st.subheader("🎯 Comparison Chart - Select Years to Compare")

# Multi-select for comparison
selected_years = st.multiselect(
    "Select years to compare:",
    df["Year"].values,
    default=[df["Year"].min(), df["Year"].max()]
)

if selected_years:
    comparison_df = df[df["Year"].isin(selected_years)].sort_values("Year")
    
    fig2 = go.Figure(data=[
        go.Bar(x=comparison_df["Year"], y=comparison_df["Large_Spills"], name="Large Spills"),
        go.Bar(x=comparison_df["Year"], y=comparison_df["Medium_Spills"], name="Medium Spills")
    ])
    
    fig2.update_layout(
        barmode='group',
        title='Oil Spill Comparison by Year',
        xaxis_title='Year',
        yaxis_title='Number of Incidents',
        plot_bgcolor='#f8f9fa',
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Warning section
st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Note:</strong> These predictions are based on historical trends using machine learning. 
    Actual oil spill incidents depend on many factors including environmental practices, regulation changes, 
    and technological improvements. Use these forecasts as a reference tool, not absolute predictions.
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 2rem;">
    <p>🌊 Marine Oil Spill Prediction Dashboard | Data Source: Global Oil Spill Database (1970-2023)</p>
    <p>Powered by AI & Machine Learning | Interactive Charts with Plotly</p>
    </div>
""", unsafe_allow_html=True)
