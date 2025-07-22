# main.py

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from dotenv import load_dotenv
from prophet import Prophet

# ─── Streamlit Page Setup ────────────────────────────────
st.set_page_config(
    page_title="Energy Market Price Tracker",
    layout="wide"
)

# ─── Load Environment Variables ──────────────────────────
load_dotenv()
API_KEY = os.getenv("EIA_API_KEY")

# ─── Fetch EIA Data ──────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading data...")
def fetch_eia_data():
    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    params = {
        "api_key": API_KEY,
        "data[0]": "price",
        "data[1]": "revenue",
        "data[2]": "sales",
        "data[3]": "customers",
        "frequency": "monthly",
        "facets[stateid][]": "NY",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "response" in data and "data" in data["response"]:
        df = pd.DataFrame(data["response"]["data"])
        df["Date"] = pd.to_datetime(df["period"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df.dropna(subset=["price"])
    else:
        return pd.DataFrame()

# ─── Main Interface ──────────────────────────────────────
st.title("⚡ Energy Market Price Tracker - New York")
df = fetch_eia_data()

if df.empty:
    st.error("Failed to load data from EIA API.")
    st.stop()

# ─── Sidebar Filters ─────────────────────────────────────
with st.sidebar:
    st.header("🔧 Filters")
    sector_options = df["sectorName"].unique()
    selected_sector = st.selectbox("Select Sector", sector_options)

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    granularity = st.radio("Granularity", ["Monthly", "Yearly"])
    show_forecast = st.toggle("Show Forecast")  # ✅ New toggle

# ─── KPI Calculations (Always from monthly data) ─────────
kpi_df = df[
    (df["sectorName"] == selected_sector) &
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
].copy()

kpi_df["sales"] = pd.to_numeric(kpi_df["sales"], errors="coerce")
kpi_df["revenue"] = pd.to_numeric(kpi_df["revenue"], errors="coerce")
kpi_df["price"] = pd.to_numeric(kpi_df["price"], errors="coerce")

average_price = kpi_df["price"].mean()
max_price = kpi_df["price"].max()
price_change = ((kpi_df["price"].iloc[-1] - kpi_df["price"].iloc[0]) / kpi_df["price"].iloc[0]) * 100
sales_total = kpi_df["sales"].sum()
revenue_total = kpi_df["revenue"].sum()

# ─── KPI Cards ───────────────────────────────────────────
kpi_cols = st.columns(5)
kpi_data = [
    ("Average Price (¢/kWh)", f"{average_price:.2f}" if pd.notna(average_price) else "N/A"),
    ("Max Price (¢/kWh)", f"{max_price:.2f}" if pd.notna(max_price) else "N/A"),
    ("% Price Change", f"{price_change:.2f}%" if pd.notna(price_change) else "N/A"),
    ("Total Sales (kWh)", f"{sales_total:,.0f}" if pd.notna(sales_total) else "N/A"),
    ("Total Revenue ($)", f"${revenue_total:,.0f}" if pd.notna(revenue_total) else "N/A")
]

for col, (label, value) in zip(kpi_cols, kpi_data):
    col.markdown(f"""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;
                box-shadow: 1px 1px 6px rgba(0,0,0,0.1); text-align: center;'>
        <h4 style='margin: 0; color: #444;'>{label}</h4>
        <p style='font-size: 24px; font-weight: bold; margin: 5px 0 0 0; color: #000;'>{value}</p>
    </div>
""", unsafe_allow_html=True)

# ─── Filter and Aggregate Data ───────────────────────────
filtered_df = kpi_df.copy()
if granularity == "Yearly":
    filtered_df["Year"] = filtered_df["Date"].dt.year
    filtered_df = filtered_df.groupby("Year").agg({"price": "mean"}).reset_index()
    x_col = "Year"
else:
    x_col = "Date"

# ─── NLP-Style Insight ───────────────────────────────────
if granularity == "Monthly":
    trend = "increased" if kpi_df["price"].iloc[-1] > kpi_df["price"].iloc[0] else "decreased"
    st.info(
        f"From {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}, electricity prices for the {selected_sector} sector in New York have {trend}. "
        f"The average price was {average_price:.2f} ¢/kWh, with a peak of {max_price:.2f} ¢/kWh."
    )

# ─── Plot Chart with Optional Forecast ───────────────────
fig = px.line(
    filtered_df,
    x=x_col,
    y="price",
    title=f"📈 Electricity Price Trend for {selected_sector} Sector",
    labels={"price": "¢/kWh", x_col: x_col},
)

# ✅ Add Prophet Forecast if enabled
if show_forecast and granularity == "Monthly":
    prophet_df = filtered_df[["Date", "price"]].rename(columns={"Date": "ds", "price": "y"})
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)

    fig.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="orange", dash="dash")
    )

st.plotly_chart(fig, use_container_width=True)