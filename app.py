import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Oil Spills global data.csv")
df = df.drop(columns=["Entity", "Code"])
df.columns = ["Year", "Large_Spills", "Medium_Spills"]

# Train ML model
model = LinearRegression()
model.fit(df[["Year"]], df["Large_Spills"])

# Streamlit app
st.title("Marine Oil Spill Prediction Dashboard")
st.subheader("Predicting Future Large Oil Spills (2025–2034)")

# Slider for year input
year = st.slider("Select a year:", 2025, 2034, 2025)
prediction = model.predict([[year]])[0]

# Output prediction
st.write(f"### 🔮 Predicted Large Oil Spills in {year}: **{round(prediction)} incidents**")

# Show trend graph
st.subheader("📈 Historical Oil Spill Trend (1970–2023)")
fig, ax = plt.subplots()
ax.plot(df["Year"], df["Large_Spills"], label="Large Spills", color="red")
ax.plot(df["Year"], df["Medium_Spills"], label="Medium Spills", color="blue")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Spills")
ax.legend()
st.pyplot(fig)
