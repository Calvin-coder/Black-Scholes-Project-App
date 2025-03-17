import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T == 0:  # Avoid division by zero for very small T
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Streamlit UI
st.title("Black Scholes Option Project")

# User inputs
S = st.number_input("Stock Price (S)", value=100.0, step=0.05)
K = st.number_input("Strike Price (K)", value=100.0, step=0.05)
T_months = st.number_input("Time to Expiration (T in Months)", value=12, step=1)
T = T_months / 12  # Convert months to years
r = st.number_input("Risk-Free Rate (r)", value=0.050, step=0.001, format="%.3f")
sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01, format="%.2f")
option_type = st.selectbox("Option Type", ["call", "put"])

# Calculate option price
if st.button("Calculate Price"):
    price = black_scholes(S, K, T, r, sigma, option_type)
    st.write(f"### {option_type.capitalize()} Option Price: ${price:.2f}")
    
    if option_type == "call":
        # Calculate Break-even Price for Call Option
        break_even_price = K + price
        st.write(f"### Break-even Stock Price (Call Option): ${break_even_price:.2f}")
    
    elif option_type == "put":
        # Calculate Break-even Price for Put Option
        break_even_price = K - price
        st.write(f"### Break-even Stock Price (Put Option): ${break_even_price:.2f}")

# Heatmap Configuration
st.subheader("Heatmap: Option Price vs. Volatility & Time (in Months)")

# Sliders to adjust heatmap ranges
min_time, max_time = st.slider("Select Time Range (Months)", 1, 36, (1, 24))
min_vol, max_vol = st.slider("Select Volatility Range", 0.05, 1.0, (0.1, 0.5))

time_range_months = np.linspace(min_time, max_time, 20)  # Time from user-selected range
volatility_range = np.linspace(min_vol, max_vol, 20)  # Volatility from user-selected range
price_matrix = np.zeros((len(volatility_range), len(time_range_months)))

for i, sigma_val in enumerate(volatility_range):
    for j, T_val_months in enumerate(time_range_months):
        T_val = T_val_months / 12  # Convert months to years
        price_matrix[i, j] = black_scholes(S, K, T_val, r, sigma_val, option_type)

# Increase the figure size further to make the squares bigger
fig, ax = plt.subplots(figsize=(12, 10))  # Larger figure size for better readability

# Create the heatmap with annotations
sns.heatmap(price_matrix, xticklabels=np.round(time_range_months, 0), 
            yticklabels=np.round(volatility_range, 2), cmap="coolwarm", 
            annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=0.7, ax=ax)

# Adding labels and title
ax.set_xlabel("Time to Expiration (Months)")
ax.set_ylabel("Volatility (σ)")
ax.set_title(f"Option Price Heatmap ({option_type.capitalize()} Option)")

# Show the heatmap
st.pyplot(fig)
