import streamlit as st
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🎯",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif !important;
  }

  /* Professional light background */
  .stApp {
    background: #f5f7fa;
    color: #1f2933;
  }

  h1, h2, h3 {
    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 700 !important;
    color: #1f2933;
  }

  /* Hero title */
  .hero-title {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #1f2933;
    margin-bottom: 0.2rem;
  }

  .hero-sub {
    font-size: 0.95rem;
    color: #6b6b85;
    font-weight: 400;
    margin-bottom: 2rem;
  }

  /* Input card */
  .input-card {
    background: #ffffff;
    border: 1px solid #e4e7eb;
    border-radius: 8px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
  }

  .input-card h4 {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1f2933;
    margin-bottom: 1rem;
  }

  /* Result card */
  .result-card {
    background: #ffffff;
    border: 1px solid #e4e7eb;
    border-radius: 8px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
  }

  .result-card.high-value {
    border-top: 4px solid #3ddc84;
  }
  .result-card.low-spending {
    border-top: 4px solid #ff6b6b;
  }
  .result-card.average {
    border-top: 4px solid #5b9cf6;
  }

  .result-label {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }

  .result-type {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 2rem;
    font-weight: 700;
  }

  .result-cluster {
    font-size: 0.85rem;
    color: #6b6b85;
    margin-top: 0.3rem;
  }

  .result-desc {
    font-size: 0.9rem;
    margin-top: 0.8rem;
    color: #1f2933;
    line-height: 1.6;
  }

  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.6rem;
    background: #f5f7fa !important;
    color: #1f2933 !important;
    border: 1px solid #e4e7eb;
  }

  .error-box {
    background: #fee2e2;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #b91c1c;
    font-size: 0.9rem;
  }

  /* Streamlit slider tweaks */
  [data-testid="stSlider"] > div > div > div > div {
    background: #2563eb !important;
  }

  /* Button */
  .stButton > button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.65rem 2rem !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
  }

  .stButton > button:hover {
    background: #1d4ed8 !important;
  }

  .divider {
    border: none;
    border-top: 1px solid #e4e7eb;
    margin: 1.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    errors = []
    model, scaler = None, None

    if not os.path.exists("kmeans_model.pkl"):
        errors.append("❌ `kmeans_model.pkl` not found in the working directory.")
    else:
        try:
            with open("kmeans_model.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            errors.append(f"❌ Failed to load `kmeans_model.pkl`: {e}")

    if not os.path.exists("scaler.pkl"):
        errors.append("❌ `scaler.pkl` not found in the working directory.")
    else:
        try:
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            errors.append(f"❌ Failed to load `scaler.pkl`: {e}")

    return model, scaler, errors


# ── Business logic ────────────────────────────────────────────────────────────
def get_customer_type(income: float, spending: float) -> str:
    if income > 70 and spending > 60:
        return "High Value"
    elif spending < 40:
        return "Low Spending"
    else:
        return "Average"


CUSTOMER_META = {
    "High Value": {
        "css_class": "high-value",
        "color": "#3ddc84",
        "icon": "💎",
        "desc": "High income with strong spending behaviour. Prime target for premium offers and loyalty programmes.",
        "badge_bg": "#1a3d2a",
        "badge_color": "#3ddc84",
    },
    "Low Spending": {
        "css_class": "low-spending",
        "color": "#ff6b6b",
        "icon": "💡",
        "desc": "Conservative spender regardless of income. Consider targeted promotions or re-engagement campaigns.",
        "badge_bg": "#3d1a1a",
        "badge_color": "#ff6b6b",
    },
    "Average": {
        "css_class": "average",
        "color": "#5b9cf6",
        "icon": "📊",
        "desc": "Mid-tier customer with balanced income and spending. Upsell opportunities through personalised recommendations.",
        "badge_bg": "#1a253d",
        "badge_color": "#5b9cf6",
    },
}


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Customer Segmentation System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">K-Means · 5 Clusters · StandardScaler</div>', unsafe_allow_html=True)

model, scaler, load_errors = load_artifacts()

if load_errors:
    for err in load_errors:
        st.markdown(f'<div class="error-box">{err}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem;color:#6b6b85;margin-top:1rem;'>
    Place <code>kmeans_model.pkl</code> and <code>scaler.pkl</code> in the same directory as <code>app.py</code>, then refresh.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Input sliders ─────────────────────────────────────────────────────────────
st.markdown('<div class="input-card"><h4>Customer Profile</h4>', unsafe_allow_html=True)

age = st.slider("Age", min_value=18, max_value=70, value=35, step=1)
income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
spending = st.slider("Spending Score (1–100)", min_value=1, max_value=100, value=50, step=1)

st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("Predict Segment")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    try:
        input_array = np.array([[age, income, spending]])          # strict column order
        input_scaled = scaler.transform(input_array)               # apply SAME scaler
        cluster = int(model.predict(input_scaled)[0])
        customer_type = get_customer_type(income, spending)
        meta = CUSTOMER_META[customer_type]

        st.markdown(f"""
        <div class="result-card {meta['css_class']}">
          <div class="result-label" style="color:{meta['color']};">Predicted Segment</div>
          <div class="result-type" style="color:{meta['color']};">{meta['icon']} {customer_type}</div>
          <div class="result-cluster">K-Means Cluster &nbsp;#{cluster}</div>
          <div class="result-desc">{meta['desc']}</div>
          <span class="badge" style="background:{meta['badge_bg']};color:{meta['badge_color']};">
            Age {age} &nbsp;·&nbsp; Income ${income}k &nbsp;·&nbsp; Score {spending}
          </span>
        </div>
        """, unsafe_allow_html=True)

    except ValueError as ve:
        st.markdown(f'<div class="error-box">⚠️ Invalid input: {ve}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-box">⚠️ Prediction failed: {e}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-size:0.75rem;color:#3a3b55;'>
  Features: Age · Annual Income · Spending Score &nbsp;|&nbsp; Algorithm: KMeans (k=5)
</div>
""", unsafe_allow_html=True)
