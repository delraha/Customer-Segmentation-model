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
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: "Times New Roman", Times, serif;
  }

  /* Dark card background */
  .stApp {
  background: #f5f7fa;   /* light neutral gray (professional) */
  color: #1f2933;        /* deep gray (better than black) */

  }

  h1, h2, h3 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-weight: 600;
    color: #1f2933;
}

  /* Hero title */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8e8f0 30%, #7c6af7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }

  .hero-sub {
    font-size: 0.95rem;
    color: #6b6b85;
    font-weight: 300;
    margin-bottom: 2rem;
  }

  /* Input card */
  .input-card {
    background: #16172080;
    border: 1px solid #2a2b3d;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
  }

  .input-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c6af7;
    margin-bottom: 1rem;
  }

  /* Result card */
  .result-card {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    border: 1px solid;
    animation: fadeUp 0.4s ease;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-card.high-value {
    background: linear-gradient(135deg, #1a2a1a, #0e1a0e);
    border-color: #3ddc84;
  }
  .result-card.low-spending {
    background: linear-gradient(135deg, #2a1a1a, #1a0e0e);
    border-color: #ff6b6b;
  }
  .result-card.average {
    background: linear-gradient(135deg, #1a1f2a, #0e1420);
    border-color: #5b9cf6;
  }

  .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }

  .result-type {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
  }

  .result-cluster {
    font-size: 0.85rem;
    opacity: 0.55;
    margin-top: 0.3rem;
  }

  .result-desc {
    font-size: 0.9rem;
    margin-top: 0.8rem;
    opacity: 0.75;
    line-height: 1.6;
  }

  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 0.6rem;
  }

  .error-box {
    background: #2a0e0e;
    border: 1px solid #8b2020;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #ff8080;
    font-size: 0.9rem;
  }

  /* Streamlit slider tweaks */
  [data-testid="stSlider"] > div > div > div > div {
    background: #7c6af7 !important;
  }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #7c6af7, #5b4dd4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
  }

  .stButton > button:hover {
    opacity: 0.88 !important;
  }

  .divider {
    border: none;
    border-top: 1px solid #2a2b3d;
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
