import streamlit as st
import pandas as pd
import joblib
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

#  Page configuration
st.set_page_config(
    page_title="Flood Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        h1, h2, h3, h4 {margin-top: 0.5rem; margin-bottom: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load model & encoders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

rf_model       = joblib.load(MODEL_DIR / "flood_risk_rf_model.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
target_encoder = joblib.load(MODEL_DIR / "target_encoder.pkl")
FEATURE_ORDER  = list(rf_model.feature_names_in_)

def encode_input(df: pd.DataFrame) -> pd.DataFrame:
    for col, le in label_encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df

# Email utilities 
def mask_email(email: str) -> str:
    """Return masked email like k*****e@gmail.com."""
    user, _, domain = email.partition("@")
    if len(user) <= 2:
        return "*" * len(user) + "@" + domain
    return user[0] + "*" * (len(user) - 2) + user[-1] + "@" + domain

def send_alert_email(sender_email, sender_pass, recipient_email, subject, body):
    """Send email via Gmail SMTP with masked confirmation."""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        # Mask recipient in the success message
        st.success(f" Email sent to {mask_email(recipient_email)}")
    except Exception as e:
        st.error(f" Email send error: {e}")
        st.stop()

#  Sidebar – hidden email inputs
st.sidebar.header(" Email Settings")
# type="password" hides the text while the user types
sender_email    = st.sidebar.text_input("Sender Gmail", type="password")
app_password    = st.sidebar.text_input("App Password (16-digit)", type="password")
recipient_email = st.sidebar.text_input("Recipient Email", type="password")

# Main layout 
st.title(" Flood Risk Prediction & Email Alert")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    latitude        = st.number_input("Latitude", value=25.0)
    longitude       = st.number_input("Longitude", value=80.0)
    rainfall_mm     = st.number_input("Rainfall (mm)", value=700.0)
    temperature     = st.number_input("Temperature (°C)", value=28.0)
    humidity        = st.number_input("Humidity (%)", value=95.0)
    river_discharge = st.number_input("River Discharge (m³/s)", value=8000.0)

with col2:
    water_level        = st.number_input("Water Level (m)", value=12.0)
    elevation          = st.number_input("Elevation (m)", value=20.0)
    population_density = st.number_input("Population Density", value=3000.0)
    infrastructure     = st.number_input("Infrastructure Score (0–1)", value=0.0)
    historical_floods  = st.number_input("Historical Flood Count", value=8)

with col3:
    flood_occurred  = st.number_input("Flood Occurred (0 or 1)", value=1)
    year            = st.number_input("Year", value=2024)
    monsoon         = st.number_input("Monsoon Jun–Sep (0 or 1)", value=1)
    annual_rainfall = st.number_input("Annual Rainfall (mm)", value=3500.0)
    land_cover      = st.selectbox("Land Cover", ["Urban", "Rural", "Forest", "Agriculture"])
    soil_type       = st.selectbox("Soil Type",  ["Clay", "Sandy", "Loam", "Silt"])

# Prediction 
if st.button("Predict Flood Risk", use_container_width=True):
    input_data = {
        "latitude": latitude,
        "longitude": longitude,
        "rainfall_mm": rainfall_mm,
        "temperature_°c": temperature,
        "humidity_": humidity,
        "river_discharge_m³s": river_discharge,
        "water_level_m": water_level,
        "elevation_m": elevation,
        "land_cover": land_cover,
        "soil_type": soil_type,
        "population_density": population_density,
        "infrastructure": infrastructure,
        "historical_floods": historical_floods,
        "flood_occurred": flood_occurred,
        "year": year,
        "monsoon_jun_sep": monsoon,
        "annual_rainfall": annual_rainfall,
    }

    df = pd.DataFrame([input_data])
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_ORDER]
    df = encode_input(df)

    prediction = rf_model.predict(df)
    risk_level = target_encoder.inverse_transform(prediction)[0]

    if risk_level == "High":
        alert_message = (
            " HIGH Flood Risk!\n"
            "Immediate action required: Evacuate low-lying areas, secure valuables, "
            "and stay tuned to official warnings."
        )
    elif risk_level == "Medium":
        alert_message = (
            " MEDIUM Flood Risk.\n"
            "Remain alert, monitor weather updates, and prepare an emergency kit "
            "in case conditions worsen."
        )
    else:
        alert_message = "LOW Flood Risk.\nNo immediate danger detected."

    st.session_state["last_prediction"] = (risk_level, input_data, alert_message)

# Always show last prediction
if "last_prediction" in st.session_state:
    risk_level, input_data, alert_message = st.session_state["last_prediction"]

    if risk_level == "High":
        st.error(alert_message)
    elif risk_level == "Medium":
        st.warning(alert_message)
    else:
        st.success(alert_message)

    if st.button("Send Alert Email Now", use_container_width=True):
        if not (sender_email and app_password and recipient_email):
            st.error("Please fill all email fields in the sidebar.")
        else:
            subject = f" Flood Risk Alert: {risk_level.upper()}"
            body = f"{alert_message}\n\nDetails of Prediction:\n"
            body += "\n".join([f"{k}: {v}" for k, v in input_data.items()])
            send_alert_email(sender_email, app_password, recipient_email, subject, body)
