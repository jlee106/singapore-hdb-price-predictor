import numpy as np
import pandas as pd
import joblib
import streamlit as st
import qrcode
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

@st.cache_resource
def load_model_from_url():
    file_id = "1d2fdEIT7gNgsLAiFoKgFf7Zo7ujDvJ73"
    destination = "model.pkl"
    download_file_from_google_drive(file_id, destination)
    
    size = os.path.getsize(destination)
    st.write(f"Downloaded model.pkl size: {size} bytes")
    
    with open(destination, "rb") as f:
        start = f.read(100)
    st.write(f"Start of file bytes: {start[:20]}")
    
    return joblib.load(destination)

rf = load_model_from_url()

APP_URL = "https://hdb-price-predictor-team-2.streamlit.app"

def make_qr(data: str):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

qr_img = make_qr(APP_URL)

buf = BytesIO()
qr_img.save(buf, format="PNG")
buf.seek(0)

st.download_button(
    label="Download this QR code",
    data=buf,
    file_name="app_qrcode.png",
    mime="image/png"
)

st.image(buf, caption=APP_URL, use_column_width=False)

hdb = pd.read_csv('./hdb_data_clean.csv')

towns = hdb["town"].sort_values().unique().tolist()

st.title("HDB Price Predictor")
town = st.selectbox("Town", towns)

flat_types = hdb["flat_type"].sort_values().unique().tolist()
flat_type = st.selectbox("Flat Type", flat_types)

mid_storey = st.slider("Storey", 1, 50, 8)
floor_area_sqft = st.slider("Floor Area (sqft)", 300.00, 3100.00, 1022.58)
hdb_age = st.slider("Property Age", 2, 65, 29)  # max 65 coz 1st hdb was built 1960
total_dwelling_units = st.slider("Total Dwelling Units", 2, 570, 112)
mrt_nearest_distance = st.slider("Nearest MRT Distance(m)", 0.00, 3600.00, 682.62)

towns.pop(0)
towns_full = []

for t in towns:
    towns_full.append(f'town_{t}')

town_list = []
for t in towns:
    town_list.append(False)

town_dict = dict(zip(towns_full, town_list))

if town != "ANG MO KIO":
    town_dict[f'town_{town}'] = True

flat_types.pop(0)

flat_types_full = []

for ft in flat_types:
    flat_types_full.append(f'flat_type_{ft}')

flat_types_list = []

for ft in flat_types:
    flat_types_list.append(False)

flat_types_dict = dict(zip(flat_types_full, flat_types_list))

if flat_type != "1 ROOM":
    flat_types_dict[f'flat_type_{flat_type}'] = True

mid_storey = {"mid_storey": mid_storey}
floor_area_sqft = {"floor_area_sqft": floor_area_sqft}
hdb_age = {"hdb_age": hdb_age}
total_dwelling_units = {"total_dwelling_units": total_dwelling_units}
mrt_nearest_distance = {"mrt_nearest_distance": mrt_nearest_distance}

merged = mid_storey | floor_area_sqft | hdb_age | total_dwelling_units | mrt_nearest_distance | town_dict | flat_types_dict

inputs = pd.DataFrame(merged, index=[0])

if st.button("predict"):
    predict = rf.predict(inputs)
    st.success(f'Predicted price: ${predict[0]:,.2f}')
