import numpy as np
import pandas as pd
import joblib
import streamlit as st
import qrcode
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
import requests

@st.cache_resource
def load_model_from_url():
    url = "https://huggingface.co/jlee106/hdb-price-model/resolve/main/random_forest_model.pkl"
    response = requests.get(url)
    with open("model.pkl", "wb") as f:
        f.write(response.content)
    return joblib.load("model.pkl")

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

st.image(buf, caption=APP_URL, use_container_width=False)

hdb = pd.read_csv('./hdb_data_clean.csv')

towns = hdb["town"].sort_values().unique().tolist()

st.title("HDB Price Predictor")
select_town = st.selectbox("Town", towns)

flat_types = hdb["flat_type"].sort_values().unique().tolist()
select_flat_type = st.selectbox("Flat Type", flat_types)

mid_storey = st.slider("Storey", 1, 50, 8)
floor_area_sqft = st.slider("Floor Area (sqft)", 300.00, 3100.00, 1022.58)
hdb_age = st.slider("Property Age", 2, 65, 29) #max 65 coz 1st hdb was built 1960
total_dwelling_units = st.slider("Total Dwelling Units", 2, 570, 112)
mrt_nearest_distance = st.slider("Nearest MRT Distance(m)", 0.00, 3600.00, 682.62)

towns.pop(0)
towns_full = []

for town in towns:
    towns_full.append(f'town_{town}')

town_list = []
for town in towns:
    town_list.append(False)

town_dict = dict(zip(towns_full, town_list))

if select_town != "ANG MO KIO":
    town_dict[f'town_{select_town}'] = True

flat_types.pop(0)

flat_types_full = []

for flat_type in flat_types:
    flat_types_full.append(f'flat_type_{flat_type}')

flat_types_list = []

for flat_type in flat_types:
    flat_types_list.append(False)

flat_types_dict = dict(zip(flat_types_full, flat_types_list))

if select_flat_type != "1 ROOM":
    flat_types_dict[f'flat_type_{select_flat_type}'] = True

mid_storey = {"mid_storey": mid_storey}
floor_area_sqft = {"floor_area_sqft": floor_area_sqft}
hdb_age = {"hdb_age": hdb_age}
total_dwelling_units = {"total_dwelling_units": total_dwelling_units}
mrt_nearest_distance = {"mrt_nearest_distance": mrt_nearest_distance}

merged = mid_storey | floor_area_sqft | hdb_age | total_dwelling_units | mrt_nearest_distance | town_dict | flat_types_dict

inputs = pd.DataFrame(merged, index = [0])

if st.button("Predict"):
    predict = rf.predict(inputs)
    st.success(f'Predicted price: ${predict[0]:,.2f}')
