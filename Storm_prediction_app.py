#Importing librairies
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# Load the pre-trained model
# The model is a Random Forest model that predicts cyclone severity TD9636_STAGE
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to preprocess user input data before feeding it to the model
def preprocess_input(data):
    # Define the expected feature columns for the model
    columns = ['SEASON', 'LAT', 'LON', 'DIST2LAND', 'USA_WIND', 'STORM_SPEED', 'STORM_DIR',
               'BASIN_NI', 'BASIN_SI', 'BASIN_SP', 'BASIN_WP', 'NATURE_ET', 'NATURE_MX', 'NATURE_TS']
    # Convert input dictionary into a pandas DataFrame
    df = pd.DataFrame([data], columns=columns)
    
    #Treating the encoded features BASIN and NATURE
    # If the user selects BASIN_EP, set all other basins to False
    if data.get('BASIN_EP', False):
        df[['BASIN_NI', 'BASIN_SI', 'BASIN_SP', 'BASIN_WP']] = False
     # If the user selects NATURE_DS, set all other nature categories to False
    if data.get('NATURE_DS', False):
        df[['NATURE_ET', 'NATURE_MX', 'NATURE_TS']] = False
    
    return df

# Streamlit UI Config
st.set_page_config(page_title="Cyclone Severity Prediction", layout="wide")
st.title("🌪️ Cyclone Severity Prediction 🌪️")# Display the title in the center using HTML

# Layout with two columns:left for input, right for prediction output
col1, col2 = st.columns([1, 2])

# Left column: User input fields
with col1:
    st.subheader("🌍 Enter Geographical Data 🌍")
    season = st.number_input("Season", min_value=1900, max_value=2100, value=1980)
    lat = st.number_input("Latitude", value=-12.5)
    lon = st.number_input("Longitude", value=172.5)
    dist2land = st.number_input("Distance to Land (km)", value=647)
    usa_wind = st.number_input("Wind Speed (knots)", value=25.0)
    storm_speed = st.number_input("Storm Speed (knots)", value=6.0)
    storm_dir = st.number_input("Storm Direction (degrees)",  min_value=0.0, max_value=360.0,value=350.0)
    
    # Dropdown selection for Basin and Nature type
    st.subheader("Select Basin and Nature")
    available_basins = ['BASIN_EP', 'BASIN_NI', 'BASIN_SI', 'BASIN_SP', 'BASIN_WP']
    selected_basin = st.selectbox("Cyclone Basin", available_basins)
    available_nature = ['NATURE_DS', 'NATURE_ET', 'NATURE_MX', 'NATURE_TS']
    selected_nature = st.selectbox("Cyclone Nature", available_nature)
    
     # Button to trigger prediction
    predict_button = st.button("⚠️Predict Cyclone Severity⚠️")
    st.markdown("</div>", unsafe_allow_html=True)

# Right column: Display prediction and map
with col2:
    st.subheader("Prediction")
    if predict_button:
        # Prepare input data as a dictionary
        input_data = {
            'SEASON': season,
            'LAT': lat,
            'LON': lon,
            'DIST2LAND': dist2land,
            'USA_WIND': usa_wind,
            'STORM_SPEED': storm_speed,
            'STORM_DIR': storm_dir,
            'BASIN_NI': selected_basin == 'BASIN_NI',
            'BASIN_SI': selected_basin == 'BASIN_SI',
            'BASIN_SP': selected_basin == 'BASIN_SP',
            'BASIN_WP': selected_basin == 'BASIN_WP',
            'NATURE_ET': selected_nature == 'NATURE_ET',
            'NATURE_MX': selected_nature == 'NATURE_MX',
            'NATURE_TS': selected_nature == 'NATURE_TS',
            'BASIN_EP': selected_basin == 'BASIN_EP',
            'NATURE_DS': selected_nature == 'NATURE_DS'
        }
         # Preprocess the input before making a prediction
        input_df = preprocess_input(input_data)
        # Make a prediction using the loaded model
        prediction = model.predict(input_df)[0]
        # Display the predicted cyclone severity level
        st.success(f"⚠️Predicted Cyclone Severity⚠️: {prediction}")
        
        
        # Define severity color scale
        severity_colors = {
            0:"blue",
            1: "green",
            2: "yellow",
            3: "orange",
            4: "red",
            5: "darkred",
            6: "purple"
        }
        
        color = severity_colors.get(prediction, "gray")
        
        # Display cyclone location on a map
        st.subheader("Cyclone Location Map")
        map_data = pd.DataFrame({"lat": [lat], "lon": [lon], "severity": [prediction]})
        fig = px.scatter_geo(map_data, lat="lat", lon="lon", scope="world", 
                             title="Predicted Cyclone Location",
                             color_discrete_sequence=[color],
                             size_max=10)
        st.plotly_chart(fig, use_container_width=True)
