import streamlit as st
import pandas as pd
import pickle

def main():
    st.title(f":blue[WEATHER Classifier] :umbrella_with_rain_drops: :mostly_sunny: :snowflake:")
        
    temperature = st.sidebar.slider("**Temperature**", min_value=-25.0, max_value=110.0, step=0.5)
    humidity = st.sidebar.slider("**Humidity**", min_value=0.0, max_value=100.0, step=1.0)
    wind = st.sidebar.slider("**Wind Speed**", min_value=0.0, max_value=50.0, step=0.5)
    precipitation = st.sidebar.slider("**Precipitation (%)**", min_value=0.0, max_value=100.0, step=1.0)
    atmospheric_pressure = st.sidebar.slider("**Atmospheric Pressure**", min_value=800.0, max_value=1200.0, step=1.0)
    uv = st.sidebar.slider("**UV Index**", min_value=0.0, max_value=20.0, step=1.0)
    visibility = st.sidebar.slider("**Visibility (km)**", min_value=0.0, max_value=20.0, step=0.5)
    cloud_cover_clear = st.sidebar.selectbox("**Cloud Cover - clear**", (0,1))
    cloud_cover_cloudy = st.sidebar.selectbox("**Cloud Cover - cloudy**", (0,1))
    cloud_cover_overcast = st.sidebar.selectbox("**Cloud Cover - overcast**", (0,1))
    cloud_cover_partly_cloudy = st.sidebar.selectbox("**Cloud Cover - partly cloudy**", (0,1))
    season_autumn = st.sidebar.selectbox("**Season - Autumn**", (0,1))
    season_spring = st.sidebar.selectbox("**Season - Spring**", (0,1))
    season_summer = st.sidebar.selectbox("**Season - Summer**", (0,1))
    season_winter = st.sidebar.selectbox("**Season - Winter**", (0,1))
    location_coastal = st.sidebar.selectbox("**Location - Coastal**", (0,1))
    location_inland = st.sidebar.selectbox("**Location - Inland**", (0,1))
    location_mountain = st.sidebar.selectbox("**Location - Mountain**", (0,1))
    
    columns = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
               'Atmospheric Pressure', 'UV Index', 'Visibility (km)', 
               'Cloud Cover_clear', 'Cloud Cover_cloudy', 'Cloud Cover_overcast',
               'Cloud Cover_partly cloudy', 'Season_Autumn', 'Season_Spring',
               'Season_Summer', 'Season_Winter', 'Location_coastal', 'Location_inland',
               'Location_mountain']
    data = [[temperature, humidity, wind, precipitation, atmospheric_pressure, uv, visibility,
            cloud_cover_clear, cloud_cover_cloudy, cloud_cover_overcast, cloud_cover_partly_cloudy,
            season_autumn, season_spring, season_summer, season_winter,
            location_coastal, location_inland, location_mountain]]
    
    
    st.subheader("Original Data")
    df = pd.DataFrame(data=data, columns=columns)
    st.write(df)
    
    st.subheader("Scaled Data")
    scaler = pickle.load(open("weather-scaler.pkl", "rb"))
    features_to_scale = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", "Atmospheric Pressure", 
                        "UV Index", "Visibility (km)"]
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    st.write(df.head(3))
    
    st.subheader("Classification")
    model = pickle.load(open("weather-model.pkl", "rb"))
    predictions = model.predict(df)
    st.write(f":blue[**{predictions[0]}**]")
    st.image(f"{predictions[0]}.jpeg")
    
if __name__ == "__main__":
    st.set_page_config(
        page_title="Weather Classifier",
        page_icon="☔",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/raquelcolares',
            'Report a bug': "https://github.com/raquelcolares",
            'About': "# WEATHER Classifier. A Neural Network Classifier on Weather dataset"
            }
    )
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #80ced6;
        }
    </style>
    """, unsafe_allow_html=True)
    main()