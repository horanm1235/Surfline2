"""
ULTIMATE SURF FORECAST V3.1 - PROFESSIONAL ENHANCED VERSION
=======================================================
Improved UI/UX to resemble Surfline with better features.
Direct live cam embeds (YouTube streams or images).
Satellite maps with wind overlays.
Smoother tide charts with spline interpolation.
Enhanced styling and layouts.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass, asdict

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    background-color: #f0f4f8;
    color: #333;
}

.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e40af;
    text-align: center;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.stMetric {
    background-color: white;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.stMetric label {
    font-weight: 700;
    color: #1e40af;
}

.stMetric value {
    font-size: 1.5rem;
}

.stTabs > div > button {
    background-color: #dbeafe;
    color: #1e40af;
    border: none;
    padding: 10px 20px;
    border-radius: 4px 4px 0 0;
}

.stTabs > div > button:hover {
    background-color: #bfdbfe;
}

.sidebar .stSelectbox, .sidebar .stSlider, .sidebar .stCheckbox {
    background-color: white;
    border-radius: 4px;
    padding: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="🌊 Ultimate Surf Forecast",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "Ultimate Surf Forecast - Professional Surf Forecasting Platform"
    }
)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# ==================== HELPER FUNCTIONS ====================

def safe_extract_float(value, multiplier: float = 1.0, offset: float = 0.0, default=None):
    try:
        if value is None or value == 'MM' or value == '':
            return default
        converted = float(value) * multiplier + offset
        return converted
    except (ValueError, TypeError):
        return default

# ==================== DATA MODELS ====================

@dataclass
class Beach:
    name: str
    buoy: str
    backup_buoys: List[str]
    lat: float
    lon: float
    station_id: str  # For tides
    break_type: str
    bottom: str
    exposure: str
    best_swell: Dict[str, float]
    best_wind: Dict[str, float]
    best_tide: Dict[str, Any]
    optimal_size: Dict[str, float]
    crowd: str
    skill_level: str
    hazards: List[str]
    best_season: List[str]
    parking: str
    facilities: List[str]
    notes: str
    cam_urls: List[str] = None  # Surf cam URLs (YouTube embed or image)
    
    def __post_init__(self):
        if self.cam_urls is None:
            self.cam_urls = []

# ==================== EXPANDED BEACH DATABASE ====================

BEACHES = {
    "Rockaway Beach, NY": Beach(
        name="Rockaway Beach, NY",
        buoy="44065",
        backup_buoys=["44025", "44017"],
        lat=40.5833,
        lon=-73.8167,
        station_id="8531680",
        break_type="Beach Break",
        bottom="Sand",
        exposure="Open Ocean",
        best_swell={"min": 135, "max": 225, "optimal": 180},
        best_wind={"min": 315, "max": 45, "optimal": 0},
        best_tide={"phase": "mid_to_high", "range": [2, 5]},
        optimal_size={"min": 2, "max": 6},
        crowd="Heavy",
        skill_level="Beginner to Intermediate",
        hazards=["Strong rip currents", "Crowds", "Jetty on west end"],
        best_season=["Fall", "Winter"],
        parking="Street parking, paid lots",
        facilities=["Bathrooms", "Showers", "Food"],
        notes="NYC's premier surf spot. Can get crowded. Best on NW winds.",
        cam_urls=["https://www.youtube.com/embed/ge2TsXglKgQ"]
    ),
    "Long Beach, NY": Beach(
        name="Long Beach, NY",
        buoy="44065",
        backup_buoys=["44025", "44017"],
        lat=40.5892,
        lon=-73.6579,
        station_id="8510560",
        break_type="Beach Break",
        bottom="Sand",
        exposure="Open Ocean",
        best_swell={"min": 90, "max": 180, "optimal": 135},
        best_wind={"min": 270, "max": 360, "optimal": 315},
        best_tide={"phase": "all_tides", "range": [0, 6]},
        optimal_size={"min": 2, "max": 8},
        crowd="Moderate to Heavy",
        skill_level="Beginner to Advanced",
        hazards=["Jetties", "Strong currents"],
        best_season=["Fall", "Winter", "Spring"],
        parking="Paid parking lots",
        facilities=["Bathrooms", "Boardwalk", "Food"],
        notes="Consistent surf with multiple peaks. Watch the jetties.",
        cam_urls=["https://www.youtube.com/embed/GTMX34w5Gxc"]
    ),
    "Montauk Point, NY": Beach(
        name="Montauk Point, NY",
        buoy="44017",
        backup_buoys=["44025", "44065"],
        lat=41.0715,
        lon=-71.8569,
        station_id="8510560",
        break_type="Point Break",
        bottom="Rock/Sand",
        exposure="Open Ocean",
        best_swell={"min": 45, "max": 135, "optimal": 90},
        best_wind={"min": 225, "max": 315, "optimal": 270},
        best_tide={"phase": "low_to_mid", "range": [0, 3]},
        optimal_size={"min": 3, "max": 10},
        crowd="Light to Moderate",
        skill_level="Intermediate to Advanced",
        hazards=["Rocks", "Strong currents", "Sharks"],
        best_season=["Fall", "Winter"],
        parking="Free parking",
        facilities=["Bathrooms", "Lighthouse"],
        notes="World-class waves when it's on. Not for beginners.",
        cam_urls=["https://www.youtube.com/embed/Oru2l3Hzh2A"]
    ),
    "Jones Beach, NY": Beach(
        name="Jones Beach, NY",
        buoy="44065",
        backup_buoys=["44025", "44017"],
        lat=40.5900,
        lon=-73.5500,
        station_id="8516385",
        break_type="Beach Break",
        bottom="Sand",
        exposure="Open Ocean",
        best_swell={"min": 135, "max": 225, "optimal": 180},
        best_wind={"min": 315, "max": 45, "optimal": 0},
        best_tide={"phase": "mid_to_high", "range": [2, 5]},
        optimal_size={"min": 2, "max": 6},
        crowd="Moderate",
        skill_level="Beginner to Intermediate",
        hazards=["Rip currents", "Crowds"],
        best_season=["Fall", "Winter"],
        parking="Paid lots",
        facilities=["Bathrooms", "Food"],
        notes="Popular spot with consistent waves.",
        cam_urls=["https://www.youtube.com/embed/7shCjhL0kcA"]
    ),
    "Gilgo Beach, NY": Beach(
        name="Gilgo Beach, NY",
        buoy="44065",
        backup_buoys=["44025", "44017"],
        lat=40.6180,
        lon=-73.3980,
        station_id="8516385",
        break_type="Beach Break",
        bottom="Sand",
        exposure="Open Ocean",
        best_swell={"min": 135, "max": 225, "optimal": 180},
        best_wind={"min": 315, "max": 45, "optimal": 0},
        best_tide={"phase": "mid_to_high", "range": [2, 5]},
        optimal_size={"min": 2, "max": 6},
        crowd="Light",
        skill_level="Intermediate",
        hazards=["Remote access", "Strong currents"],
        best_season=["Fall", "Winter"],
        parking="Permit required",
        facilities=["Limited"],
        notes="Less crowded alternative to Jones.",
        cam_urls=["https://gilgo.com/cam/surfcam.jpg"]
    ),
    "Manasquan Inlet, NJ": Beach(
        name="Manasquan Inlet, NJ",
        buoy="44025",
        backup_buoys=["44065", "44091"],
        lat=40.1010,
        lon=-74.0350,
        station_id="8532594",
        break_type="Jetty Break",
        bottom="Sand/Rock",
        exposure="Open Ocean",
        best_swell={"min": 90, "max": 180, "optimal": 135},
        best_wind={"min": 270, "max": 360, "optimal": 315},
        best_tide={"phase": "low_to_mid", "range": [0, 4]},
        optimal_size={"min": 3, "max": 8},
        crowd="Heavy",
        skill_level="Advanced",
        hazards=["Jetty", "Strong currents", "Crowds"],
        best_season=["Fall", "Winter"],
        parking="Street parking",
        facilities=["Bathrooms"],
        notes="Classic NJ spot with powerful waves.",
        cam_urls=["https://www.youtube.com/embed/71unjYWqD7A"]
    ),
}

BEACHES_DICT = {k: asdict(v) for k, v in BEACHES.items()}

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=900)
def get_buoy_data_with_fallback(buoy_id: str, backup_buoys: List[str] = None) -> Tuple[Optional[Dict], str]:
    buoys_to_try = [buoy_id] + (backup_buoys or [])
    
    for buoy in buoys_to_try:
        try:
            url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy}.txt"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                if len(lines) >= 3:
                    headers = lines[0].split()
                    data = lines[2].split()
                    
                    buoy_dict = {'buoy_used': buoy}
                    for i, header in enumerate(headers):
                        if i < len(data):
                            try:
                                buoy_dict[header] = float(data[i]) if data[i] != 'MM' else None
                            except ValueError:
                                buoy_dict[header] = data[i]
                    
                    return buoy_dict, buoy
        except Exception:
            continue
    
    return None, None

@st.cache_data(ttl=3600)
def get_marine_forecast(lat: float, lon: float, days: int = 7) -> Optional[Dict]:
    try:
        url = "https://marine-api.open-meteo.com/v1/marine"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'wave_height,wave_direction,wave_period,wind_wave_height,wind_wave_direction,wind_wave_period,swell_wave_height,swell_wave_direction,swell_wave_period',
            'daily': 'wave_height_max,wave_direction_dominant,wave_period_max',
            'timezone': 'America/New_York',
            'forecast_days': min(days, 7)
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_weather_forecast(lat: float, lon: float, days: int = 7) -> Optional[Dict]:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,wind_gusts_10m',
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset',
            'timezone': 'America/New_York',
            'forecast_days': days
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_tide_predictions(station_id: str, days: int = 7) -> List[Dict]:
    today = datetime.now().date()
    begin_date = today.strftime('%Y%m%d')
    end_date = (today + timedelta(days=days)).strftime('%Y%m%d')
    url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={begin_date}&end_date={end_date}&station={station_id}&product=predictions&datum=MLLW&time_zone=lst_ldt&interval=hilo&format=json&units=english&application=web_services"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('predictions', [])
        return []
    except Exception:
        return []

# ==================== ANALYSIS FUNCTIONS ====================

def compute_surf_score(beach: Dict, current: Dict) -> float:
    score = 0.0
    
    # Wave height (30%)
    height = current.get('WVHT', 0) * 3.28084 if current.get('WVHT') else 0
    opt_min = beach['optimal_size']['min']
    opt_max = beach['optimal_size']['max']
    if opt_min <= height <= opt_max:
        score += 30
    elif height > opt_max:
        score += 10
    elif height > 0:
        score += 15
    
    # Period (20%)
    period = current.get('DPD', 0) or 0
    if period > 10:
        score += 20
    elif period > 6:
        score += 10
    
    # Swell direction (25%)
    swell_dir = current.get('MWD', 0) or 0
    best_swell = beach['best_swell']['optimal']
    delta = min(abs(swell_dir - best_swell), 360 - abs(swell_dir - best_swell))
    if delta < 30:
        score += 25
    elif delta < 60:
        score += 15
    
    # Wind (25%)
    wind_dir = current.get('WDIR', 0) or 0
    best_wind = beach['best_wind']['optimal']
    delta_wind = min(abs(wind_dir - best_wind), 360 - abs(wind_dir - best_wind))
    if delta_wind < 45:
        score += 25
    else:
        score -= 10  # Onshore penalty
    
    # Normalize to 5 stars
    final_score = min(max(score / 100 * 5, 1), 5)
    return round(final_score, 1)

def generate_condition_analysis(score: float, current: Dict, beach: Dict) -> str:
    analysis = []
    
    height_ft = safe_extract_float(current.get('WVHT'), 3.28084, default=0)
    period = safe_extract_float(current.get('DPD'), default=0)
    wind_speed = safe_extract_float(current.get('WSPD'), 2.23694, default=0)
    wind_dir = safe_extract_float(current.get('WDIR'), default=0)
    swell_dir = safe_extract_float(current.get('MWD'), default=0)
    
    if score >= 4:
        analysis.append("Excellent conditions! Get out there.")
    elif score >= 3:
        analysis.append("Good surf potential. Worth checking.")
    elif score >= 2:
        analysis.append("Fair conditions. Might be fun for practice.")
    else:
        analysis.append("Poor conditions. Consider waiting for better swell.")
    
    wind_type = "offshore" if abs(wind_dir - beach['best_wind']['optimal']) < 45 else "onshore"
    analysis.append(f"Winds are {wind_type} at {wind_speed:.0f} mph.")
    
    analysis.append(f"Swell from {swell_dir:.0f}° at {height_ft:.1f} ft / {period:.0f} sec.")
    
    return " ".join(analysis)

# ==================== VISUALIZATIONS ====================

def create_forecast_charts(marine_data: Dict, weather_data: Dict):
    if not marine_data or not weather_data:
        return None
    
    # Hourly data
    times = pd.to_datetime(marine_data['hourly']['time'])
    wave_height = marine_data['hourly']['wave_height']
    wave_period = marine_data['hourly']['wave_period']
    wind_speed = weather_data['hourly']['wind_speed_10m']
    wind_dir = weather_data['hourly']['wind_direction_10m']
    temp = weather_data['hourly']['temperature_2m']
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=('Wave Height & Period', 'Wind Speed & Direction', 'Air Temperature', 'Precipitation'))
    
    # Wave
    fig.add_trace(go.Scatter(x=times, y=wave_height, name='Wave Height (m)', line=dict(color='#3b82f6', width=2, shape='spline')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=wave_period, name='Period (s)', line=dict(color='#22c55e', width=2, shape='spline'), yaxis='y2'), row=1, col=1)
    
    # Wind
    fig.add_trace(go.Scatter(x=times, y=wind_speed, name='Wind Speed (km/h)', line=dict(color='#a855f7', width=2, shape='spline')), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=wind_dir, name='Wind Dir (°)', mode='markers', marker=dict(color='#f59e0b', size=6)), row=2, col=1)
    
    # Temp
    fig.add_trace(go.Scatter(x=times, y=temp, name='Temp (°C)', line=dict(color='#ef4444', width=2, shape='spline')), row=3, col=1)
    
    # Precip
    precip = weather_data['hourly']['precipitation']
    fig.add_trace(go.Bar(x=times, y=precip, name='Precip (mm)', marker_color='#93c5fd'), row=4, col=1)
    
    fig.update_layout(
        height=800, 
        showlegend=True, 
        hovermode='x unified',
        template='plotly_white',
        font=dict(family='Roboto', color='#333')
    )
    return fig

def create_tide_chart(tides: List[Dict]):
    if not tides:
        return None
    
    df = pd.DataFrame(tides)
    df['t'] = pd.to_datetime(df['t'])
    df['v'] = df['v'].astype(float)
    
    fig = go.Figure()
    colors = ['#ef4444' if h == 'H' else '#3b82f6' for h in df['type']]
    fig.add_trace(go.Scatter(
        x=df['t'], 
        y=df['v'], 
        mode='lines+markers',
        line=dict(shape='spline', width=3, color='#1d4ed8'),
        marker=dict(color=colors, size=8)
    ))
    
    fig.update_layout(
        title='Tide Predictions (Smooth Curve)',
        xaxis_title='Time',
        yaxis_title='Height (ft)',
        hovermode='x unified',
        template='plotly_white',
        font=dict(family='Roboto', color='#333')
    )
    return fig

def create_interactive_map(beaches_dict: Dict[str, Dict], current_conditions: Dict[str, Dict]):
    """Enhanced satellite map with color-coded markers and wind arrows"""
    map_html = """
    <div id="map" style="width: 100%; height: 600px;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([40.5, -73.5], 9);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri'
        }).addTo(map);
        
        var beaches = """ + json.dumps(list(beaches_dict.values())) + """;
        var conditions = """ + json.dumps(current_conditions) + """;
        
        beaches.forEach(function(beach) {
            var cond = conditions[beach.name] || {};
            var score = cond.score || 0;
            var color = score >= 4 ? 'green' : score >= 3 ? 'orange' : 'red';
            
            L.circleMarker([beach.lat, beach.lon], {
                color: color,
                radius: 10,
                fillOpacity: 0.8
            }).addTo(map);
            
            if (cond.wind_dir) {
                var arrowIcon = L.divIcon({
                    className: 'wind-arrow',
                    html: '<div style="font-size: 24px; color: white; text-shadow: 0 0 3px black; transform: rotate(' + cond.wind_dir + 'deg);">➤</div>'
                });
                L.marker([beach.lat + 0.01, beach.lon + 0.01], {icon: arrowIcon}).addTo(map);
            }
            
            var popup = "<b>" + beach.name + "</b><br>" +
                        "Score: " + score.toFixed(1) + " ⭐<br>" +
                        "Waves: " + (cond.height || 'N/A') + " ft<br>" +
                        "Wind: " + (cond.wind_speed || 'N/A') + " mph " + (cond.wind_dir || '') + "°<br>" +
                        "Swell: " + (cond.swell_dir || '') + "°";
            L.marker([beach.lat, beach.lon]).bindPopup(popup).addTo(map);
        });
    </script>
    """
    return map_html

def create_beach_satellite_map(lat: float, lon: float, wind_dir: float = None, wind_speed: float = None):
    """Small satellite map for individual beach with wind overlay"""
    map_html = """
    <div id="beach_map" style="width: 100%; height: 300px;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('beach_map').setView([""" + str(lat) + """, """ + str(lon) + """], 14);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri'
        }).addTo(map);
        
        """ + (f"""
        var arrowIcon = L.divIcon({{
            html: '<div style="font-size: 30px; color: #ef4444; transform: rotate({wind_dir}deg);">➤</div><div style="text-align: center; color: white; font-weight: bold;">{wind_speed:.0f} mph</div>'
        }});
        L.marker([{lat}, {lon}], {{icon: arrowIcon}}).addTo(map);
        """ if wind_dir is not None and wind_speed is not None else "") + """
    </script>
    """
    return map_html

# ==================== INTERNATIONALIZATION ====================

TRANSLATIONS = {
    'en': {
        'title': 'ULTIMATE SURF FORECAST',
        'subtitle': 'Professional Surf Intel for NY/NJ Spots',
        'beach_selection': 'Spot Selection',
        'current_conditions': 'Current Conditions',
        'wave_height': 'Wave Height',
        'wave_period': 'Wave Period',
        'wind_speed': 'Wind Speed',
        'wind_dir': 'Wind Dir',
        'swell_dir': 'Swell Dir',
        'water_temp': 'Water Temp',
        'surf_score': 'Surf Score',
        'wave_energy': 'Wave Energy',
        'forecast': 'Forecast',
        'tides': 'Tides',
        'cams': 'Live Cams',
        'analysis': 'Analysis',
        'map': 'Map',
    },
    'es': {
        'title': 'PRONÓSTICO ULTIMATE DE SURF',
        'subtitle': 'Inteligencia Avanzada de Surf para NY/NJ',
        'beach_selection': 'Selección de Spot',
        'current_conditions': 'Condiciones Actuales',
        'wave_height': 'Altura de Ola',
        'wave_period': 'Período de Ola',
        'wind_speed': 'Velocidad del Viento',
        'wind_dir': 'Dir Viento',
        'swell_dir': 'Dir Swell',
        'water_temp': 'Temp. del Agua',
        'surf_score': 'Puntaje Surf',
        'wave_energy': 'Energía Ola',
        'forecast': 'Pronóstico',
        'tides': 'Mareas',
        'cams': 'Cámaras en Vivo',
        'analysis': 'Análisis',
        'map': 'Mapa',
    }
}

def t(key: str) -> str:
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# ==================== MAIN APPLICATION ====================

def main():
    # Header
    st.markdown('<div class="main-header">🌊 ' + t('title') + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">' + t('subtitle') + '</div>', unsafe_allow_html=True)
    
    col_lang, col_theme = st.columns([1,1])
    with col_lang:
        lang = st.selectbox("🌐 Language", ["en", "es"], index=0 if st.session_state.language == 'en' else 1)
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
    with col_theme:
        if st.button("🌓 Theme"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.rerun()
    
    # Navigation tabs
    tabs = st.tabs([
        t('forecast'),
        t('map'),
        t('cams')
    ])
    
    # Pre-fetch conditions for map
    current_conditions = {}
    for name, beach in BEACHES.items():
        buoy_data, _ = get_buoy_data_with_fallback(beach.buoy, beach.backup_buoys)
        if buoy_data:
            score = compute_surf_score(asdict(beach), buoy_data)
            current_conditions[name] = {
                'score': score,
                'height': safe_extract_float(buoy_data.get('WVHT'), 3.28084),
                'wind_speed': safe_extract_float(buoy_data.get('WSPD'), 2.23694),
                'wind_dir': safe_extract_float(buoy_data.get('WDIR')),
                'swell_dir': safe_extract_float(buoy_data.get('MWD'))
            }
    
    # Tab 1: Forecast
    with tabs[0]:
        # Sidebar
        with st.sidebar:
            st.markdown("## 🏄 " + t('beach_selection'))
            selected_beach = st.selectbox(
                "Choose Your Spot",
                options=list(BEACHES.keys()),
                index=0
            )
            
            st.markdown("---")
            st.markdown("## ⚙️ Options")
            forecast_days = st.slider("Forecast Days", 1, 7, 3)
            show_charts = st.checkbox("📊 Charts", value=True)
            show_analysis = st.checkbox("🔍 Analysis", value=True)
        
        # Main content
        beach = BEACHES[selected_beach]
        beach_info = asdict(beach)
        
        st.markdown(f"### 🏖️ {selected_beach}")
        st.write(f"{beach_info['break_type']} • {beach_info['bottom']} • {beach_info['skill_level']}")
        
        # Fetch data
        buoy_data, buoy_used = get_buoy_data_with_fallback(beach.buoy, beach.backup_buoys)
        
        if not buoy_data:
            st.error("Unable to fetch buoy data. Try later.")
            return
        
        # Satellite map with wind
        st.markdown("## 🛰️ Beach Overview")
        wind_dir = safe_extract_float(buoy_data.get('WDIR'))
        wind_speed = safe_extract_float(buoy_data.get('WSPD'), 2.23694)
        map_html_beach = create_beach_satellite_map(beach.lat, beach.lon, wind_dir, wind_speed)
        st.components.v1.html(map_html_beach, height=300)
        
        # Current conditions
        st.markdown("## 🎯 " + t('current_conditions'))
        
        wave_height_ft = safe_extract_float(buoy_data.get('WVHT'), 3.28084)
        wave_period = safe_extract_float(buoy_data.get('DPD'))
        wind_speed_mph = safe_extract_float(buoy_data.get('WSPD'), 2.23694)
        wind_dir = safe_extract_float(buoy_data.get('WDIR'))
        swell_dir = safe_extract_float(buoy_data.get('MWD'))
        water_temp_f = safe_extract_float(buoy_data.get('WTMP'), 1.8, 32)
        wave_energy = (wave_height_ft ** 2 * wave_period) if wave_height_ft and wave_period else None
        surf_score = compute_surf_score(beach_info, buoy_data)
        
        cols = st.columns(8)
        cols[0].metric(t('wave_height'), f"{wave_height_ft:.1f} ft" if wave_height_ft else "N/A")
        cols[1].metric(t('wave_period'), f"{wave_period:.0f} sec" if wave_period else "N/A")
        cols[2].metric(t('wind_speed'), f"{wind_speed_mph:.0f} mph" if wind_speed_mph else "N/A")
        cols[3].metric(t('wind_dir'), f"{wind_dir:.0f}°" if wind_dir else "N/A")
        cols[4].metric(t('swell_dir'), f"{swell_dir:.0f}°" if swell_dir else "N/A")
        cols[5].metric(t('water_temp'), f"{water_temp_f:.0f}°F" if water_temp_f else "N/A")
        cols[6].metric(t('surf_score'), f"{surf_score} ⭐")
        cols[7].metric(t('wave_energy'), f"{wave_energy:.0f}" if wave_energy else "N/A")
        
        # Analysis
        if show_analysis:
            st.markdown("## 🔍 " + t('analysis'))
            analysis = generate_condition_analysis(surf_score, buoy_data, beach_info)
            st.write(analysis)
        
        # Forecast
        st.markdown("## 📅 " + t('forecast'))
        marine_data = get_marine_forecast(beach.lat, beach.lon, forecast_days)
        weather_data = get_weather_forecast(beach.lat, beach.lon, forecast_days)
        
        if show_charts and marine_data and weather_data:
            fig = create_forecast_charts(marine_data, weather_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tides
        st.markdown("## 🌊 " + t('tides'))
        tides = get_tide_predictions(beach.station_id, forecast_days)
        if tides:
            tide_df = pd.DataFrame(tides)
            st.table(tide_df[['t', 'v', 'type']])
            tide_fig = create_tide_chart(tides)
            st.plotly_chart(tide_fig, use_container_width=True)
        else:
            st.info("Tide data unavailable.")
    
    # Tab 2: Map
    with tabs[1]:
        st.markdown("## 🗺️ " + t('map'))
        map_html = create_interactive_map(BEACHES_DICT, current_conditions)
        st.components.v1.html(map_html, height=600)
    
    # Tab 3: Cams
    with tabs[2]:
        st.markdown("## 📹 " + t('cams'))
        selected_cam_beach = st.selectbox("Select Spot for Cams", list(BEACHES.keys()))
        cams = BEACHES[selected_cam_beach].cam_urls
        if cams:
            for url in cams:
                st.markdown(f"### Live Cam: {selected_cam_beach}")
                if 'youtube' in url:
                    embed_url = f'{url}?autoplay=1&mute=1&controls=0&loop=1'
                    st.components.v1.html(f'<iframe src="{embed_url}" width="100%" height="500" frameborder="0" allowfullscreen></iframe>', height=500)
                else:
                    st.image(url, use_column_width=True, caption="Updated every minute")
        else:
            st.info("No cams available for this spot.")

if __name__ == "__main__":
    main()
