"""
ULTIMATE SURF FORECAST V2.0 - FIXED & IMPROVED
==============================================
A world-class surf forecasting platform for NY/NJ - Now with robust error handling!

Key Fixes:
- Fixed NoneType division errors
- Proper null checking throughout
- Graceful error handling
- Optional PyTorch (doesn't crash if missing)
- Better data validation
- Improved user feedback
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from pathlib import Path
import hashlib
import sqlite3
from dataclasses import dataclass, asdict
import base64
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="🌊 Ultimate Surf Forecast V2",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "Ultimate Surf Forecast V2 - Open Source Surf Forecasting Platform"
    }
)

# Try to import PyTorch, but don't fail if it's not available
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. ML predictions disabled.")

# Initialize session state
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'favorite_beaches' not in st.session_state:
    st.session_state.favorite_beaches = []
if 'alert_preferences' not in st.session_state:
    st.session_state.alert_preferences = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# ==================== UTILITY FUNCTIONS ====================

def safe_float(value: Any, multiplier: float = 1.0, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float with multiplier"""
    try:
        if value is not None and value != 'MM' and str(value).strip() != '':
            return float(value) * multiplier
        return default
    except (ValueError, TypeError):
        return default

def safe_divide(numerator: Optional[float], denominator: float, default: Optional[float] = None) -> Optional[float]:
    """Safely divide, handling None values"""
    if numerator is None or denominator == 0:
        return default
    try:
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

# ==================== DATA MODELS ====================

@dataclass
class Beach:
    """Enhanced beach data model"""
    name: str
    buoy: str
    backup_buoys: List[str]
    lat: float
    lon: float
    station_id: str
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
    water_quality: str = "Unknown"
    accessibility: str = "Moderate"
    eco_rating: float = 0.0
    community_rating: float = 0.0
    report_count: int = 0
    
@dataclass
class UserProfile:
    """User profile data model"""
    username: str
    email: str
    skill_level: str
    preferred_beaches: List[str]
    alert_settings: Dict[str, Any]
    equipment: List[str]
    theme: str
    language: str
    eco_score: int = 0
    badges: List[str] = None
    
    def __post_init__(self):
        if self.badges is None:
            self.badges = []

@dataclass
class SpotReport:
    """Community spot report"""
    beach_name: str
    username: str
    timestamp: datetime
    rating: int
    wave_quality: int
    crowd_level: str
    conditions: str
    photos: List[str]
    helpful_count: int = 0

# ==================== DATABASE SETUP ====================

class DatabaseManager:
    """SQLite database manager for user data and community features"""
    
    def __init__(self, db_path: str = "surf_forecast.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    skill_level TEXT,
                    theme TEXT DEFAULT 'light',
                    language TEXT DEFAULT 'en',
                    eco_score INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Favorite beaches table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS favorite_beaches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    beach_name TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Spot reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spot_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    beach_name TEXT NOT NULL,
                    user_id INTEGER,
                    rating INTEGER,
                    wave_quality INTEGER,
                    crowd_level TEXT,
                    conditions TEXT,
                    photo_paths TEXT,
                    helpful_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def create_user(self, username: str, email: str, password: str) -> bool:
        """Create new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple hash for demo (use bcrypt in production)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor.execute('''
                SELECT id, username, email, skill_level, theme, language, eco_score
                FROM users
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'username': result[1],
                    'email': result[2],
                    'skill_level': result[3],
                    'theme': result[4],
                    'language': result[5],
                    'eco_score': result[6]
                }
            return None
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def get_user_favorites(self, user_id: int) -> List[str]:
        """Get user's favorite beaches"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT beach_name FROM favorite_beaches
                WHERE user_id = ?
            ''', (user_id,))
            
            favorites = [row[0] for row in cursor.fetchall()]
            conn.close()
            return favorites
        except Exception as e:
            logger.error(f"Error fetching favorites: {e}")
            return []
    
    def add_favorite_beach(self, user_id: int, beach_name: str):
        """Add beach to favorites"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO favorite_beaches (user_id, beach_name)
                VALUES (?, ?)
            ''', (user_id, beach_name))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error adding favorite: {e}")
    
    def add_spot_report(self, beach_name: str, user_id: int, rating: int, 
                       wave_quality: int, crowd_level: str, conditions: str):
        """Add new spot report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO spot_reports (beach_name, user_id, rating, wave_quality, 
                                         crowd_level, conditions)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (beach_name, user_id, rating, wave_quality, crowd_level, conditions))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error adding report: {e}")
    
    def get_recent_reports(self, beach_name: str, limit: int = 10) -> List[Dict]:
        """Get recent spot reports for a beach"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sr.rating, sr.wave_quality, sr.crowd_level, sr.conditions,
                       sr.created_at, u.username, sr.helpful_count
                FROM spot_reports sr
                JOIN users u ON sr.user_id = u.id
                WHERE sr.beach_name = ?
                ORDER BY sr.created_at DESC
                LIMIT ?
            ''', (beach_name, limit))
            
            reports = []
            for row in cursor.fetchall():
                reports.append({
                    'rating': row[0],
                    'wave_quality': row[1],
                    'crowd_level': row[2],
                    'conditions': row[3],
                    'timestamp': row[4],
                    'username': row[5],
                    'helpful_count': row[6]
                })
            
            conn.close()
            return reports
        except Exception as e:
            logger.error(f"Error fetching reports: {e}")
            return []

# Initialize database
try:
    db = DatabaseManager()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    db = None

# ==================== BEACH DATABASE ====================

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
        water_quality="Good",
        accessibility="Easy",
        eco_rating=4.2,
        community_rating=4.5,
        report_count=0
    ),
    "Long Beach, NY": Beach(
        name="Long Beach, NY",
        buoy="44065",
        backup_buoys=["44025"],
        lat=40.5887,
        lon=-73.6579,
        station_id="8531680",
        break_type="Beach Break",
        bottom="Sand",
        exposure="Open Ocean",
        best_swell={"min": 135, "max": 225, "optimal": 180},
        best_wind={"min": 315, "max": 45, "optimal": 0},
        best_tide={"phase": "mid", "range": [2, 4]},
        optimal_size={"min": 2, "max": 5},
        crowd="Moderate",
        skill_level="Beginner to Advanced",
        hazards=["Rip currents", "Jetties"],
        best_season=["Fall", "Winter", "Spring"],
        parking="Paid parking",
        facilities=["Bathrooms", "Showers", "Restaurants"],
        notes="Consistent waves. Multiple peaks. Resident parking required in summer.",
        water_quality="Good",
        accessibility="Easy",
        eco_rating=4.0,
        community_rating=4.3
    ),
}

# Convert to dict for backward compatibility
BEACHES_DICT = {k: asdict(v) for k, v in BEACHES.items()}

# ==================== ML PREDICTIONS (OPTIONAL) ====================

if PYTORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        """LSTM-based wave height prediction model"""
        
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

def predict_wave_trend_simple(wave_height_ft: Optional[float], hours_ahead: int = 24) -> Tuple[List[float], List[datetime]]:
    """Simple trend prediction without ML - more reliable"""
    if wave_height_ft is None or wave_height_ft <= 0:
        return [], []
    
    try:
        predictions = []
        future_times = []
        current_time = datetime.now()
        
        # Simple model: slight decay with random variation
        for i in range(hours_ahead):
            # Add some realistic variation
            variation = np.random.normal(0, 0.2)
            decay = 0.98 ** i  # Slight decay over time
            pred = max(0.5, wave_height_ft * decay + variation)
            
            predictions.append(pred)
            future_times.append(current_time + timedelta(hours=i+1))
        
        return predictions, future_times
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return [], []

# ==================== INTERNATIONALIZATION ====================

TRANSLATIONS = {
    'en': {
        'title': 'ULTIMATE SURF FORECAST',
        'subtitle': 'The Most Advanced Surf Forecasting Platform for NY/NJ',
        'beach_selection': 'Beach Selection',
        'current_conditions': 'Current Surf Conditions',
        'go_surf_now': 'GO SURF NOW!',
        'excellent_conditions': 'Conditions are excellent! Don\'t miss this session.',
        'wave_height': 'Wave Height',
        'wave_period': 'Wave Period',
        'wind_speed': 'Wind Speed',
        'water_temp': 'Water Temp',
        'crowd_level': 'Crowd Level',
        'login': 'Login',
        'signup': 'Sign Up',
        'favorites': 'Favorites',
        'community': 'Community',
        'forecast': 'Forecast',
        'about': 'About',
    },
    'es': {
        'title': 'PRONÓSTICO ULTIMATE DE SURF',
        'subtitle': 'La Plataforma Más Avanzada de Pronóstico de Surf para NY/NJ',
        'beach_selection': 'Selección de Playa',
        'current_conditions': 'Condiciones Actuales de Surf',
        'go_surf_now': '¡SURFEA AHORA!',
        'excellent_conditions': '¡Las condiciones son excelentes! No te pierdas esta sesión.',
        'wave_height': 'Altura de Ola',
        'wave_period': 'Período de Ola',
        'wind_speed': 'Velocidad del Viento',
        'water_temp': 'Temp. del Agua',
        'crowd_level': 'Nivel de Multitud',
        'login': 'Iniciar Sesión',
        'signup': 'Registrarse',
        'favorites': 'Favoritos',
        'community': 'Comunidad',
        'forecast': 'Pronóstico',
        'about': 'Acerca de',
    }
}

def t(key: str) -> str:
    """Translate text based on current language"""
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# ==================== AUTHENTICATION UI ====================

def show_auth_page():
    """Show login/signup page"""
    st.markdown('<div class="main-header">🌊 ' + t('title') + '</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([t('login'), t('signup')])
    
    with tab1:
        st.subheader(t('login'))
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if db:
                user = db.authenticate_user(username, password)
                if user:
                    st.session_state.user_authenticated = True
                    st.session_state.user_data = user
                    st.session_state.favorite_beaches = db.get_user_favorites(user['id'])
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            else:
                st.error("Database not available")
    
    with tab2:
        st.subheader(t('signup'))
        new_username = st.text_input("Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.button("Sign Up", key="signup_button"):
            if new_password != confirm_password:
                st.error("Passwords don't match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            elif db:
                if db.create_user(new_username, new_email, new_password):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username or email already exists")
            else:
                st.error("Database not available")
    
    # Guest mode option
    st.markdown("---")
    if st.button("Continue as Guest"):
        st.session_state.user_authenticated = True
        st.session_state.user_data = {'username': 'Guest', 'id': None}
        st.rerun()

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=900)
def get_buoy_data_with_fallback(buoy_id: str, backup_buoys: Optional[List[str]] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """Fetch buoy data with automatic fallback to backup buoys"""
    if backup_buoys is None:
        backup_buoys = []
    
    buoys_to_try = [buoy_id] + backup_buoys
    
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
                    
                    logger.info(f"Successfully fetched data from buoy {buoy}")
                    return buoy_dict, buoy
        except Exception as e:
            logger.warning(f"Failed to fetch buoy {buoy}: {e}")
            continue
    
    logger.error(f"All buoy fetches failed for {buoy_id}")
    return None, None

# ==================== COMMUNITY FEATURES ====================

def show_community_section(beach_name: str):
    """Display community features"""
    st.markdown("## 👥 " + t('community'))
    
    if not db:
        st.warning("Community features require database connection")
        return
    
    # Recent spot reports
    st.subheader("Recent Spot Reports")
    reports = db.get_recent_reports(beach_name, limit=5)
    
    if reports:
        for report in reports:
            with st.expander(f"Report by {report['username']} - {report['timestamp']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Rating:** {'⭐' * report['rating']}")
                    st.write(f"**Wave Quality:** {report['wave_quality']}/10")
                    st.write(f"**Crowd:** {report['crowd_level']}")
                    st.write(f"**Conditions:** {report['conditions']}")
                with col2:
                    if st.button("👍 Helpful", key=f"helpful_{report['timestamp']}"):
                        st.success("Thanks for the feedback!")
    else:
        st.info("No recent reports. Be the first to contribute!")
    
    # Submit new report
    if st.session_state.user_authenticated and st.session_state.user_data.get('id'):
        st.subheader("Submit a Report")
        
        with st.form("spot_report_form"):
            rating = st.slider("Overall Rating", 1, 5, 3)
            wave_quality = st.slider("Wave Quality", 1, 10, 5)
            crowd_level = st.selectbox("Crowd Level", ["Empty", "Light", "Moderate", "Heavy", "Packed"])
            conditions = st.text_area("Conditions & Notes")
            
            submitted = st.form_submit_button("Submit Report")
            if submitted:
                db.add_spot_report(
                    beach_name,
                    st.session_state.user_data['id'],
                    rating,
                    wave_quality,
                    crowd_level,
                    conditions
                )
                st.success("Report submitted! Thank you for contributing.")
                st.rerun()

# ==================== MAIN APPLICATION ====================

def main():
    # Check authentication
    if not st.session_state.user_authenticated:
        show_auth_page()
        return
    
    # Custom CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            .subtitle {
                font-size: 0.9rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">🌊 ' + t('title') + '</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">' + t('subtitle') + '</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.user_data.get('username'):
            st.write(f"👤 {st.session_state.user_data['username']}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏄 " + t('beach_selection'))
        selected_beach = st.selectbox(
            "Choose Your Spot",
            options=list(BEACHES.keys()),
            index=0
        )
        
        # Add to favorites
        if st.session_state.user_data.get('id') and db:
            is_favorite = selected_beach in st.session_state.favorite_beaches
            if st.button("⭐ Add to Favorites" if not is_favorite else "⭐ Remove from Favorites"):
                if not is_favorite:
                    db.add_favorite_beach(st.session_state.user_data['id'], selected_beach)
                    st.session_state.favorite_beaches.append(selected_beach)
                    st.success("Added to favorites!")
                else:
                    st.session_state.favorite_beaches.remove(selected_beach)
                    st.success("Removed from favorites!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("## ⚙️ Display Options")
        
        show_predictions = st.checkbox("🔮 Wave Predictions", value=True)
        show_community = st.checkbox("👥 Community Reports", value=True)
        
        if not PYTORCH_AVAILABLE and show_predictions:
            st.info("ℹ️ Using simple prediction model (PyTorch not installed)")
    
    # Main content
    beach_info = BEACHES_DICT[selected_beach]
    
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <h1 style='margin: 0; font-size: 2.5rem;'>🏖️ {selected_beach}</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;'>
                {beach_info['break_type']} • {beach_info['bottom']} Bottom • {beach_info['skill_level']}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner("Fetching surf conditions..."):
        buoy_data, buoy_used = get_buoy_data_with_fallback(
            beach_info['buoy'],
            beach_info.get('backup_buoys')
        )
    
    if buoy_data:
        st.markdown("## 🎯 " + t('current_conditions'))
        
        if buoy_used != beach_info['buoy']:
            st.info(f"ℹ️ Using backup buoy: {buoy_used}")
        
        # Extract and convert data safely
        wave_height_m = buoy_data.get('WVHT')
        wave_height_ft = safe_float(wave_height_m, 3.28084)
        
        wave_period = safe_float(buoy_data.get('DPD'))
        wind_speed_ms = buoy_data.get('WSPD')
        wind_speed_mph = safe_float(wind_speed_ms, 2.23694)
        
        water_temp_c = buoy_data.get('WTMP')
        water_temp_f = safe_float(water_temp_c, 9/5, 0)
        if water_temp_f is not None:
            water_temp_f += 32
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if wave_height_ft is not None:
                st.metric(t('wave_height'), f"{wave_height_ft:.1f} ft")
                
                # Quality indicator
                optimal = beach_info['optimal_size']
                if optimal['min'] <= wave_height_ft <= optimal['max']:
                    st.success("🟢 Optimal!")
                elif wave_height_ft < optimal['min']:
                    st.warning("🟡 Small")
                else:
                    st.error("🔴 Large")
            else:
                st.metric(t('wave_height'), "N/A")
        
        with col2:
            if wave_period is not None:
                st.metric(t('wave_period'), f"{wave_period:.0f} sec")
                if wave_period > 10:
                    st.success("🟢 Good period!")
                elif wave_period > 7:
                    st.info("🟡 Moderate")
                else:
                    st.warning("🔴 Short period")
            else:
                st.metric(t('wave_period'), "N/A")
        
        with col3:
            if wind_speed_mph is not None:
                st.metric(t('wind_speed'), f"{wind_speed_mph:.0f} mph")
                if wind_speed_mph < 10:
                    st.success("🟢 Light winds")
                elif wind_speed_mph < 15:
                    st.info("🟡 Moderate")
                else:
                    st.warning("🔴 Strong winds")
            else:
                st.metric(t('wind_speed'), "N/A")
        
        with col4:
            if water_temp_f is not None:
                st.metric(t('water_temp'), f"{water_temp_f:.0f}°F")
                if water_temp_f > 70:
                    st.write("🩳 Shorts")
                elif water_temp_f > 60:
                    st.write("🧥 Spring suit")
                else:
                    st.write("🥶 Full suit + boots")
            else:
                st.metric(t('water_temp'), "N/A")
        
        # Wave predictions
        if show_predictions and wave_height_ft is not None:
            st.markdown("## 🔮 24-Hour Forecast")
            
            predictions, pred_times = predict_wave_trend_simple(wave_height_ft, hours_ahead=24)
            
            if predictions:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pred_times, 
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted Wave Height',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Wave Height Forecast",
                    xaxis_title="Time",
                    yaxis_title="Wave Height (ft)",
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("⚠️ Forecast is a simple prediction model. Check local surf reports for accuracy.")
        
        # Beach details
        with st.expander("🏖️ Beach Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Break Type:** {beach_info['break_type']}")
                st.write(f"**Bottom:** {beach_info['bottom']}")
                st.write(f"**Exposure:** {beach_info['exposure']}")
                st.write(f"**Crowd Level:** {beach_info['crowd']}")
            with col2:
                st.write(f"**Best Seasons:** {', '.join(beach_info['best_season'])}")
                st.write(f"**Parking:** {beach_info['parking']}")
                st.write(f"**Facilities:** {', '.join(beach_info['facilities'])}")
            
            st.write(f"**Notes:** {beach_info['notes']}")
            st.write(f"**⚠️ Hazards:** {', '.join(beach_info['hazards'])}")
        
        # Community section
        if show_community:
            show_community_section(selected_beach)
    
    else:
        st.error(f"""
        ⚠️ **Unable to fetch surf data for {selected_beach}**
        
        We tried the following buoys:
        - Primary: {beach_info['buoy']}
        - Backups: {', '.join(beach_info.get('backup_buoys', []))}
        
        **Possible reasons:**
        - Buoy maintenance or offline
        - Network connectivity issues  
        - NOAA service temporary outage
        
        **What to do:**
        - Try refreshing in a few minutes
        - Select a different beach
        - Check [NDBC directly](https://www.ndbc.noaa.gov/)
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("⚙️ Settings"):
            st.info("Settings coming soon!")
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.user_authenticated = False
            st.session_state.user_data = {}
            st.rerun()
    
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    border-radius: 15px; margin-top: 2rem;'>
            <p>Made with ❤️ for the NY/NJ surf community | Open Source</p>
            <p style='font-size: 0.8rem; color: #666;'>Version 2.0 Fixed | Powered by NOAA</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application crashed: {e}", exc_info=True)
        if st.button("🔄 Restart Application"):
            st.rerun()
