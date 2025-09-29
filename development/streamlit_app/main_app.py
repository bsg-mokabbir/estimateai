import sys
import os
import hashlib 
import streamlit as st 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from development.streamlit_app.tab1_pdf_processing import PDFProcessor
from development.streamlit_app.tab2_legend_detection import LegendDetector
from development.streamlit_app.tab3_symbol_counting import SymbolCounter
from development.streamlit_app.tab4_cable_length_with_cable_size import CableLengthCalculator

# Simple credentials
VALID_CREDENTIALS = {
    "demo": "demo123",
    "client": "client2024",
    "admin": "admin456"
}

def hash_password(password):
    """Simple password hashing"""
    return hashlib.md5(password.encode()).hexdigest()

def check_login():
    """Simple login function"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        # Welcome header with styling
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #2E86AB; font-size: 3rem; margin-bottom: 0.5rem;'>
                🤖 Welcome to EstimateAI
            </h1>
            <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
                Advanced Electrical Symbol Detection & Counting System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        with st.form("login_form"):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.subheader("🔐 Please Login to Continue")
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                login_btn = st.form_submit_button("🔓 Login", use_container_width=True)
        
        if login_btn:
            if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid username or password!")
        
        return False
    return True

# Configure Streamlit page
st.set_page_config(
    page_title="EstimateAI - PDF Symbol Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'pdf_drawings' not in st.session_state:
        st.session_state.pdf_drawings = None
    if 'matched_patches' not in st.session_state:
        st.session_state.matched_patches = []
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = []
    if 'display_pdf_results' not in st.session_state:
        st.session_state.display_pdf_results = False
    if 'editable_counts' not in st.session_state:
        st.session_state.editable_counts = {}
    if 'cad_results' not in st.session_state:
        st.session_state.cad_results = None
    if 'circuit_editable_counts' not in st.session_state:
        st.session_state.circuit_editable_counts = {}
    if 'circuit_results' not in st.session_state:
        st.session_state.circuit_results = None

def main():
    """Main Streamlit application"""
    # Check login first
    if not check_login():
        return
    
    # Add logout button in sidebar
    with st.sidebar:
        st.write(f"👤 Logged in as: **{st.session_state.username}**")
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        st.divider()
    
    st.title("🤖 EstimateAI - Electrical Symbol Matching & Counting System")
    st.write("Upload PDF to extract pages, detect symbols, select specific symbols, and count them across pages.")
    
    # Initialize session state
    initialize_session_state()
    
    # Model and folder paths
    MODEL_PATH = "/app/data/development/detection-model/iteration-4_500Epoch_50Black_50Red.pt" 
    CLASSIFIER_MODEL_PATH = "/app/data/development/pdf-classifier/fine-tuned-dit-20" 
    SYMBOL_FOLDER = "/app/data/development/raw-legend"

    
    # Initialize processors with model paths
    pdf_processor = PDFProcessor(classifier_model_path=CLASSIFIER_MODEL_PATH)
    legend_detector = LegendDetector(MODEL_PATH, SYMBOL_FOLDER)
    symbol_counter = SymbolCounter(MODEL_PATH, SYMBOL_FOLDER)
    cable_length_calculator = CableLengthCalculator(MODEL_PATH, SYMBOL_FOLDER)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📋 Instructions")
        st.write("""
        1. **Upload a PDF** for page extraction
        2. **🔍 Legend Detection** for Run detection & select symbols
        3. **🔢 Symbol Counting** for Click 🏗️ Count in AutoCAD Pages
        4. **🔌 Cable Length** for Click 🔌 Count in Circuit Pages
        5. **📊 Results** for View counts & summary report
        """)
       
        st.header("ℹ️ About")
        st.write("This   tool extracts pages from PDF, detects symbols in legend pages, and counts them across AutoCAD and Circuit pages.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF Upload & Processing", "🔍 Legend Detection", "🔢 Symbol Counting", "🔌 Cable Length"])
    
    # Tab 1: PDF Processing
    with tab1:
        pdf_processor.run_tab1()
    

    
    # Tab 2: Legend Detection
    with tab2:
        legend_detector.run_tab2()
    
    # Tab 3: Symbol Counting
    with tab3:
        symbol_counter.run_tab3()
    
    # Tab 4: Cable Length
    with tab4:
        cable_length_calculator.run_tab4()

if __name__ == "__main__":
    main()