import streamlit as st

# Import your four tab modules
from rnd.streamlit_app.tab1_pdf_processing import PDFProcessor
from rnd.streamlit_app.tab2_legend_detection import LegendDetector
from rnd.streamlit_app.tab3_symbol_counting import SymbolCounter
from rnd.streamlit_app.tab4_cable_length_with_cable_size import CableLengthCalculator

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Symbol Matching & Counting System",
    page_icon="🔍",
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
    st.title("🔍 PDF Symbol Matching & Counting System")
    st.write("Upload PDF to extract pages, detect symbols, select specific symbols, and count them across pages.")
    
    # Initialize session state
    initialize_session_state()
    
    # Model and folder paths
    MODEL_PATH = r"layout_detection\inference\save_model"
    SYMBOL_FOLDER = r"layout_detection\inference\Row_Legend"
    
    # Initialize processors
    pdf_processor = PDFProcessor()
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
        st.write("This tool extracts pages from PDF, detects symbols in legend pages, and counts them across AutoCAD and Circuit pages.")
    
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