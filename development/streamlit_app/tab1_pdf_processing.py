import sys
import os
import tempfile
from datetime import datetime
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from development.document_classification.inference_pdf_classifier import extract_all_drawings


class PDFProcessor:
    def __init__(self, classifier_model_path=None):
        self.classifier_model_path = classifier_model_path

    def process_pdf(self, pdf_file):
        """Process uploaded PDF and extract drawings by type"""
        try:
            # Create unique folder with timestamp for this PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_folder = f"dynamic_extracted_{timestamp}"
            os.makedirs(unique_folder, exist_ok=True)
            
            # Store the current folder path in session state
            st.session_state.current_extraction_folder = unique_folder
            
            # Save uploaded PDF to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.read())
                temp_pdf_path = temp_file.name
            
            # Extract drawings using your existing code with classifier model path
            with st.spinner('📄 Processing PDF and extracting pages...'):
                # Pass the classifier model path to extract_all_drawings
                drawings = extract_all_drawings(temp_pdf_path, classifier_model_path=self.classifier_model_path)
                
                # Move extracted files to unique folder
                if os.path.exists("dynamic_extracted"):
                    import shutil
                    for file in os.listdir("dynamic_extracted"):
                        if file.endswith('.png'):
                            shutil.move(
                                os.path.join("dynamic_extracted", file),
                                os.path.join(unique_folder, file)
                            )
            
            # Separate drawings by type including Others
            legend_drawings = [d for d in drawings if d['type'] == 'legend']
            circuit_drawings = [d for d in drawings if d['type'] == 'circuit'] 
            cad_drawings = [d for d in drawings if d['type'] == 'cad']
            others_drawings = [d for d in drawings if d['type'] == 'others']  # New Others category
            
            # Clean up temp file
            os.unlink(temp_pdf_path)
            
            return {
                'legend': legend_drawings,
                'circuit': circuit_drawings,
                'cad': cad_drawings,
                'others': others_drawings,  # New Others category
                'total': len(drawings)
            }
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

    def run_tab1(self):
        """Tab 1: PDF Upload & Processing"""
        st.header("📄 Upload PDF for Processing")
        
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF containing engineering drawings"
        )
        
        if uploaded_pdf is not None:
            if st.button("🚀 Process PDF", type="primary"):
                # COMPLETELY clear ALL previous session data when processing new PDF
                st.session_state.pdf_processed = False
                st.session_state.pdf_drawings = None
                st.session_state.matched_patches = []
                st.session_state.selected_symbols = []
                st.session_state.editable_counts = {}  # Clear editable counts
                st.session_state.display_pdf_results = False
                # Clear any other results
                if 'cad_results' in st.session_state:
                    del st.session_state.cad_results
                if 'current_extraction_folder' in st.session_state:
                    del st.session_state.current_extraction_folder
                
                with st.spinner("📄 Processing PDF and extracting pages by type..."):
                    pdf_results = self.process_pdf(uploaded_pdf)
                
                if pdf_results:
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_drawings = pdf_results
                    st.session_state.display_pdf_results = True
        
        if st.session_state.get('display_pdf_results', False) and st.session_state.get('pdf_drawings'):
            pdf_results = st.session_state.pdf_drawings
            
            st.success("✅ PDF processed successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📋 Legend Pages", len(pdf_results['legend']))
            with col2:
                st.metric("🔌 Circuit Pages", len(pdf_results['circuit']))
            with col3:
                st.metric("🏗️ AutoCAD Pages", len(pdf_results['cad']))
            with col4:
                st.metric("📄 Others Pages", len(pdf_results['others']))
            
            # Show legend pages for selection
            if pdf_results['legend']:
                st.subheader("📋 Legend Pages Found")
                for i, drawing in enumerate(pdf_results['legend']):
                    st.write(f"**{i+1}.** {drawing['drawing_number']}: {drawing['description']}")
            else:
                st.subheader("📋 Legend Pages No Found")
                st.warning("⚠️ No Legend pages found in this PDF.")
            
            # Table format for all pages (Drawing Index)
            st.subheader("📊 All Pages Drawing Index")
            table_data = []

            for drawing in pdf_results['legend']:
                table_data.append({
                    'PDF Page': drawing.get('page_number', 'N/A'),
                    'Page': drawing['drawing_number'],
                    'Page Type': 'Legend',
                    'Description': drawing['description']
                })

            for drawing in pdf_results['circuit']:
                table_data.append({
                    'PDF Page': drawing.get('page_number', 'N/A'),
                    'Page': drawing['drawing_number'],
                    'Page Type': 'Circuit',
                    'Description': drawing['description']
                })
                
            for drawing in pdf_results['cad']:
                table_data.append({
                    'PDF Page': drawing.get('page_number', 'N/A'),
                    'Page': drawing['drawing_number'],
                    'Page Type': 'AutoCAD',
                    'Description': drawing['description']
                })
                
            for drawing in pdf_results['others']:
                table_data.append({
                    'PDF Page': drawing.get('page_number', 'N/A'),
                    'Page': drawing['drawing_number'],
                    'Page Type': 'Others',
                    'Description': drawing['description']
                })

            if table_data:
                df = pd.DataFrame(table_data)
                # Sort by PDF Page number if it's numeric
                try:
                    df['PDF Page'] = pd.to_numeric(df['PDF Page'], errors='coerce')
                    df = df.sort_values('PDF Page')
                except:
                    pass  # Keep original order if sorting fails
                
                st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    processor = PDFProcessor()
    processor.run_tab1()


if __name__ == "__main__":
    main()