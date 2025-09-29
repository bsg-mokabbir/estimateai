import base64
import os
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_image_zoom import image_zoom

from rnd.layout_detection.inference.inference_layout_detector import create_tiled_inference
from rnd.layout_detection.inference.matching_symbol import EnhancedSymbolMatcher


class SymbolCounter:
    def __init__(self, model_path: str, symbol_folder: str):
        self.model_path = model_path
        self.symbol_folder = symbol_folder

        # Initialize session state
        if 'model_loaded' not in st.session_state:
            with st.spinner('Loading model and symbol matcher...'):
                self.tiled_inference = create_tiled_inference(model_path, 'high_precision')
                self.symbol_matcher = EnhancedSymbolMatcher(symbol_folder)
                
                st.session_state.model_loaded = True
                st.session_state.tiled_inference = self.tiled_inference
                st.session_state.symbol_matcher = self.symbol_matcher
                
                st.success("✅ Model and symbol matcher loaded successfully!")
        else:
            self.tiled_inference = st.session_state.tiled_inference
            self.symbol_matcher = st.session_state.symbol_matcher

    def get_page_images_from_folder(self, page_type):
        """Get page images from the current unique extraction folder and match with PDF data"""
        # Use current extraction folder from session state
        folder_path = getattr(st.session_state, 'current_extraction_folder', 'dynamic_extracted')
        
        if not os.path.exists(folder_path):
            return []
        
        page_images = []
        
        # Get corresponding drawings from PDF data
        if hasattr(st.session_state, 'pdf_drawings') and st.session_state.pdf_drawings:
            pdf_drawings = st.session_state.pdf_drawings[page_type]
        else:
            pdf_drawings = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') and page_type in filename:
                img_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(img_path)
                    # Extract drawing number from filename
                    base_name = filename.replace('.png', '')
                    if f'_{page_type}' in base_name:
                        drawing_number = base_name.replace(f'_{page_type}', '')
                    else:
                        drawing_number = base_name
                    
                    # Find matching drawing info from PDF data to get page number
                    page_number = 0
                    for i, drawing_info in enumerate(pdf_drawings):
                        if drawing_info['drawing_number'] == drawing_number:
                            # Page number is i+1 (1-based) from the PDF extraction
                            page_number = i + 1
                            break
                    
                    page_images.append({
                        'image': img,
                        'filename': filename,
                        'drawing_number': drawing_number,
                        'page_number': page_number
                    })
                except Exception as e:
                    st.warning(f"Could not load image {filename}: {e}")
        
        # Sort by page number
        page_images.sort(key=lambda x: x['page_number'])
        return page_images

    def generate_excel_summary_only(self, all_results, selected_symbols):
        """Generate Excel with only Summary sheet using editable counts"""
        if not all_results or not selected_symbols:
            return None
        
        excel_buffer = BytesIO()
        
        try:
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                summary_data = []
                for result in all_results:
                    drawing_num = result['drawing_number']
                    page_num = result.get('page_number', 0)
                    
                    row_data = {
                        'Filename': result['filename'],
                    }
                    
                    total_symbols = 0
                    for symbol_name in selected_symbols:
                        # Add safety check for editable_counts
                        if drawing_num in st.session_state.editable_counts:
                            count = st.session_state.editable_counts[drawing_num].get(symbol_name, 0)
                        else:
                            count = 0
                        row_data[f'{symbol_name} Count'] = count
                        total_symbols += count
                    
                    row_data['Total Symbols Found'] = total_symbols
                    summary_data.append(row_data)
                
                if summary_data:  # Only create sheet if data exists
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                else:
                    # Create empty sheet with headers if no data
                    empty_data = {'Filename': [], 'Total Symbols Found': []}
                    for symbol_name in selected_symbols:
                        empty_data[f'{symbol_name} Count'] = []
                    empty_df = pd.DataFrame(empty_data)
                    empty_df.to_excel(writer, sheet_name='Summary', index=False)
            
            excel_buffer.seek(0)
            return excel_buffer
            
        except Exception as e:
            st.error(f"Error generating Excel file: {str(e)}")
            return None

    def count_symbols_in_pages(self, page_images, selected_symbols):
        """Count selected symbols in multiple pages"""
        if not page_images or not selected_symbols:
            return []

        all_results = []
        
        for page_info in page_images:
            page_image = page_info['image']
            filename = page_info['filename']
            drawing_number = page_info['drawing_number']
            page_number = page_info.get('page_number', 0)
            
            # Process this page
            result_img, summary, count_data = self.count_selected_symbols(page_image, selected_symbols)
            
            if result_img is not None:
                all_results.append({
                    'drawing_number': drawing_number,
                    'filename': filename,
                    'page_number': page_number,
                    'result_image': result_img,
                    'summary': summary,
                    'count_data': count_data,
                    'total_found': sum(result['total_count'] for result in count_data.values())
                })
        
        return all_results

    def count_selected_symbols(self, image, selected_symbols):
        """Count occurrences of selected symbols in new image"""
        if image is None:
            return None, "Please upload an image for counting first.", {}

        if not selected_symbols:
            return None, "Please select at least one symbol to count.", {}

        try:
            # Convert image for processing
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array

            height, width = img.shape[:2]
            
            # Run detection with progress indicator
            with st.spinner('🔍 Detecting objects in counting image...'):
                detections = self.tiled_inference.predict_with_multiple_scales(img)
            
            # Initialize counting results
            count_results = {}
            result_img = img.copy()
            
            for symbol_name in selected_symbols:
                count_results[symbol_name] = {
                    'total_count': 0,
                    'high_confidence': 0,
                    'medium_confidence': 0,
                    'low_confidence': 0,
                    'detections': []
                }

            # Process each detection with progress indicator
            with st.spinner(f'🎯 Counting {len(selected_symbols)} selected symbols in {len(detections)} detected objects...'):
                for i, det in enumerate(detections):
                    bbox = det['bbox'].astype(int)
                    x1, y1, x2, y2 = np.clip([bbox[0], bbox[1], bbox[2], bbox[3]],
                                             [0, 0, 0, 0], [width, height, width, height])

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detected_region = img[y1:y2, x1:x2]
                    symbol_matches = self.symbol_matcher.match_symbol(detected_region, top_k=10)

                    symbol_colors = {name: color for name, color in zip(selected_symbols, get_distinct_colors(len(selected_symbols)))}

                    # Check if any match belongs to selected symbols
                    for match in symbol_matches:
                        if match['symbol_name'] in selected_symbols and match['confidence'] > 0.3:
                            symbol_name = match['symbol_name']
                            conf = match['confidence']

                            # Use the color assigned to this symbol
                            color = symbol_colors[symbol_name]
                            
                            # Determine confidence category and color
                            if conf > 0.7:
                                conf_category = "high_confidence"
                            elif conf > 0.4:
                                conf_category = "medium_confidence"
                            else:
                                conf_category = "low_confidence"

                            # Update counts
                            count_results[symbol_name]['total_count'] += 1
                            count_results[symbol_name][conf_category] += 1
                            count_results[symbol_name]['detections'].append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf
                            })

                            # Draw on result image
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                            label = f"{symbol_name} ({conf:.2f})"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            break  # Only count the best match per detection

            # Add summary to image
            total_found = sum(result['total_count'] for result in count_results.values())
            summary = f"Total Found: {total_found}"
            cv2.putText(result_img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert result image
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)

            # Create summary text
            summary_lines = [f"Counting Results for {len(selected_symbols)} selected symbol(s):"]
            for symbol_name in selected_symbols:
                result = count_results[symbol_name]
                summary_lines.append(f"• {symbol_name}: {result['total_count']} found (H:{result['high_confidence']} M:{result['medium_confidence']} L:{result['low_confidence']})")

            summary_text = "\n".join(summary_lines)

            return result_pil, summary_text, count_results

        except Exception as e:
            error_msg = f"Error during counting: {str(e)}"
            st.error(error_msg)
            return None, error_msg, {}

    def create_excel_download_link(self, excel_buffer, filename):
        """Create download link for Excel file"""
        if excel_buffer is None:
            return None
        
        # Encode the Excel file as base64
        excel_str = base64.b64encode(excel_buffer.getvalue()).decode()
        
        href = f'''
        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_str}" 
        download="{filename}">
            <button style="background-color: #28a745; color: white; padding: 10px 20px; 
                        border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
                📊 Download Excel Report
            </button>
        </a>
        '''
        return href

    def run_tab3(self):
        """Tab 3: Symbol Counting in Pages"""
        st.header("🔢 Symbol Counting in AutoCAD Pages")
        
        if not st.session_state.get('selected_symbols', []):
            st.warning("⚠️ Please detect symbols and make selections in the 'Legend Detection' tab first.")
        else:
            st.info("🎯 Ready to count the following symbols:")
            for sym in st.session_state.selected_symbols:
                st.markdown(f"- {sym}")
            
            # Only AutoCAD button
            if st.button("🏗️ Count in AutoCAD Pages", type="primary"):
                cad_images = self.get_page_images_from_folder('cad')
                
                if not cad_images:
                    st.warning("No AutoCAD page images found.")
                else:
                    with st.spinner(f"🔢 Processing {len(cad_images)} AutoCAD pages..."):
                        cad_results = self.count_symbols_in_pages(cad_images, st.session_state.selected_symbols)
                    
                    if cad_results:
                        st.session_state.cad_results = cad_results
                        # Initialize editable counts
                        for result in cad_results:
                            drawing_num = result['drawing_number']
                            st.session_state.editable_counts[drawing_num] = {}
                            for symbol_name, count_data in result['count_data'].items():
                                st.session_state.editable_counts[drawing_num][symbol_name] = count_data['total_count']
                        st.success(f"✅ Processed {len(cad_results)} AutoCAD pages!")

            # Display results
            if hasattr(st.session_state, 'cad_results') and st.session_state.cad_results is not None:
                # NON-EDITABLE Summary table
                st.write("### 🏗️ AutoCAD Pages Results - Summary")
                
                cad_results = st.session_state.cad_results
                
                # Create columns for the table header
                cols = ['Page'] + st.session_state.selected_symbols + ['Total']
                table_cols = st.columns(len(cols))
                
                for i, col_name in enumerate(cols):
                    with table_cols[i]:
                        st.write(f"**{col_name}**")
                
                # Display NON-EDITABLE summary rows
                for result in cad_results:
                    drawing_num = result['drawing_number']
                    row_cols = st.columns(len(cols))
                    
                    with row_cols[0]:
                        st.write(drawing_num)
                    
                    total_count = 0
                    for j, symbol_name in enumerate(st.session_state.selected_symbols):
                        with row_cols[j + 1]:
                            # Add safety check for editable_counts
                            if drawing_num in st.session_state.editable_counts:
                                count = st.session_state.editable_counts[drawing_num].get(symbol_name, 0)
                            else:
                                count = 0
                            st.write(f"**{count}**")  # Display only, no input
                            total_count += count
                    
                    with row_cols[-1]:
                        st.write(f"**{total_count}**")
                
                # Excel download
                excel_buffer = self.generate_excel_summary_only(cad_results, st.session_state.selected_symbols)
                if excel_buffer:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    excel_filename = f"AutoCAD_Summary_Report_{timestamp}.xlsx"
                    excel_link = self.create_excel_download_link(excel_buffer, excel_filename)
                    st.markdown(excel_link, unsafe_allow_html=True)
                
                # EDITABLE Individual Pages with Plus/Minus buttons
                st.subheader("📄 Individual Page")
                if cad_results:  # Add safety check here too
                    for result in cad_results:
                        drawing_num = result['drawing_number']
                        with st.expander(f"📄 {drawing_num}", expanded=False):
                            # Full width image display (preview)
                            img = result['result_image']
                            st.image(img, caption=f"AutoCAD: {drawing_num}", use_container_width=True)

                            # Zoom / Pan viewer on click
                            with st.popover("🔍 Zoom / Pan"):
                                image_zoom(img, zoom_factor=2)

                            # Editable counts below image
                            st.write("**Symbol Counts:**")
                            
                            for symbol_name in st.session_state.selected_symbols:
                                # Add safety check for editable_counts
                                if drawing_num in st.session_state.editable_counts:
                                    current_count = st.session_state.editable_counts[drawing_num].get(symbol_name, 0)
                                else:
                                    current_count = 0
                                
                                # Create columns for symbol editing
                                col_title, col_count, col_minus, col_plus = st.columns([3, 1, 1, 1])
                                
                                with col_title:
                                    st.write(f"**{symbol_name}**")
                                
                                with col_count:
                                    st.write(f"**{current_count}**")
                                
                                with col_minus:
                                    if st.button("➖", key=f"minus_{drawing_num}_{symbol_name}"):
                                        if current_count > 0:
                                            # Initialize if not exists
                                            if drawing_num not in st.session_state.editable_counts:
                                                st.session_state.editable_counts[drawing_num] = {}
                                            st.session_state.editable_counts[drawing_num][symbol_name] = current_count - 1
                                            st.rerun()
                                
                                with col_plus:
                                    if st.button("➕", key=f"plus_{drawing_num}_{symbol_name}"):
                                        # Initialize if not exists
                                        if drawing_num not in st.session_state.editable_counts:
                                            st.session_state.editable_counts[drawing_num] = {}
                                        st.session_state.editable_counts[drawing_num][symbol_name] = current_count + 1
                                        st.rerun()
                            
                        st.divider()
                else:
                    st.info("No AutoCAD results to display. Click '🏗️ Count in AutoCAD Pages' first.")


def get_distinct_colors(n):
    """Generate distinct colors for different symbols"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # OpenCV uses hue in [0,180]
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors


def main():
    MODEL_PATH = r"model/path"
    SYMBOL_FOLDER = r"Row_Legend"
    
    counter = SymbolCounter(MODEL_PATH, SYMBOL_FOLDER)
    counter.run_tab3()


if __name__ == "__main__":
    main()