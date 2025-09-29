import base64
import os
from datetime import datetime
from io import BytesIO
import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st


from development.layout_detection.inference_layout_detector import create_tiled_inference
from development.layout_detection.matching_symbol import EnhancedSymbolMatcher
from development.cable_length.cable_size import CableSizingCalculator

class CableLengthCalculator:
    def __init__(self, model_path: str, symbol_folder: str):
        self.model_path = model_path
        self.symbol_folder = symbol_folder
        
        # Initialize cable sizing calculator
        self.cable_calculator = CableSizingCalculator()
        
        # Initialize session state for cable length calculator
        self.init_cable_session_state()

    def init_cable_session_state(self):
        """Initialize session state variables for cable length calculator"""
        if "current_image_idx" not in st.session_state:
            st.session_state.current_image_idx = 0
        if "points" not in st.session_state:
            st.session_state.points = {}
        if "drawing_active" not in st.session_state:
            st.session_state.drawing_active = {}
        if "batches" not in st.session_state:
            st.session_state.batches = {}
        if "last_click" not in st.session_state:
            st.session_state.last_click = None
        if "circuit_images_processed" not in st.session_state:
            st.session_state.circuit_images_processed = False
        if "circuit_detections" not in st.session_state:
            st.session_state.circuit_detections = {}
        if "project_summary_data" not in st.session_state:
            st.session_state.project_summary_data = []
        if "building_scales" not in st.session_state:
            st.session_state.building_scales = {}
        if "scale_configured" not in st.session_state:
            st.session_state.scale_configured = False
        if "calculation_results" not in st.session_state:
            st.session_state.calculation_results = {}

    def get_circuit_images_from_folder(self):
        """Get circuit page images from the current extraction folder"""
        folder_path = getattr(st.session_state, 'current_extraction_folder', 'dynamic_extracted')
        
        if not os.path.exists(folder_path):
            return []
        
        circuit_images = []
        
        # Get corresponding drawings from PDF data
        if hasattr(st.session_state, 'pdf_drawings') and st.session_state.pdf_drawings:
            pdf_drawings = st.session_state.pdf_drawings['circuit']
        else:
            pdf_drawings = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') and 'circuit' in filename:
                img_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(img_path)
                    # Extract drawing number from filename
                    base_name = filename.replace('.png', '')
                    if '_circuit' in base_name:
                        drawing_number = base_name.replace('_circuit', '')
                    else:
                        drawing_number = base_name
                    
                    # Find matching drawing info from PDF data to get page number
                    page_number = 0
                    for i, drawing_info in enumerate(pdf_drawings):
                        if drawing_info['drawing_number'] == drawing_number:
                            page_number = i + 1
                            break
                    
                    circuit_images.append({
                        'image': img,
                        'filename': filename,
                        'drawing_number': drawing_number,
                        'page_number': page_number,
                        'image_path': img_path
                    })
                except Exception as e:
                    st.warning(f"Could not load image {filename}: {e}")
        
        # Sort by page number
        circuit_images.sort(key=lambda x: x['page_number'])
        return circuit_images

    def process_circuit_images(self, circuit_images):
        """Process circuit images with detection and symbol matching"""
        if not circuit_images:
            return {}
        
        # Initialize model and symbol matcher if not loaded
        if 'model_loaded' not in st.session_state:
            with st.spinner('Loading model and symbol matcher...'):
                tiled_inference = create_tiled_inference(self.model_path, 'high_precision')
                symbol_matcher = EnhancedSymbolMatcher(self.symbol_folder)
                
                st.session_state.model_loaded = True
                st.session_state.tiled_inference = tiled_inference
                st.session_state.symbol_matcher = symbol_matcher
                
                st.success("Model and symbol matcher loaded successfully!")
        else:
            tiled_inference = st.session_state.tiled_inference
            symbol_matcher = st.session_state.symbol_matcher
        
        detections_data = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, image_info in enumerate(circuit_images):
            filename = image_info['filename']
            img = image_info['image']
            
            status_text.text(f"Processing {filename}...")
            progress_bar.progress((idx + 1) / len(circuit_images))
            
            try:
                # Convert image for processing
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    cv_img = img_array

                height, width = cv_img.shape[:2]
                
                # Run detection
                detections = tiled_inference.predict_with_multiple_scales(cv_img)
                
                # Process detections and match symbols
                processed_detections = []
                
                for det in detections:
                    bbox = det['bbox'].astype(int)
                    x1, y1, x2, y2 = np.clip([bbox[0], bbox[1], bbox[2], bbox[3]],
                                           [0, 0, 0, 0], [width, height, width, height])

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detected_region = cv_img[y1:y2, x1:x2]
                    symbol_matches = symbol_matcher.match_symbol(detected_region, top_k=3)
                    
                    if symbol_matches and symbol_matches[0]['confidence'] > 0.3:
                        best_match = symbol_matches[0]
                        processed_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'symbol_name': best_match['symbol_name'],
                            'confidence': best_match['confidence']
                        })
                
                detections_data[filename] = {
                    'detections': processed_detections,
                    'image_info': image_info
                }
                
            except Exception as e:
                st.warning(f"Error processing {filename}: {str(e)}")
                detections_data[filename] = {
                    'detections': [],
                    'image_info': image_info
                }
        
        progress_bar.empty()
        status_text.empty()
        
        return detections_data

    def draw_detections_on_image(self, pil_image, detections, scale):
        """Draw detection bounding boxes and labels on image"""
        image_with_bbox = pil_image.copy()
        draw = ImageDraw.Draw(image_with_bbox)
        
        # Define colors for different symbol types
        colors = {
            'default': 'green',
            'panel': 'red', 
            'switch': 'blue',
            'outlet': 'orange',
            'light': 'purple'
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            symbol_name = detection['symbol_name']
            confidence = detection['confidence']
            
            # Scale coordinates
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            
            # Choose color based on symbol type
            color = colors.get('default', 'green')
            for key, col in colors.items():
                if key.lower() in symbol_name.lower():
                    color = col
                    break
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f"{symbol_name} ({confidence:.2f})"
            if len(label) > 25:
                label = label[:22] + "..."
            
            try:
                font = ImageFont.truetype("arial.ttf", 10)
            except IOError:
                font = ImageFont.load_default()
            
            draw.text((x1+2, y1-18), label, fill=color, font=font)
        
        return image_with_bbox

    def draw_cables(self, pil_image, scale, filename):
        """Draw cable routes on image"""
        image_with_cables = pil_image.copy()
        draw = ImageDraw.Draw(image_with_cables)
        colors = ['red', 'lime', 'blue', 'orange', 'magenta']
        
        current_points = st.session_state.points.get(filename, [])
        current_batches = st.session_state.batches.get(filename, [[]])
        
        # Draw completed routes
        for batch_idx, batch in enumerate(current_batches):
            if batch:
                color = colors[batch_idx % len(colors)]
                for p1, p2 in batch:
                    x1, y1 = int(p1[0] * scale), int(p1[1] * scale)
                    x2, y2 = int(p2[0] * scale), int(p2[1] * scale)
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        
        # Draw start and end points
        for batch_idx, batch in enumerate(current_batches):
            if batch:
                color = colors[batch_idx % len(colors)]
                first_point = batch[0][0]
                x1, y1 = int(first_point[0] * scale), int(first_point[1] * scale)
                draw.ellipse([x1-5, y1-5, x1+5, y1+5], fill=color)
                
                last_point = batch[-1][1]
                x2, y2 = int(last_point[0] * scale), int(last_point[1] * scale)
                draw.ellipse([x2-5, y2-5, x2+5, y2+5], fill=color)
        
        # Draw current drawing route
        drawing_active = st.session_state.drawing_active.get(filename, False)
        if drawing_active and len(current_points) > 1:
            current_color = colors[(len(current_batches)-1) % len(colors)]
            for i in range(len(current_points) - 1):
                p1, p2 = current_points[i], current_points[i + 1]
                x1, y1 = int(p1[0] * scale), int(p1[1] * scale)
                x2, y2 = int(p2[0] * scale), int(p2[1] * scale)
                draw.line([(x1, y1), (x2, y2)], fill=current_color, width=3)
            
            if current_points:
                first_p = current_points[0]
                x, y = int(first_p[0] * scale), int(first_p[1] * scale)
                draw.ellipse([x-5, y-5, x+5, y+5], fill=current_color)
                last_p = current_points[-1]
                x, y = int(last_p[0] * scale), int(last_p[1] * scale)
                draw.ellipse([x-5, y-5, x+5, y+5], fill=current_color)
        
        return image_with_cables

    def calculate_distance(self, p1, p2, scale_factor):
        """Calculate distance between two points"""
        x1, y1 = p1
        x2, y2 = p2
        pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return pixel_distance * scale_factor

    def get_bbox_label(self, point, detections):
        """Find the label of the detection that contains the given point"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                return detection['symbol_name']
        return "Unknown"

    def get_route_info(self, scale_factor, filename):
        """Get detailed route information with cable lengths"""
        routes_info = []
        current_batches = st.session_state.batches.get(filename, [])
        detections = st.session_state.circuit_detections.get(filename, {}).get('detections', [])
        
        for i, batch in enumerate(current_batches, 1):
            if batch and len(batch) > 0:
                route_length = sum(self.calculate_distance(p1, p2, scale_factor) for p1, p2 in batch)
                if route_length > 0:
                    start_point = batch[0][0]
                    end_point = batch[-1][1]
                    
                    start_label = self.get_bbox_label(start_point, detections)
                    end_label = self.get_bbox_label(end_point, detections)
                    
                    route_name = f"{start_label} to {end_label}"
                    routes_info.append({
                        'name': route_name,
                        'length': route_length,
                        'route_id': f"{filename}_route_{i}"
                    })
                
        return routes_info

    def set_building_scale(self, building_length, building_width, image_height, image_width):
        """Calculate scale based on user input building dimensions"""
        length_scale = building_length / image_height
        width_scale = building_width / image_width
        return (length_scale + width_scale) / 2

    def configure_scale_interface(self, circuit_images):
        """Interface for configuring building scale"""
        st.subheader("Building Scale Configuration")
        st.info("Configure the building dimensions to calculate accurate cable lengths.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            building_length = st.number_input("Building Length (m)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
        with col2:
            building_width = st.number_input("Building Width (m)", min_value=1.0, max_value=1000.0, value=40.0, step=1.0)
        with col3:
            if st.button("Set Scale", type="primary"):
                # Store scale for all circuit images
                for image_info in circuit_images:
                    filename = image_info['filename']
                    img = image_info['image']
                    orig_width, orig_height = img.size
                    scale_factor = self.set_building_scale(building_length, building_width, orig_height, orig_width)
                    st.session_state.building_scales[filename] = scale_factor
                
                st.session_state.scale_configured = True
                st.success("Scale configured successfully!")
                st.rerun()

    def add_point(self, x, y, scale, filename):
        """Add point to cable route only if scale is configured"""
        if not st.session_state.scale_configured:
            st.warning("Please configure building scale first!")
            return
            
        actual_x, actual_y = int(x / scale), int(y / scale)
        new_point = (actual_x, actual_y)
        
        if filename not in st.session_state.points:
            st.session_state.points[filename] = []
        if filename not in st.session_state.drawing_active:
            st.session_state.drawing_active[filename] = False
        if filename not in st.session_state.batches:
            st.session_state.batches[filename] = [[]]
        
        if not st.session_state.drawing_active[filename]:
            st.session_state.points[filename] = [new_point]
            st.session_state.drawing_active[filename] = True
        else:
            prev_point = st.session_state.points[filename][-1]
            st.session_state.points[filename].append(new_point)
            st.session_state.batches[filename][-1].append((prev_point, new_point))

    def clear_drawing(self, filename):
        """Clear all drawings for a filename"""
        st.session_state.points[filename] = []
        st.session_state.batches[filename] = [[]]
        st.session_state.drawing_active[filename] = False
        # Clear results for this image when drawing is cleared
        if filename in st.session_state.calculation_results:
            del st.session_state.calculation_results[filename]
        # Remove entries from project summary
        st.session_state.project_summary_data = [
            entry for entry in st.session_state.project_summary_data
            if entry['Floor Plan'] != filename
        ]

    def undo_last_point(self, filename):
        """Undo the last point added"""
        if filename in st.session_state.points and st.session_state.points[filename]:
            if len(st.session_state.points[filename]) > 1:
                st.session_state.points[filename].pop()
                if st.session_state.batches[filename] and st.session_state.batches[filename][-1]:
                    st.session_state.batches[filename][-1].pop()
            else:
                self.clear_drawing(filename)

    def add_to_project_summary(self, floor_plan, route_name, cable_length, results):
        """Add calculation results to project summary"""
        summary_entry = {
            'Floor Plan': floor_plan,
            'Route Details': route_name,
            'Cable Length': f"{cable_length:.1f}m",
            'Suggested Cable Size': results['suggested_cable_size'],
            'Neutral Core': results['neutral_core'],
            'Earth Size': results['earth_size'],
            'Current Rating': results['current_rating'],
            'Voltage Drop': results['voltage_drop'],
            'Conductor Material': results['conductor_material']
        }
        
        # Check if entry already exists and update, otherwise add new
        found = False
        for i, entry in enumerate(st.session_state.project_summary_data):
            if entry['Floor Plan'] == floor_plan and entry['Route Details'] == route_name:
                st.session_state.project_summary_data[i] = summary_entry
                found = True
                break
        
        if not found:
            st.session_state.project_summary_data.append(summary_entry)

    def run_tab4(self):
        """Tab 4: Cable Length Calculation in Circuit Pages"""
        st.header("Cable Length Calculation")
        
        # Load circuit pages button
        if st.button("Count in Circuit Pages", type="primary"):
            circuit_images = self.get_circuit_images_from_folder()
            
            if not circuit_images:
                st.warning("No circuit page images found. Please process PDF in Tab 1 first.")
            else:
                with st.spinner(f"Processing {len(circuit_images)} circuit pages..."):
                    detections_data = self.process_circuit_images(circuit_images)
                
                st.session_state.circuit_detections = detections_data
                st.session_state.circuit_images_processed = True
                st.success(f"Processed {len(circuit_images)} circuit pages successfully!")
                st.rerun()

        # Display results after processing
        if st.session_state.circuit_images_processed and st.session_state.circuit_detections:
            
            circuit_images = [info['image_info'] for info in st.session_state.circuit_detections.values()]
            
            # Scale configuration interface
            if not st.session_state.scale_configured:
                self.configure_scale_interface(circuit_images)
                st.warning("Please configure building scale before drawing cable routes.")
                return
            
            # Project Summary at the top
            if st.session_state.project_summary_data:
                st.subheader("Project Summary")
                
                # Create DataFrame
                summary_df = pd.DataFrame(st.session_state.project_summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Excel download
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, sheet_name='Cable Sizing Report', index=False)
                excel_buffer.seek(0)
                
                excel_str = base64.b64encode(excel_buffer.getvalue()).decode()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_filename = f"Cable_Sizing_Report_{timestamp}.xlsx"
                
                href = f'''
                <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_str}" 
                download="{excel_filename}">
                    <button style="background-color: #28a745; color: white; padding: 10px 20px; 
                                    border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
                        Download Excel Report
                    </button>
                </a>
                '''
                st.markdown(href, unsafe_allow_html=True)
                st.markdown("---")

            # Individual Circuit Image Display
            for idx, (filename, detection_data) in enumerate(st.session_state.circuit_detections.items()):
                st.markdown("---")
                
                image_info = detection_data['image_info']
                detections = detection_data['detections']
                
                # Display filename as header first
                st.subheader(f"**{idx + 1}. {image_info['drawing_number']}**")
                
                # Status and Control buttons
                st.markdown(f"**Status:** {'Active' if st.session_state.current_image_idx == idx else 'Inactive'}")
                
                # Control buttons
                col_buttons = st.columns(4)
                with col_buttons[0]:
                    if st.button("Activate", key=f"act_{idx}"):
                        st.session_state.current_image_idx = idx
                        st.rerun()
                with col_buttons[1]:
                    if st.session_state.current_image_idx == idx:
                        if st.button("New Route", key=f"new_{idx}"):
                            if filename not in st.session_state.batches:
                                st.session_state.batches[filename] = []
                            if filename not in st.session_state.drawing_active:
                                st.session_state.drawing_active[filename] = False
                            st.session_state.drawing_active[filename] = False
                            st.session_state.points[filename] = []
                            st.session_state.batches[filename].append([])
                            st.rerun()
                with col_buttons[2]:
                    if st.session_state.current_image_idx == idx:
                        if st.button("Undo", key=f"undo_{idx}"):
                            self.undo_last_point(filename)
                            st.rerun()
                with col_buttons[3]:
                    if st.session_state.current_image_idx == idx:
                        if st.button("Clear", key=f"clear_{idx}"):
                            self.clear_drawing(filename)
                            st.rerun()

                # Image processing and display
                pil_image = image_info['image']
                orig_width, orig_height = pil_image.size
                scale_factor = st.session_state.building_scales.get(filename, 1.0)
                
                max_size = 1800
                scale = min(max_size / orig_width, max_size / orig_height, 1.0)
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Draw detections
                display_image = self.draw_detections_on_image(display_image, detections, scale)
                
                # Interactive image display
                if st.session_state.current_image_idx == idx:
                    display_image = self.draw_cables(display_image, scale, filename)
                    value = streamlit_image_coordinates(display_image, key=f"img_{idx}")
                    
                    if value and value != st.session_state.last_click:
                        if st.session_state.current_image_idx != idx:
                            st.session_state.current_image_idx = idx
                        st.session_state.last_click = value
                        self.add_point(value['x'], value['y'], scale, filename)
                        st.rerun()
                else:
                    display_image = self.draw_cables(display_image, scale, filename)
                    value = streamlit_image_coordinates(display_image, key=f"img_static_{idx}")
                    if value:
                        st.session_state.current_image_idx = idx
                        st.rerun()
                
                # Route Details with Cable Calculator
                st.subheader("Route Details")
                routes_info = self.get_route_info(scale_factor, filename)
                
                if routes_info:
                    for route in routes_info:
                        route_key = route['route_id']
                        
                        # Checkbox for route selection
                        checkbox_key = f"select_{route_key}_{idx}"
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = False
                        
                        selected = st.checkbox(
                            f"{route['name']}: {route['length']:.1f}m",
                            value=st.session_state[checkbox_key],
                            key=checkbox_key
                        )
                        
                        # Handle checkbox state change
                        if selected != st.session_state[checkbox_key]:
                            st.session_state[checkbox_key] = selected
                            if selected:
                                for other_route in routes_info:
                                    other_checkbox_key = f"select_{other_route['route_id']}_{idx}"
                                    if other_checkbox_key != checkbox_key and other_checkbox_key in st.session_state:
                                        st.session_state[other_checkbox_key] = False
                            st.rerun()

                        # Show cable calculator when route is selected
                        if st.session_state[checkbox_key]:
                            st.markdown("---")
                            st.markdown("### Cable Sizing Calculator")
                            
                            # Cable calculator parameters
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                phase = st.selectbox("Phase Type", ["3 Phase AC", "Single Phase AC", "DC"], key=f"phase_{route_key}")
                                voltage = st.number_input("Voltage (V)", value=400 if phase == "3 Phase AC" else 230, key=f"voltage_{route_key}")
                                load_current = st.number_input("Load Current (A)", min_value=1, max_value=1000, value=250, key=f"current_{route_key}")
                            
                            with col2:
                                distance = st.number_input("Cable Length (m)", value=round(route['length'], 1), key=f"distance_{route_key}")
                                max_voltage_drop = st.slider("Max Voltage Drop (%)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, key=f"drop_{route_key}")
                                conductor = st.selectbox("Conductor Material", ["Copper", "Aluminium"], key=f"conductor_{route_key}")
                            
                            with col3:
                                insulation = st.selectbox("Insulation Type", ["XLPE X-90 Standard 90°", "PVC V-90 Standard 75°"], key=f"insulation_{route_key}")
                                cable_type = st.selectbox("Cable Configuration", ["Multi-core 3C+E", "Multi-core 4C+E", "Single-cores 3x1C+E"], key=f"cable_type_{route_key}")
                                installation_method = st.selectbox("Installation Method", ["Direct buried", "In conduit", "On cable tray", "Clipped direct", "Free air"], key=f"installation_{route_key}")
                            
                            # Calculate button
                            if st.button("Calculate Cable Size", key=f"calc_{route_key}"):
                                results = self.cable_calculator.get_calculation_results(
                                    load_current, distance, voltage, max_voltage_drop, phase, conductor
                                )
                                
                                # Store results
                                if filename not in st.session_state.calculation_results:
                                    st.session_state.calculation_results[filename] = {}
                                st.session_state.calculation_results[filename][route_key] = results

                                # Update project summary
                                self.add_to_project_summary(filename, route['name'], route['length'], results)
                                st.rerun()
                            
                            # Display stored results
                            if filename in st.session_state.calculation_results and route_key in st.session_state.calculation_results[filename]:
                                results = st.session_state.calculation_results[filename][route_key]
                                if results['success']:
                                    st.success("**Suitable Cable Found!**")
                                    
                                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                                    
                                    with col1:
                                        st.markdown("**Suggested Cable Size**")
                                        st.info(results['suggested_cable_size'])
                                    
                                    with col2:
                                        st.markdown("**Neutral Core**")
                                        st.info(results['neutral_core'])
                                    
                                    with col3:
                                        st.markdown("**Earth Size**")
                                        st.info(results['earth_size'])
                                    
                                    with col4:
                                        st.markdown("**Current Rating**")
                                        st.info(results['current_rating'])
                                    
                                    with col5:
                                        st.markdown("**Voltage Drop**")
                                        st.info(results['voltage_drop'])
                                    
                                    with col6:
                                        st.markdown("**Conductor Material**")
                                        st.info(results['conductor_material'])
                                else:
                                    st.error("No suitable cable found!")
                                    st.warning("Try increasing the voltage drop limit or reducing the cable length.")

                            st.markdown("---")
                else:
                    st.markdown("- No routes drawn yet")

        else:
            st.info("Click 'Count in Circuit Pages' to start cable length calculation")