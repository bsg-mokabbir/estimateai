import gradio as gr
import cv2
import numpy as np
from PIL import Image

# Your imports
from inference import TiledInference, create_tiled_inference
from matching_symbol import EnhancedSymbolMatcher


class GradioSymbolMatcher:
    def __init__(self, model_path: str, symbol_folder: str):
        self.model_path = model_path
        self.symbol_folder = symbol_folder            # Display selected symbols count
        selection_status = gr.Textbox(
            label="📊 Selection Status",
            value="No symbols selected yet. Run detection first, then select symbols above.",
            interactive=False,
            lines=2)

        print(f"Loading model from: {model_path}")
        print(f"Loading symbols from: {symbol_folder}")

        self.tiled_inference = create_tiled_inference(model_path, 'high_precision')
        self.symbol_matcher = EnhancedSymbolMatcher(symbol_folder)

        print("✅ Model and symbol matcher loaded successfully!")

    def process_image(self, image):
        if image is None:
            return None, "Please upload an image first.", []

        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img = img_array
            else:
                img = image

            height, width = img.shape[:2]
            detections = self.tiled_inference.predict_with_multiple_scales(img)

            matched_patches = []
            result_img = img.copy()
            high_conf_count, medium_conf_count, low_conf_count = 0, 0, 0

            for i, det in enumerate(detections):
                bbox = det['bbox'].astype(int)
                x1, y1, x2, y2 = np.clip([bbox[0], bbox[1], bbox[2], bbox[3]],
                                         [0, 0, 0, 0], [width, height, width, height])

                if x2 <= x1 or y2 <= y1:
                    continue

                detected_region = img[y1:y2, x1:x2]
                symbol_matches = self.symbol_matcher.match_symbol(detected_region, top_k=3)

                if symbol_matches:
                    best_match = symbol_matches[0]
                    conf = best_match['confidence']

                    if conf > 0.7:
                        color = (0, 255, 0)
                        conf_category = "High"
                        high_conf_count += 1
                    elif conf > 0.4:
                        color = (0, 255, 255)
                        conf_category = "Medium"
                        medium_conf_count += 1
                    else:
                        color = (0, 0, 255)
                        conf_category = "Low"
                        low_conf_count += 1

                    label = f"{best_match['symbol_name']} ({conf:.2f})"
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    detected_rgb = cv2.cvtColor(detected_region, cv2.COLOR_BGR2RGB)
                    patch_pil = Image.fromarray(detected_rgb)

                    matched_patches.append({
                        'patch': patch_pil,
                        'symbol_name': best_match['symbol_name'],
                        'confidence': conf,
                        'confidence_category': conf_category,
                        'bbox': [x1, y1, x2, y2],
                        'all_matches': symbol_matches
                    })

            summary = f"H:{high_conf_count} M:{medium_conf_count} L:{low_conf_count}"
            cv2.putText(result_img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)

            summary_text = f"""Processing Complete!
                            Total Detections: {len(matched_patches)}
                            High Confidence (>0.7): {high_conf_count}
                            Medium Confidence (0.4-0.7): {medium_conf_count}
                            Low Confidence (0.3-0.4): {low_conf_count}
                            Image Size: {width}x{height}"""

            return result_pil, summary_text, matched_patches

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)
            return None, error_msg, []

    def create_symbol_selection_cards(self, matched_patches):
        """Create individual symbol cards with image, name, and checkbox for pure Gradio"""
        if not matched_patches:
            return [], []
        
        # Get unique symbols to avoid duplicates
        unique_symbols = {}
        for patch_data in matched_patches:
            symbol_name = patch_data['symbol_name']
            if symbol_name not in unique_symbols or patch_data['confidence'] > unique_symbols[symbol_name]['confidence']:
                unique_symbols[symbol_name] = patch_data

        symbol_cards = []
        symbol_choices = []
        
        for i, (symbol_name, patch_data) in enumerate(unique_symbols.items()):
            # Create choice text for checkbox
            choice_text = f"{symbol_name} (conf: {patch_data['confidence']:.2f})"
            symbol_choices.append(choice_text)
            
            # Store the card data for rendering
            symbol_cards.append({
                'image': patch_data['patch'],
                'name': symbol_name,
                'confidence': patch_data['confidence'],
                'choice_text': choice_text,
                'index': i + 1
            })
        
        return symbol_cards, symbol_choices
    def extract_symbol_names_from_selection(self, selected_display_names, symbol_name_mapping):
        """Extract actual symbol names from display names selected in CheckboxGroup"""
        if not selected_display_names or not symbol_name_mapping:
            return []
        
        selected_symbols = []
        for display_name in selected_display_names:
            # Find corresponding symbol name
            for actual_name in symbol_name_mapping:
                if display_name.startswith(actual_name):
                    selected_symbols.append(actual_name)
                    break
        
        return selected_symbols

    def count_selected_symbols(self, image, selected_symbol_names):
        """Count occurrences of selected symbols in new image"""
        if image is None:
            return None, "Please upload an image for counting first.", {}

        if not selected_symbol_names:
            return None, "Please select at least one symbol to count.", {}

        try:
            # Convert image for processing
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img = img_array
            else:
                img = image

            height, width = img.shape[:2]
            
            # Use existing TiledInference pipeline
            detections = self.tiled_inference.predict_with_multiple_scales(img)
            
            # Count results for each selected symbol
            count_results = {}
            result_img = img.copy()
            
            # Initialize counts for selected symbols
            for symbol_name in selected_symbol_names:
                count_results[symbol_name] = {
                    'total_count': 0,
                    'high_confidence': 0,
                    'medium_confidence': 0,
                    'low_confidence': 0,
                    'detections': []
                }

            # Process each detection
            for i, det in enumerate(detections):
                bbox = det['bbox'].astype(int)
                x1, y1, x2, y2 = np.clip([bbox[0], bbox[1], bbox[2], bbox[3]],
                                         [0, 0, 0, 0], [width, height, width, height])

                if x2 <= x1 or y2 <= y1:
                    continue

                detected_region = img[y1:y2, x1:x2]
                symbol_matches = self.symbol_matcher.match_symbol(detected_region, top_k=10)

                # Check if any match belongs to selected symbols
                for match in symbol_matches:
                    if match['symbol_name'] in selected_symbol_names and match['confidence'] > 0.3:
                        symbol_name = match['symbol_name']
                        conf = match['confidence']
                        
                        # Determine confidence category and color
                        if conf > 0.7:
                            color = (0, 255, 0)  # Green
                            conf_category = "high_confidence"
                        elif conf > 0.4:
                            color = (0, 255, 255)  # Yellow
                            conf_category = "medium_confidence"
                        else:
                            color = (0, 0, 255)  # Red
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
            summary_lines = [f"Counting Results for {len(selected_symbol_names)} selected symbol(s):"]
            for symbol_name in selected_symbol_names:
                result = count_results[symbol_name]
                summary_lines.append(f"• {symbol_name}: {result['total_count']} found (H:{result['high_confidence']} M:{result['medium_confidence']} L:{result['low_confidence']})")

            summary_text = "\n".join(summary_lines)

            return result_pil, summary_text, count_results

        except Exception as e:
            error_msg = f"Error during counting: {str(e)}"
            print(error_msg)
            return None, error_msg, {}

    def create_interface(self):
        with gr.Blocks(title="Symbol Matching & Counting System", theme=gr.themes.Default()) as interface:
            gr.Markdown("# 🔍 Symbol Matching & Counting System")
            gr.Markdown("*Pure Gradio Implementation - No HTML/JS Required*")

            # Section 1: Detection
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="📸 Upload Image for Detection",
                        type="pil",
                        height=400
                    )
                    process_btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="📊 Detection Result", height=400)
                    summary_text = gr.Textbox(label="📋 Processing Summary", lines=8, max_lines=10)

            # State variables
            matched_patches_state = gr.State([])
            symbol_names_mapping = gr.State([])

            # Section 2: Pure Gradio Symbol Selection Interface - Card Based
            gr.Markdown("## 🧩 Detected Symbols - Select for Counting")
            gr.Markdown("*Each detected symbol is shown as a card with image, name, and selection checkbox*")
            
            # Container for dynamic symbol cards
            symbol_cards_container = gr.Column(visible=False)
            
            with symbol_cards_container:
                # This will be populated dynamically with symbol cards
                symbol_card_components = []
                
                # We'll create up to 10 card slots (can be expanded)
                for i in range(10):
                    with gr.Row(visible=False, elem_id=f"card_row_{i}") as card_row:
                        with gr.Column(scale=1, min_width=200):
                            card_image = gr.Image(
                                label=f"Symbol {i+1}",
                                show_label=True,
                                height=150,
                                width=150,
                                interactive=False,
                                visible=False
                            )
                        with gr.Column(scale=2):
                            card_info = gr.Markdown(
                                value="",
                                visible=False
                            )
                            card_checkbox = gr.Checkbox(
                                label="Select for counting",
                                value=False,
                                visible=False,
                                interactive=True
                            )
                    
                    symbol_card_components.append({
                        'row': card_row,
                        'image': card_image,
                        'info': card_info,
                        'checkbox': card_checkbox
                    })

            # Alternative: Simple but effective approach using CheckboxGroup + Gallery
            gr.Markdown("### Alternative View: Gallery + Selection")
            
            with gr.Row():
                with gr.Column(scale=3):
                    symbol_gallery = gr.Gallery(
                        label="🖼️ Detected Symbol Images",
                        show_label=True,
                        columns=4,
                        rows=2,
                        object_fit="contain",
                        height="auto",
                        allow_preview=True
                    )
                
                with gr.Column(scale=2):
                    symbol_info_table = gr.Dataframe(
                        label="📝 Symbol Details",
                        headers=["#", "Symbol Name", "Confidence", "Category"],
                        datatype=["number", "str", "str", "str"],
                        interactive=False
                    )

            # Selection interface using CheckboxGroup
            symbol_selector = gr.CheckboxGroup(
                label="✅ Select Symbols for Counting",
                choices=[],
                value=[],
                interactive=True,
                info="Check the symbols you want to count in new images"
            )
            
            # Display selected symbols count
            selection_status = gr.Textbox(
                label="� Selection Status",
                value="No symbols selected yet. Run detection first, then select symbols above.",
                interactive=False,
                lines=2
            )
            
            # Section 3: Symbol Counting
            gr.Markdown("## 🔢 Count Selected Symbols in New Image")
            with gr.Row():
                with gr.Column(scale=1):
                    count_image = gr.Image(
                        label="📷 Upload New Image for Counting",
                        type="pil",
                        height=400
                    )
                    count_btn = gr.Button("🎯 Count Selected Symbols", variant="secondary", size="lg")

                with gr.Column(scale=1):
                    count_result_image = gr.Image(label="🔢 Counting Results", height=400)
                    count_summary = gr.Textbox(label="📊 Counting Summary", lines=8, max_lines=10)

            # Functions
            def update_symbol_cards(patches):
                """Update symbol card components with detected symbols"""
                if not patches:
                    # Hide all cards
                    updates = []
                    for comp in symbol_card_components:
                        updates.extend([
                            gr.update(visible=False),  # row
                            gr.update(visible=False),  # image
                            gr.update(visible=False),  # info
                            gr.update(visible=False)   # checkbox
                        ])
                    return updates
                
                # Get unique symbols
                symbol_cards, symbol_choices = self.create_symbol_selection_cards(patches)
                
                updates = []
                for i, comp in enumerate(symbol_card_components):
                    if i < len(symbol_cards):
                        card_data = symbol_cards[i]
                        conf_category = "High" if card_data['confidence'] > 0.7 else "Medium" if card_data['confidence'] > 0.4 else "Low"
                        conf_color = "🟢" if conf_category == "High" else "🟡" if conf_category == "Medium" else "🔴"
                        
                        card_info_text = f"""
**{card_data['index']}. {card_data['name']}**

{conf_color} **Confidence:** {card_data['confidence']:.3f} ({conf_category})

*Select checkbox below to include in counting*
"""
                        
                        updates.extend([
                            gr.update(visible=True),   # row
                            gr.update(value=card_data['image'], visible=True, label=f"Symbol {card_data['index']}"),  # image
                            gr.update(value=card_info_text, visible=True),  # info
                            gr.update(visible=True, label=f"Select {card_data['name']}")   # checkbox
                        ])
                    else:
                        # Hide unused cards
                        updates.extend([
                            gr.update(visible=False),  # row
                            gr.update(visible=False),  # image
                            gr.update(visible=False),  # info
                            gr.update(visible=False)   # checkbox
                        ])
                
                return updates

            def process_and_update(image):
                """Process image and update all symbol-related components"""
                result_img, summary, patches = self.process_image(image)
                
                if not patches:
                    empty_gallery = []
                    empty_table = []
                    empty_choices = []
                    empty_mapping = []
                    status = "No symbols detected. Try uploading a different image."
                    
                    # Hide symbol cards container
                    card_updates = update_symbol_cards([])
                    
                    return ([result_img, summary, patches, empty_gallery, empty_table, 
                           gr.update(choices=empty_choices, value=[]), empty_mapping, status, 
                           gr.update(visible=False)] + card_updates)
                
                # Prepare data for components
                symbol_cards, symbol_choices = self.create_symbol_selection_cards(patches)
                
                # Prepare table data
                table_data = []
                actual_names = []
                for i, card in enumerate(symbol_cards):
                    conf_category = "High" if card['confidence'] > 0.7 else "Medium" if card['confidence'] > 0.4 else "Low"
                    table_data.append([i + 1, card['name'], f"{card['confidence']:.3f}", conf_category])
                    actual_names.append(card['name'])
                
                gallery_images = [card['image'] for card in symbol_cards]
                
                status = f"✅ {len(symbol_cards)} unique symbols detected. Use the card view or checkboxes below to select symbols for counting."
                
                # Update symbol cards
                card_updates = update_symbol_cards(patches)
                
                return ([result_img, summary, patches, gallery_images, table_data, 
                       gr.update(choices=symbol_choices, value=[]), actual_names, status,
                       gr.update(visible=True)] + card_updates)

            def get_selected_from_cards():
                """Get selected symbols from individual card checkboxes"""
                selected = []
                # This will be handled by the event handlers for each checkbox
                return selected

            def update_selection_status(selected_display_names, symbol_mapping):
                """Update the selection status based on current selections"""
                if not selected_display_names:
                    return "No symbols selected yet. Use the card checkboxes or the selection list below."
                
                actual_names = self.extract_symbol_names_from_selection(selected_display_names, symbol_mapping)
                count = len(actual_names)
                names_str = ", ".join(actual_names)
                
                return f"✅ {count} symbol(s) selected for counting: {names_str}"

            def count_symbols_with_selection(image, selected_display_names, symbol_mapping):
                """Count symbols using the current selection"""
                if not image:
                    return None, "Please upload an image for counting."
                
                if not selected_display_names:
                    return None, "Please select symbols using the card checkboxes or the selection list below."
                
                # Extract actual symbol names from display names
                selected_symbols = self.extract_symbol_names_from_selection(selected_display_names, symbol_mapping)
                
                if not selected_symbols:
                    return None, "No valid symbols selected. Please check your selection."
                
                result_img, summary, count_data = self.count_selected_symbols(image, selected_symbols)
                return result_img, summary

            # Event handlers
            all_outputs = [output_image, summary_text, matched_patches_state, symbol_gallery, 
                          symbol_info_table, symbol_selector, symbol_names_mapping, selection_status,
                          symbol_cards_container]
            
            # Add all card component outputs
            for comp in symbol_card_components:
                all_outputs.extend([comp['row'], comp['image'], comp['info'], comp['checkbox']])
            
            process_btn.click(
                fn=process_and_update,
                inputs=[input_image],
                outputs=all_outputs
            )

            # Update selection status when user changes selection
            symbol_selector.change(
                fn=update_selection_status,
                inputs=[symbol_selector, symbol_names_mapping],
                outputs=[selection_status]
            )

            count_btn.click(
                fn=count_symbols_with_selection,
                inputs=[count_image, symbol_selector, symbol_names_mapping],
                outputs=[count_result_image, count_summary]
            )

        return interface


def main():
    MODEL_PATH = r"/workspace/iteration-4/legent-count-detection/layout_detection/runs/train/experiment_20250805_180318/train/weights/best.pt"
    SYMBOL_FOLDER = r"Row_Legend"

    print("🚀 Initializing Symbol Matching & Counting System...")
    matcher = GradioSymbolMatcher(MODEL_PATH, SYMBOL_FOLDER)
    interface = matcher.create_interface()
    
    print("✅ System ready! Starting interface...")
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
