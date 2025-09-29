import os
import re
from io import BytesIO
from collections import Counter
import fitz
import cv2
import pytesseract
import numpy as np
from PIL import Image
from .inference_pdf_classifier import DocumentClassifier  

class DynamicPDFExtractor:
    """
    A PDF extractor that performs OCR-based drawing number detection,
    dynamic pattern recognition, description cleaning, classification,
    and page extraction/saving.
    """
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.base_patterns = []

    def preprocess_image_for_ocr(self, img_array):
        """Improve image quality for better OCR"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 3)
        height, width = gray.shape
        gray = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        return gray
    
    def detect_base_patterns(self, text):
        """Dynamically detect base patterns from OCR text"""
        # Normalize OCR errors - O to 0, I/i to 1
        normalized_text = text.replace('O', '0').replace('o', '0').replace('I', '1').replace('i', '1')
        
        potential_patterns = []
        
        # Pattern 1: Hyphen formats with 3-4 digits (LCE23094-E000)
        matches_hyphen_3_4 = re.findall(r'([A-Z][A-Z0-9]+-[A-Z])(\d{3,4})', normalized_text)
        for base, num in matches_hyphen_3_4:
            potential_patterns.append((base, len(num)))
        
        # Pattern 2: Hyphen formats with 2 digits (E-01) 
        matches_hyphen_2 = re.findall(r'([A-Z][A-Z0-9]*-?)(\d{2})(?![0-9])', normalized_text)
        for base, num in matches_hyphen_2:
            if '-' in base:
                potential_patterns.append((base, len(num)))
        
        # Pattern 3: Simple formats with 2 or more letters followed by digits (e.g., EL000)
        matches_simple_3_4 = re.findall(r'([A-Z]{2,})(\W*)(\d{3,4})', normalized_text)
        for base, _, num in matches_simple_3_4:
            potential_patterns.append((base, len(num)))
        
        # Pattern 4: Single letter + 4-5 digits (A0000-A0000) - Universal pattern
        matches_single = re.findall(r'([A-Z])(\d{4,5})', normalized_text)  # 4-5 digits to cover all cases
        for base, num in matches_single:
            potential_patterns.append((base, len(num)))
        
        # Count occurrences
        pattern_counts = Counter(potential_patterns)
        
        # Filter patterns - Changed from 3 to 2 minimum occurrences for A0000, A0001 type patterns
        confirmed_patterns = []
        for (base, num_len), count in pattern_counts.items():
            if count >= 2:  
                confirmed_patterns.append({
                    'base': base,
                    'num_length': num_len,
                    'count': count
                })
        
        # Sort by count and length to prioritize more specific patterns
        confirmed_patterns = sorted(confirmed_patterns, key=lambda x: (x['count'], len(x['base'])), reverse=True)
        
        # Keep only the pattern with highest priority (most specific)
        if confirmed_patterns:
            best_pattern = confirmed_patterns[0]
            confirmed_patterns = [best_pattern]
        
        print("=== DETECTED BASE PATTERNS ===")
        for pattern in confirmed_patterns:
            print(f"Base: '{pattern['base']}' + {pattern['num_length']} digits (found {pattern['count']} times)")
        
        return confirmed_patterns

    def extract_drawing_index_advanced(self):
        """Advanced drawing index extraction with dynamic pattern detection"""
        page = self.doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        img_data = pix.tobytes("png")
        
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_img = self.preprocess_image_for_ocr(img)
        
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        print("=== RAW OCR TEXT ===")
        print(text)
        print("===================")
        
        # Detect base patterns dynamically
        self.base_patterns = self.detect_base_patterns(text)
        
        # Parse drawings using detected patterns
        drawings = self.parse_drawings_with_patterns(text)
        return drawings
    
    def clean_description_smart(self, description):
        """Description cleaning based on first 3 words pattern"""
        if not description:
            return "NO DESCRIPTION FOUND"
        
        # Extract only words for counting (no numbers, punctuation)
        words_for_counting = re.findall(r'[A-Za-z]+', description)
        
        if len(words_for_counting) < 3:
            return description.strip() if description.strip() else "NO DESCRIPTION FOUND"
        
        # Check first 3 words (letters only)
        first_three = words_for_counting[:3]
        all_caps = all(word.isupper() for word in first_three)
        
        # Split original description into tokens (preserving everything)
        tokens = description.split()
        
        if all_caps:
            # Continue taking tokens as long as the letter-parts are caps
            result_tokens = []
            word_count = 0
            
            for token in tokens:
                # Extract letters from this token
                letters_in_token = ''.join(re.findall(r'[A-Za-z]', token))
                
                if letters_in_token and letters_in_token.isupper():
                    result_tokens.append(token)
                    word_count += 1
                elif not letters_in_token:  # Pure numbers/punctuation, keep it
                    result_tokens.append(token)
                else:
                    break  # Stop at first token with non-caps letters
        else:
            # Mixed case in first 3 - take max 10 word-tokens
            result_tokens = []
            word_count = 0
            
            for token in tokens:
                if word_count >= 10:
                    break
                result_tokens.append(token)
                # Only count if token contains letters
                if re.search(r'[A-Za-z]', token):
                    word_count += 1
        
        result = ' '.join(result_tokens).strip()
        return result if len(result) >= 3 else "NO DESCRIPTION FOUND"

    def parse_drawings_with_patterns(self, text):
        
        """Parse drawings using detected base patterns with OCR error handling - fully automated"""
        parsed_drawings = []
            
        # Keep original text for descriptions, only normalize for finding drawing numbers
        original_text = text
            
        found_drawing_numbers = set()
            
        for pattern in self.base_patterns:
            base = pattern['base']
            num_len = pattern['num_length']
                
            # Create flexible regex pattern for different formats
            drawing_pattern_regex = rf'(?<![A-Z0-9]){re.escape(base)}(\d{{{num_len}}})(?![0-9])'
                
            # Iterate line by line to handle descriptions better
            for line in original_text.split('\n'):
                # Normalize line for finding drawing numbers, but use original line for descriptions
                normalized_line = line.replace('O', '0').replace('o', '0').replace('I', '1').replace('i', '1').replace('l', '1')
                    
                # Find all matches in this line
                drawing_matches = list(re.finditer(drawing_pattern_regex, normalized_line))
                    
                for i, match in enumerate(drawing_matches):
                    full_num = match.group(0) # Get the full drawing number
                        
                    if full_num in found_drawing_numbers:
                        continue # Skip if already processed
                    found_drawing_numbers.add(full_num)
                        
                    if i + 1 < len(drawing_matches):
                        # Get the start position of the next drawing number on the same line
                        next_match = drawing_matches[i+1]
                        # Extract description from the original line, from the end of the current match
                        # to the start of the next match
                        description = line[match.end():next_match.start()].strip()
                    else:
                        # No more drawing numbers on this line, so take the rest of the line as the description
                        description = line[match.end():].strip()
                            
                    # Clean description using smart pattern detection
                    description = self.clean_description_smart(description)
                        
                    # Classify the page type
                    page_type = self.classify_from_description(description)
                        
                    parsed_drawings.append({
                        'drawing_number': full_num,
                        'description': description,
                        'type': page_type
                    })
            
        print(f"Found {len(parsed_drawings)} drawing entries for pattern '{base}'")
            
        # Sort drawings by drawing number
        def get_sort_key(drawing):
            numbers = re.findall(r'\d+', drawing['drawing_number'])
            return int(numbers[-1]) if numbers else 0
            
        parsed_drawings.sort(key=get_sort_key)
        return parsed_drawings

    def classify_from_description(self, description):
        """Classify using single keywords - max matches count decides"""
        if not description:
            return 'others'  # changed from 'unknown' to 'others'
        
        desc_lower = description.lower()
        
        # Convert multi-word keywords to single words
        legend_keywords = ['legend', 'symbol']
        cad_keywords = ['floor', 'level', 'basement', 'ground', 'layout', 'luminaire', 'arrangement','lighting','schedule', 'terrace']
        circuit_keywords = ['schematic', 'single', 'line', 'site', 'diagram', 'electrical', 'distribution', 'panel', 'riser', 'one', 'telecommunications', 'matv', 'intercom', 'fire', 'alarm', 'power', 'communications', 'roof', 'security']
        
        # Count matches
        legend_matches = sum(1 for keyword in legend_keywords if keyword in desc_lower)
        cad_matches = sum(1 for keyword in cad_keywords if keyword in desc_lower) 
        circuit_matches = sum(1 for keyword in circuit_keywords if keyword in desc_lower)
        
        # Return highest match
        max_matches = max(legend_matches, cad_matches, circuit_matches)
        
        if max_matches == 0:
            return 'others'  
        if legend_matches == max_matches:
            return 'legend'
        if circuit_matches == max_matches:
            return 'circuit'
        if cad_matches == max_matches:
            return 'cad'
        
        return 'others'  
    
    def extract_pages_by_type(self, target_type, drawings):
        """Extract pages by type - including index page in count"""
        target_pages = []
        
        for i, drawing in enumerate(drawings):
            if drawing['type'] == target_type:
                # PDF page number is i+1 (1-based page numbering for display)
                pdf_page_num = i + 1
                
                # But still use i for actual PDF access (0-based)
                actual_page_index = i
                
                if actual_page_index < len(self.doc):
                    target_pages.append({
                        'page_num': pdf_page_num,  # Display page number (1-based)
                        'actual_index': actual_page_index,  # Internal use (0-based)
                        'drawing_number': drawing['drawing_number'],
                        'description': drawing['description'],
                        'type': drawing['type']
                    })
        
        return target_pages
    
    def save_pages(self, pages, output_dir):
        """Save pages as images"""
        os.makedirs(output_dir, exist_ok=True)
        
        for page_info in pages:
            # Use actual_index for PDF access, page_num for display
            actual_page_index = page_info['actual_index']
            display_page_num = page_info['page_num']
            drawing_num = page_info['drawing_number'].replace('-', '_').replace('£', 'E')
            page_type = page_info['type']
            
            if actual_page_index < len(self.doc):
                page = self.doc[actual_page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                
                filename = f"Page{display_page_num}_{drawing_num}_{page_type}.png"
                output_path = os.path.join(output_dir, filename)
                pix.save(output_path)
                
                print(f"Saved: {filename}")
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()

def extract_all_drawings(pdf_path):
    """Extract all drawing entries from a PDF, classify them by type with fallback"""
    extractor = DynamicPDFExtractor(pdf_path)
    
    try:
        # Get total pages in PDF
        total_pages = len(extractor.doc)
        print(f"PDF total pages: {total_pages}")
        
        # Try OCR-based extraction first
        drawings = extractor.extract_drawing_index_advanced()
        detected_pages = len(drawings)
        
        print(f"OCR detected {detected_pages} drawings")
        
        # Check legend pages from initial detection
        legend_count = sum(1 for drawing in drawings if drawing['type'] == 'legend')
        
        # Check if legend pages exist, if not stop execution
        if legend_count == 0:
            print("No legend pages found in this document. Stopping execution.")
            return []
        
        # Check if we need fallback (8+ pages difference)
        if total_pages - detected_pages >= 8:
            print(f"Gap of {total_pages - detected_pages} pages detected. Using document classification fallback...")
            
            try:
                # Use document classifier as fallback
                classifier = DocumentClassifier()
                
                # Classify all pages
                fallback_drawings = []
                for page_num in range(total_pages):
                    page = extractor.doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("ppm")
                    
                    pil_image = Image.open(BytesIO(img_data))
                    
                    result = classifier.predict_image_from_pil(pil_image)
                    
                    # Map classifier output to our format
                    class_mapping = {"Legend": "legend", "Circuit": "circuit", "AutoCAD": "cad", "Others": "others"}
                    page_type = class_mapping.get(result['predicted_class'], 'others')
                    
                    fallback_drawings.append({
                        'drawing_number': f"PAGE_{page_num+1:03d}",
                        'description': f"Auto-classified as {result['predicted_class']}",
                        'type': page_type,
                        'page_number': page_num + 1  # ADD: Include actual PDF page number (1-based)
                    })
                    
                    print(f"Page {page_num+1}: {result['predicted_class']} ({result['confidence']:.3f})")
                
                drawings = fallback_drawings
                print(f"Fallback classification completed: {len(drawings)} pages classified")
                
            except Exception as e:
                print(f"Fallback failed: {e}")
                print("Using original OCR results...")
        else:
            # ADD: For OCR-based extraction, add page numbers based on position in list
            for i, drawing in enumerate(drawings):
                drawing['page_number'] = i + 1  # Add page number (1-based)
        
        print(f"\n=== FOUND {len(drawings)} DRAWINGS ===")
        for drawing in drawings:
            page_num = drawing.get('page_number', 'N/A')
            print(f"Page {page_num} - {drawing['drawing_number']}: {drawing['type']} - {drawing['description']}")
        
        legend_pages = extractor.extract_pages_by_type('legend', drawings)
        circuit_pages = extractor.extract_pages_by_type('circuit', drawings)
        cad_pages = extractor.extract_pages_by_type('cad', drawings)
        others_pages = extractor.extract_pages_by_type('others', drawings)
        
        print("\nSUMMARY:")
        print(f"Legend: {len(legend_pages)}")
        print(f"Circuit: {len(circuit_pages)}")
        print(f"CAD: {len(cad_pages)}")
        print(f"Others: {len(others_pages)}")
        
        all_pages = legend_pages + circuit_pages + cad_pages + others_pages
        extractor.save_pages(all_pages, "dynamic_extracted")
        
        return drawings
        
    finally:
        extractor.close()

if __name__ == "__main__":
    PDF_FILE = "pdf/path"
    drawings = extract_all_drawings(PDF_FILE)