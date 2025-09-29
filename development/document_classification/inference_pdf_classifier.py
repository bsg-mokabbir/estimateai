import os
from io import BytesIO
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import fitz  # PyMuPDF


class DocumentClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "fine_tuned_dit")
        self.device = torch.device("cpu")
        self.label_names = ["AutoCAD", "Circuit", "Legend", "Others"]
        
        print("Loading model...")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict_single(self, image_path):
        """Predict single image"""
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            "predicted_class": self.label_names[predicted_class],
            "confidence": confidence,
            "all_scores": {self.label_names[i]: predictions[0][i].item() 
                          for i in range(len(self.label_names))}
        }
    
    def predict_batch(self, image_folder):
        """Predict all images in folder"""
        results = []
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                try:
                    result = self.predict_single(image_path)
                    result['filename'] = filename
                    results.append(result)
                    print(f"{filename}: {result['predicted_class']} ({result['confidence']:.3f})")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return results
    
    def predict_image_from_pil(self, pil_image):
        """Predict from PIL Image object"""
        image = pil_image.convert('RGB')
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            "predicted_class": self.label_names[predicted_class],
            "confidence": confidence,
            "all_scores": {self.label_names[i]: predictions[0][i].item() 
                          for i in range(len(self.label_names))}
        }
    
    def predict_pdf_pages(self, pdf_path, dpi=300):
        """Predict each page of a PDF"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        results = []
        
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            print(f"Processing PDF with {total_pages} pages...")
            
            for page_num in range(total_pages):
                try:
                    # Get page
                    page = pdf_document.load_page(page_num)
                    
                    # Convert to image
                    mat = fitz.Matrix(dpi/72, dpi/72)  # scaling factor for DPI
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("ppm")
                    pil_image = Image.open(BytesIO(img_data))
                    
                    # Predict
                    result = self.predict_image_from_pil(pil_image)
                    result['page_number'] = page_num + 1
                    results.append(result)
                    
                    print(f"Page {page_num + 1}: {result['predicted_class']} ({result['confidence']:.3f})")
                    
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return []
        
        return results

def main():
    # Initialize classifier
    classifier = DocumentClassifier()
    
    # Single image prediction
    print("\n=== Single Image Prediction ===")
    single_image = input("Enter image path (or press Enter to skip): ").strip()
    if single_image and os.path.exists(single_image):
        result = classifier.predict_single(single_image)
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("All scores:", result['all_scores'])
    
    # Batch prediction
    print("\n=== Batch Prediction ===")
    folder_path = input("Enter folder path (or press Enter to skip): ").strip()
    if folder_path and os.path.exists(folder_path):
        results = classifier.predict_batch(folder_path)
        print(f"\nProcessed {len(results)} images")
    
    # PDF prediction
    print("\n=== PDF Prediction ===")
    pdf_path = input("Enter PDF path (or press Enter to skip): ").strip()
    if pdf_path and os.path.exists(pdf_path):
        try:
            results = classifier.predict_pdf_pages(pdf_path)
            
            print(f"\n=== PDF Summary ===")
            print(f"Total pages processed: {len(results)}")
            
            # Count predictions by class
            class_counts = {}
            for result in results:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            print("Predictions by class:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} pages")
                
        except Exception as e:
            print(f"Error processing PDF: {e}")

def extract_all_drawings(pdf_path, classifier_model_path=None):
    """
    Extract all drawing entries from a PDF using direct model classification
    This function can be called directly from tab1_pdf_processing.py
    """
    try:
        # Initialize classifier with provided model path
        classifier = DocumentClassifier(model_path=classifier_model_path)
        
        # Open PDF
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        
        print(f"PDF total pages: {total_pages}")
        print("Using document classification for all pages...")
        
        drawings = []
        
        for page_num in range(total_pages):
            try:
                # Get page and convert to image
                page = pdf_doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(BytesIO(img_data))
                
                # Classify page
                result = classifier.predict_image_from_pil(pil_image)
                
                # Map classifier output to our format
                class_mapping = {
                    "Legend": "legend", 
                    "Circuit": "circuit", 
                    "AutoCAD": "cad", 
                    "Others": "others"
                }
                page_type = class_mapping.get(result['predicted_class'], 'others')
                
                drawing = {
                    'drawing_number': f"PAGE_{page_num+1:03d}",
                    'description': f"Auto-classified as {result['predicted_class']}",
                    'type': page_type,
                    'page_number': page_num + 1  # 1-based page number
                }
                drawings.append(drawing)
                
                print(f"Page {page_num+1}: {result['predicted_class']} ({result['confidence']:.3f})")
                
                # Save page as image in dynamic_extracted folder
                os.makedirs("dynamic_extracted", exist_ok=True)
                filename = f"Page{page_num+1:03d}_{drawing['drawing_number']}_{page_type}.png"
                output_path = os.path.join("dynamic_extracted", filename)
                pix.save(output_path)
                
            except Exception as e:
                print(f"Error processing page {page_num+1}: {e}")
                # Add as 'others' if processing fails
                drawings.append({
                    'drawing_number': f"PAGE_{page_num+1:03d}",
                    'description': f"Processing failed: {str(e)}",
                    'type': 'others',
                    'page_number': page_num + 1
                })
        
        pdf_doc.close()
        
        print(f"Direct classification completed: {len(drawings)} pages classified")
        
        # Check if legend pages exist
        legend_count = sum(1 for drawing in drawings if drawing['type'] == 'legend')
        if legend_count == 0:
            print("No legend pages found in this document. Stopping execution.")
            return []
        
        print(f"\n=== FOUND {len(drawings)} DRAWINGS ===")
        for drawing in drawings:
            page_num = drawing.get('page_number', 'N/A')
            print(f"Page {page_num} - {drawing['drawing_number']}: {drawing['type']} - {drawing['description']}")
        
        # Summary
        legend_pages = [d for d in drawings if d['type'] == 'legend']
        circuit_pages = [d for d in drawings if d['type'] == 'circuit']
        cad_pages = [d for d in drawings if d['type'] == 'cad']
        others_pages = [d for d in drawings if d['type'] == 'others']
        
        print("\nSUMMARY:")
        print(f"Legend: {len(legend_pages)}")
        print(f"Circuit: {len(circuit_pages)}")
        print(f"CAD: {len(cad_pages)}")
        print(f"Others: {len(others_pages)}")
        
        return drawings
        
    except Exception as e:
        print(f"Error in extract_all_drawings: {e}")
        return []

if __name__ == "__main__":
    main()