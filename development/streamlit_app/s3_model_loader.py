import os
import streamlit as st
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

class ModelLoader:
    """Download models and symbols from S3"""
    
    def __init__(self):
        # AWS credentials from Streamlit secrets
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            region_name=st.secrets.get("AWS_REGION", "ap-south-1")
        )
        self.bucket_name = st.secrets.get("S3_BUCKET_NAME", "estimate-ai-models")
        
        # Local paths in /tmp for Streamlit Cloud
        self.local_base_path = Path("/tmp/estimate-ai-data")
        self.local_base_path.mkdir(parents=True, exist_ok=True)
    
    def download_file_from_s3(self, s3_key: str, local_path: str, show_progress: bool = False):
        """Download single file from S3"""
        try:
            # Create local directory
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Check if file already exists
            if os.path.exists(local_path):
                if show_progress:
                    st.info(f"✅ Already exists: {os.path.basename(local_path)}")
                return local_path
            
            # Download from S3
            if show_progress:
                st.info(f"⬇️ Downloading: {os.path.basename(local_path)}...")
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            if show_progress:
                st.success(f"✅ Downloaded: {os.path.basename(local_path)}")
            return local_path
            
        except (NoCredentialsError, PartialCredentialsError) as e:
            st.error("❌ AWS credentials missing or invalid!")
            raise
        except ClientError as e:
            st.error(f"❌ S3 Error: {str(e)}")
            raise
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            raise
    
    def download_folder_from_s3(self, s3_prefix: str, local_folder: str, show_progress: bool = False):
        """Download entire folder from S3 with optional progress tracking"""
        try:
            # Ensure local folder exists
            os.makedirs(local_folder, exist_ok=True)
            
            # List all objects in the S3 prefix
            if show_progress:
                st.info(f"🔍 Scanning S3 folder: {s3_prefix}")
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
            
            # Collect all files first
            all_files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        # Skip folders (keys ending with '/')
                        if not s3_key.endswith('/'):
                            all_files.append(s3_key)
            
            if not all_files:
                if show_progress:
                    st.warning(f"⚠️ No files found in S3 prefix: {s3_prefix}")
                return []
            
            if show_progress:
                st.info(f"📁 Found {len(all_files)} files to download")
            
            # Download files with optional progress bar
            downloaded_files = []
            progress_bar = None
            status_text = None
            
            if show_progress:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for i, s3_key in enumerate(all_files):
                # Calculate local path
                relative_path = s3_key.replace(s3_prefix, '').lstrip('/')
                local_file_path = os.path.join(local_folder, relative_path)
                
                # Update progress
                if show_progress and progress_bar and status_text:
                    progress = (i + 1) / len(all_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading: {os.path.basename(s3_key)} ({i+1}/{len(all_files)})")
                
                try:
                    # Download file (without individual progress messages)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    if not os.path.exists(local_file_path):
                        self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
                    
                    downloaded_files.append(local_file_path)
                    
                except Exception as e:
                    if show_progress:
                        st.error(f"❌ Failed to download {s3_key}: {str(e)}")
                    continue
            
            # Clear progress indicators
            if show_progress and progress_bar and status_text:
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Downloaded {len(downloaded_files)} legend images")
            
            return downloaded_files
            
        except Exception as e:
            st.error(f"❌ Error downloading folder: {str(e)}")
            raise
    
    @st.cache_resource
    def load_detection_model(_self):
        """Load detection model from S3"""
        s3_key = "development/detection-model/iteration-4_500Epoch_50Black_50Red.pt"
        local_path = str(_self.local_base_path / "development/detection-model/iteration-4_500Epoch_50Black_50Red.pt")
        
        with st.spinner("Loading detection model..."):
            return _self.download_file_from_s3(s3_key, local_path, show_progress=False)
    
    @st.cache_resource
    def load_classifier_model(_self):
        """Load classifier model files from S3"""
        s3_prefix = "development/pdf-classifier/fine-tuned-dit-20/"
        local_folder = str(_self.local_base_path / "development/pdf-classifier/fine-tuned-dit-20")
        
        # Download all files in classifier folder
        files_to_download = [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json"
        ]
        
        with st.spinner("Loading classifier model..."):
            for file in files_to_download:
                s3_key = s3_prefix + file
                local_path = os.path.join(local_folder, file)
                _self.download_file_from_s3(s3_key, local_path, show_progress=False)
        
        return local_folder
    
    @st.cache_resource
    def load_symbols(_self):
        """Load all symbol images from S3"""
        s3_prefix = "development/raw-legend/"
        local_folder = str(_self.local_base_path / "development/raw-legend")
        
        with st.spinner("Loading symbol library..."):
            downloaded_files = _self.download_folder_from_s3(s3_prefix, local_folder, show_progress=False)
            return local_folder

# Global instance
@st.cache_resource
def get_model_loader():
    """Get singleton instance of ModelLoader"""
    return ModelLoader()

# Helper function to get all paths
def get_model_paths():
    """Download all models and return paths"""
    loader = get_model_loader()
    
    # Check if models are already loaded
    if 'model_paths' not in st.session_state:
        # Show loading message only on first load
        with st.spinner("🔄 Loading models and symbols..."):
            # Download all components
            detection_model = loader.load_detection_model()
            classifier_model = loader.load_classifier_model()
            symbol_folder = loader.load_symbols()
            
            # Store paths in session state
            st.session_state.model_paths = {
                "detection_model": detection_model,
                "classifier_model": classifier_model,
                "symbol_folder": symbol_folder
            }
    
    return st.session_state.model_paths