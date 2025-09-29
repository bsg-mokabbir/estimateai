# PDF Symbol Matching & Counting System

A Streamlit application for extracting pages from engineering PDFs, detecting symbols in legend pages, and counting them across AutoCAD drawings.

## Features

- **PDF Processing**: Extract and categorize pages (Legend, Circuit, AutoCAD, Others)
- **Symbol Detection**: AI-powered symbol detection in legend pages
- **Symbol Selection**: Interactive selection of symbols to count
- **Symbol Counting**: Automated counting across AutoCAD pages
- **Manual Editing**: Plus/minus buttons for count corrections
- **Excel Export**: Generate summary reports

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Run Application**:
   ```bash
   streamlit run main_app.py
   ```

2. **Upload PDF**: Choose engineering drawing PDF in Tab 1

3. **Detect Symbols**: Go to Tab 2, run detection on legend pages, select symbols

4. **Count Symbols**: Go to Tab 3, count selected symbols in AutoCAD pages

5. **Export Results**: Download Excel summary report


