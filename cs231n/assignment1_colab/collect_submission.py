#!/usr/bin/env python3
"""
Collect submission files for CS231n Assignment 1
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

# Files to include in the submission
CODE_FILES = [
    "cs231n/classifiers/k_nearest_neighbor.py",
    "cs231n/classifiers/linear_classifier.py", 
    "cs231n/classifiers/softmax.py",
    "cs231n/classifiers/fc_net.py",
    "cs231n/optim.py",
    "cs231n/solver.py",
    "cs231n/layers.py",
]

NOTEBOOKS = [
    "knn.ipynb",
    "softmax.ipynb", 
    "two_layer_net.ipynb",
    "features.ipynb",
    "FullyConnectedNets.ipynb"
]

ZIP_FILENAME = "a1_code_submission.zip"
PDF_FILENAME = "a1_inline_submission.pdf"

def check_files_exist():
    """Check that all required files exist"""
    all_files = CODE_FILES + NOTEBOOKS
    
    for file_path in all_files:
        if not os.path.exists(file_path):
            print(f"Required file {file_path} not found, Exiting.")
            return False
    return True

def create_zip_file():
    """Create the zip file with all required files"""
    print("### Zipping file ###")
    
    # Remove existing zip file
    if os.path.exists(ZIP_FILENAME):
        os.remove(ZIP_FILENAME)
    
    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add notebooks
        for notebook in NOTEBOOKS:
            if os.path.exists(notebook):
                zipf.write(notebook)
                print(f"Added {notebook}")
        
        # Add Python files
        for py_file in CODE_FILES:
            if os.path.exists(py_file):
                zipf.write(py_file)
                print(f"Added {py_file}")
        
        # Add all other Python files in the directory
        for root, dirs, files in os.walk('.'):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and file != 'makepdf.py':
                    file_path = os.path.join(root, file)
                    if file_path not in CODE_FILES:  # Don't add files we already added
                        zipf.write(file_path)
                        print(f"Added {file_path}")
        
        # Add saved directory if it exists
        if os.path.exists('cs231n/saved'):
            for root, dirs, files in os.walk('cs231n/saved'):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path)
                    print(f"Added {file_path}")

def create_pdf():
    """Create the PDF from notebooks"""
    print("### Creating PDFs ###")
    
    # Check if makepdf.py exists
    if not os.path.exists('makepdf.py'):
        print("makepdf.py not found, skipping PDF creation")
        return
    
    # Run makepdf.py
    try:
        cmd = [sys.executable, 'makepdf.py', '--notebooks'] + NOTEBOOKS + ['--pdf_filename', PDF_FILENAME]
        subprocess.run(cmd, check=True)
        print(f"PDF created: {PDF_FILENAME}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating PDF: {e}")
    except FileNotFoundError:
        print("makepdf.py not found or not executable")

def main():
    """Main function"""
    print("Collecting CS231n Assignment 1 submission files...")
    
    # Check that all required files exist
    if not check_files_exist():
        sys.exit(1)
    
    # Create zip file
    create_zip_file()
    
    # Create PDF
    create_pdf()
    
    print(f"### Done! Please submit {ZIP_FILENAME} and {PDF_FILENAME} to Gradescope. ###")

if __name__ == "__main__":
    main()

