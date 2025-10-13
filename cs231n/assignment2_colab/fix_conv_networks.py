#!/usr/bin/env python3
"""
ConvolutionalNetworks.ipynb의 Google Colab 코드를 로컬용으로 수정
"""

import nbformat
import re

def fix_conv_networks():
    """ConvolutionalNetworks.ipynb 수정"""
    
    with open('ConvolutionalNetworks.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # 첫 번째 셀 (Google Colab 설정)
            if 'google.colab' in cell.source and 'drive.mount' in cell.source:
                cell.source = """# Local environment setup (no Google Colab needed)
import sys
import os

# Add current directory to Python path
sys.path.append('.')

# Check if datasets exist, if not download them
if not os.path.exists('cs231n/datasets/cifar-10-batches-py'):
    print("Downloading CIFAR-10 dataset...")
    os.chdir('cs231n/datasets/')
    os.system('bash get_datasets.sh')
    os.chdir('../..')
    print("Dataset download complete!")
else:
    print("CIFAR-10 dataset already exists!")"""
                print("✅ 첫 번째 셀 수정 완료")
            
            # Cython 컴파일 셀
            elif 'Remember to restart the runtime' in cell.source:
                cell.source = """# Local environment setup - compile Cython extensions
import os
import subprocess

# Change to cs231n directory and compile Cython extensions
os.chdir('cs231n/')
try:
    subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'], check=True)
    print("✅ Cython extensions compiled successfully!")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Cython compilation failed: {e}")
    print("Continuing without optimized extensions...")
except FileNotFoundError:
    print("⚠️ setup.py not found, continuing without Cython extensions...")

# Return to parent directory
os.chdir('..')"""
                print("✅ Cython 컴파일 셀 수정 완료")
    
    with open('ConvolutionalNetworks.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("🎉 ConvolutionalNetworks.ipynb 수정 완료!")

if __name__ == "__main__":
    fix_conv_networks()
