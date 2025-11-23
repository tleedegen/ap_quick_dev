""" File for creating global path variable"""
import os
import sys

# Determine the base path for accessing the 'graphics' directory
if getattr(sys, 'frozen', False):
    # If the application is frozen (bundled executable), use the temporary directory
    base_path = os.path.join(sys._MEIPASS)  # Parent directory of the bundled temp directory
else:
    # Otherwise, use the parent directory of the directory of the script
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# print(f'Base Path:{base_path}')