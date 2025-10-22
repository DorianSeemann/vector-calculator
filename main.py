#!/usr/bin/env python3
"""
3D Vector Calculator - Main Entry Point

A professional 3D vector mathematics and visualization application.
"""

import sys
import os

# Add src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def main():
    """Main entry point for the 3D Vector Calculator."""
    try:
        from src.gui.vector_gui import VectorCalculatorGUI
        from PyQt5.QtWidgets import QApplication
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("3D Vector Calculator")
        app.setApplicationVersion("1.0.0")
        
        # Create and show main window
        window = VectorCalculatorGUI()
        window.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please install required dependencies:")
        print("   pip install PyQt5 matplotlib numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting 3D Vector Calculator...")
    main()