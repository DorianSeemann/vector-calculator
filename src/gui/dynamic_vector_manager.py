"""
Dynamic Vector Manager Widget
Allows adding/removing vectors dynamically and controlling their visibility.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QCheckBox,
                             QScrollArea, QColorDialog)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor


class DynamicVectorWidget(QWidget):
    """Widget for a single dynamic vector with visibility and color controls."""
    
    vectorChanged = pyqtSignal()
    visibilityChanged = pyqtSignal()
    removeRequested = pyqtSignal(object)  # Pass self reference
    
    def __init__(self, name="Vector", default_values=(0, 0, 0), color='red', vector_id=0):
        super().__init__()
        self.vector_id = vector_id
        self.name = name
        self.color = color
        self.setupUI(default_values)
        
    def setupUI(self, default_values):
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Visibility checkbox
        self.visible_cb = QCheckBox()
        self.visible_cb.setChecked(True)
        self.visible_cb.setToolTip(f"Show/Hide {self.name}")
        self.visible_cb.toggled.connect(self.visibilityChanged.emit)
        layout.addWidget(self.visible_cb)
        
        # Vector name label (editable)
        self.name_input = QLineEdit(self.name)
        self.name_input.setMaximumWidth(60)
        self.name_input.editingFinished.connect(self._update_name)  # Use editingFinished instead of textChanged for better performance
        layout.addWidget(self.name_input)
        
        # Color button
        self.color_btn = QPushButton()
        self.color_btn.setMaximumWidth(30)
        self.color_btn.setMaximumHeight(25)
        self.color_btn.setStyleSheet(f"background-color: {self.color}; border: 1px solid black;")
        self.color_btn.clicked.connect(self.choose_color)
        layout.addWidget(self.color_btn)
        
        # X, Y, Z input fields
        self.x_input = QLineEdit(str(default_values[0]))
        self.y_input = QLineEdit(str(default_values[1]))
        self.z_input = QLineEdit(str(default_values[2]))
        
        for input_field in [self.x_input, self.y_input, self.z_input]:
            input_field.setMaximumWidth(60)
            input_field.textChanged.connect(self.vectorChanged.emit)
            input_field.textChanged.connect(self.update_magnitude)
            
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.x_input)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.y_input)
        self.z_label = QLabel("Z:")
        layout.addWidget(self.z_label)
        layout.addWidget(self.z_input)
        
        # Magnitude display
        self.magnitude_label = QLabel("0.000")
        self.magnitude_label.setMinimumWidth(50)
        self.magnitude_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(QLabel("|V|:"))
        layout.addWidget(self.magnitude_label)
        
        # Remove button
        self.remove_btn = QPushButton("×")
        self.remove_btn.setMaximumWidth(25)
        self.remove_btn.setMaximumHeight(25)
        self.remove_btn.setToolTip(f"Remove {self.name}")
        self.remove_btn.setStyleSheet("color: red; font-weight: bold;")
        self.remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))
        layout.addWidget(self.remove_btn)
        
        self.setLayout(layout)
        self.update_magnitude()
    
    def _update_name(self):
        """Update vector name."""
        self.name = self.name_input.text()
        self.vectorChanged.emit()  # Signal that vector info changed
    
    def choose_color(self):
        """Open color chooser dialog."""
        color = QColorDialog.getColor(QColor(self.color), self, f"Choose color for {self.name}")
        if color.isValid():
            self.color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {self.color}; border: 1px solid black;")
            self.vectorChanged.emit()
    
    def get_vector(self):
        """Get the vector as numpy array."""
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            return np.array([x, y, z])
        except ValueError:
            return np.array([0, 0, 0])
    
    def set_vector(self, vector):
        """Set the vector values."""
        self.x_input.setText(str(vector[0]))
        self.y_input.setText(str(vector[1]))
        self.z_input.setText(str(vector[2]))
        self.update_magnitude()
    
    def update_magnitude(self):
        """Update magnitude display."""
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            vector = np.array([x, y, z])
            magnitude = np.linalg.norm(vector)
            self.magnitude_label.setText(f"{magnitude:.3f}")
        except ValueError:
            self.magnitude_label.setText("0.000")
    
    def is_visible(self):
        """Check if vector is visible."""
        return self.visible_cb.isChecked()
    
    def set_visible(self, visible):
        """Set vector visibility."""
        self.visible_cb.setChecked(visible)
    
    def get_name(self):
        """Get current vector name."""
        return self.name_input.text()
    
    def get_color(self):
        """Get current vector color."""
        return self.color


class DynamicVectorManager(QWidget):
    """Manager for multiple dynamic vectors."""
    
    vectorsChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.vectors = []
        self.next_id = 0
        self.setupUI()
        
        # Add default vectors (these should match the reset_vectors in GUI)
        self.add_vector("A", [2, 1, 0], 'red')
        self.add_vector("B", [1, 2, 0], 'blue') 
        self.add_vector("C", [3, 3, 0], 'green')
        
    def setupUI(self):
        layout = QVBoxLayout()
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.add_vector_btn = QPushButton("+ Add Vector")
        self.add_vector_btn.clicked.connect(self.add_default_vector)
        controls_layout.addWidget(self.add_vector_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_vectors)
        controls_layout.addWidget(self.clear_all_btn)
        
        self.show_all_btn = QPushButton("Show All")
        self.show_all_btn.clicked.connect(self.show_all_vectors)
        controls_layout.addWidget(self.show_all_btn)
        
        self.hide_all_btn = QPushButton("Hide All")
        self.hide_all_btn.clicked.connect(self.hide_all_vectors)
        controls_layout.addWidget(self.hide_all_btn)
        
        layout.addLayout(controls_layout)
        
        # Scroll area for vectors
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMaximumHeight(300)
        
        self.vectors_widget = QWidget()
        self.vectors_layout = QVBoxLayout(self.vectors_widget)
        
        self.scroll_area.setWidget(self.vectors_widget)
        layout.addWidget(self.scroll_area)
        
        # Vector count info
        self.count_label = QLabel("Vectors: 0")
        layout.addWidget(self.count_label)
        
        self.setLayout(layout)
        
    def add_vector(self, name=None, values=None, color=None):
        """Add a new vector."""
        if name is None:
            name = f"V{self.next_id + 1}"
        if values is None:
            values = [0, 0, 0]
        if color is None:
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            color = colors[len(self.vectors) % len(colors)]
        
        vector_widget = DynamicVectorWidget(name, values, color, self.next_id)
        vector_widget.vectorChanged.connect(self.vectorsChanged.emit)
        vector_widget.visibilityChanged.connect(self.vectorsChanged.emit)
        vector_widget.removeRequested.connect(self.remove_vector)
        
        self.vectors.append(vector_widget)
        self.vectors_layout.addWidget(vector_widget)
        
        self.next_id += 1
        self.update_count_label()
        self.vectorsChanged.emit()
        
    def add_default_vector(self):
        """Add a new default vector."""
        self.add_vector()
        
    def remove_vector(self, vector_widget):
        """Remove a vector widget."""
        if vector_widget in self.vectors:
            self.vectors.remove(vector_widget)
            self.vectors_layout.removeWidget(vector_widget)
            vector_widget.deleteLater()
            self.update_count_label()
            self.vectorsChanged.emit()
    
    def clear_all_vectors(self):
        """Remove all vectors."""
        for vector_widget in self.vectors[:]:
            self.remove_vector(vector_widget)
    
    def show_all_vectors(self):
        """Show all vectors."""
        for vector_widget in self.vectors:
            vector_widget.set_visible(True)
        self.vectorsChanged.emit()
    
    def hide_all_vectors(self):
        """Hide all vectors."""
        for vector_widget in self.vectors:
            vector_widget.set_visible(False)
        self.vectorsChanged.emit()
    
    def update_count_label(self):
        """Update vector count label."""
        visible_count = sum(1 for v in self.vectors if v.is_visible())
        total_count = len(self.vectors)
        self.count_label.setText(f"Vectors: {visible_count}/{total_count} visible")
    
    def get_all_vectors(self):
        """Get all vectors as list of dicts with metadata."""
        vectors_data = []
        for vector_widget in self.vectors:
            if vector_widget.is_visible():
                vectors_data.append({
                    'vector': vector_widget.get_vector(),
                    'name': vector_widget.get_name(),
                    'color': vector_widget.get_color(),
                    'id': vector_widget.vector_id
                })
        return vectors_data
    
    def get_vectors_for_formula(self):
        """Get vectors dictionary for formula parser."""
        vectors_dict = {}
        for vector_widget in self.vectors:
            name = vector_widget.get_name()
            if name and vector_widget.is_visible():
                vectors_dict[name] = vector_widget.get_vector()
        return vectors_dict
    
    def get_vector_by_name(self, name):
        """Get vector by name."""
        for vector_widget in self.vectors:
            if vector_widget.get_name() == name:
                return vector_widget.get_vector()
        return None