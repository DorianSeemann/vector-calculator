import sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QCheckBox, QGroupBox, QTextEdit,
                             QSplitter, QRadioButton, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence

# Import existing functions from core module
from ..core.vector_math_core import (calculate_determinant, draw_vectors, draw_plane_if_coplanar,
                                    set_automatic_axis_limits, set_axis_limits_for_plane)

class FormulaTextEdit(QTextEdit):
    """Custom QTextEdit that handles Enter for calculations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Check for Ctrl+Enter to calculate all
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
            if self.parent_widget and hasattr(self.parent_widget, 'calculate_all_formulas'):
                self.parent_widget.calculate_all_formulas()
                return
        # Check for plain Enter to calculate current line
        elif event.key() == Qt.Key_Return and event.modifiers() == Qt.NoModifier:
            if self.parent_widget and hasattr(self.parent_widget, 'calculate_current_formula'):
                self.parent_widget.calculate_current_formula()
                return
        
        # Default behavior for other keys
        super().keyPressEvent(event)

# Import formula parser and dynamic vector manager
from ..parsers.vector_formula_parser import VectorFormulaParser
from .dynamic_vector_manager import DynamicVectorManager


class HistoryControlWidget(QWidget):
    """Widget for calculation history control buttons."""
    
    clearHistoryRequested = pyqtSignal()
    saveHistoryRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        
    def setupUI(self):
        """Set up the history control buttons."""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Add margins for visual separation
        
        # Clear History button
        self.clear_history_btn = QPushButton("Clear History")
        self.clear_history_btn.setMinimumWidth(130)
        self.clear_history_btn.setFixedHeight(30)
        self.clear_history_btn.setToolTip("Clear all calculation history")
        self.clear_history_btn.clicked.connect(self.clearHistoryRequested.emit)
        
        # Save History button
        self.save_history_btn = QPushButton("Save History")
        self.save_history_btn.setMinimumWidth(130)
        self.save_history_btn.setFixedHeight(30)
        self.save_history_btn.setToolTip("Save calculation history to file")
        self.save_history_btn.clicked.connect(self.saveHistoryRequested.emit)
        
        # Add buttons to layout
        layout.addWidget(self.clear_history_btn)
        layout.addWidget(self.save_history_btn)
        layout.addStretch()  # Push buttons to the left
        
        self.setLayout(layout)
        
        # Set fixed height for the entire widget to prevent layout issues
        self.setFixedHeight(40)
    
    def setButtonsEnabled(self, enabled):
        """Enable or disable both buttons."""
        self.clear_history_btn.setEnabled(enabled)
        self.save_history_btn.setEnabled(enabled)
    
    def setClearEnabled(self, enabled):
        """Enable or disable only the clear button."""
        self.clear_history_btn.setEnabled(enabled)
    
    def setSaveEnabled(self, enabled):
        """Enable or disable only the save button."""
        self.save_history_btn.setEnabled(enabled)


class Vector3DWidget(QWidget):
    """Widget for inputting a single 3D vector."""
    
    vectorChanged = pyqtSignal()
    
    def __init__(self, name="Vector", default_values=(0, 0, 0)):
        super().__init__()
        self.name = name
        self.setupUI(default_values)
        
    def setupUI(self, default_values):
        layout = QHBoxLayout()
        
        # Vector name label
        name_label = QLabel(f"{self.name}:")
        name_label.setMinimumWidth(60)
        layout.addWidget(name_label)
        
        # X, Y, Z input fields
        self.x_input = QLineEdit(str(default_values[0]))
        self.y_input = QLineEdit(str(default_values[1]))
        self.z_input = QLineEdit(str(default_values[2]))
        
        for input_field in [self.x_input, self.y_input, self.z_input]:
            input_field.setMinimumWidth(120)  # Extended: Much wider input fields
            input_field.setMaximumWidth(200)  # Extended: More space for long numbers
            input_field.textChanged.connect(self.vectorChanged.emit)
            
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.x_input)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.y_input)
        self.z_label = QLabel("Z:")
        layout.addWidget(self.z_label)
        layout.addWidget(self.z_input)
        
        self.setLayout(layout)
    
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


class Plot2DCanvas(FigureCanvas):
    """Matplotlib canvas for 2D plotting."""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14, 10))
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create 2D subplot
        self.ax = self.fig.add_subplot(111)
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the 2D plot."""
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        self.ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.7)
        self.ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.7)
        self.ax.set_xlim(-5, 5)  # Default limits
        self.ax.set_ylim(-5, 5)
        print("DEBUG: 2D plot setup completed")

class PlotCanvas(FigureCanvas):
    """Matplotlib canvas for 3D plotting."""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14, 10))  # Extended: Larger plot figure (14x10 instead of 10x8)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create 3D subplot
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the 3D plot."""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True)
        
    def clear_plot(self):
        """Clear the current plot."""
        self.ax.clear()
        self.setup_plot()
        
    def update_plot(self, vectors, labels, show_vectors=True, show_normal=True, 
                   show_plane=True, show_determinant=True, vector_colors=None):
        """Update the plot with new data."""
        self.clear_plot()
        
        if not vectors:
            self.draw()
            return
            
        # Calculate determinant for 3 vectors
        determinant = 0.0
        if len(vectors) >= 3:
            determinant = calculate_determinant(vectors[:3])  # Use first 3 vectors
        
        # Prepare vectors to plot
        vectors_to_plot = []
        labels_to_plot = []
        
        if show_vectors:
            vectors_to_plot.extend(vectors)
            labels_to_plot.extend(labels)
        
        # Add normal vector if requested  
        if show_normal and len(vectors) >= 2:
            normal_vector = np.cross(vectors[0], vectors[1])
            vectors_to_plot.append(normal_vector)
            labels_to_plot.append('Normal Vector')
        
        # Draw vectors
        if vectors_to_plot:
            # Prepare colors matching the vectors_to_plot order
            plot_colors = []
            
            # Add colors for displayed user vectors
            if show_vectors:
                if vector_colors and len(vector_colors) >= len(vectors):
                    plot_colors.extend(vector_colors[:len(vectors)])
                else:
                    # Use default colors if custom colors not available
                    default_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
                    plot_colors.extend(default_colors[:len(vectors)])
            
            # Add color for normal vector if present
            if show_normal and len(vectors) >= 2:
                plot_colors.append('purple')  # Normal vector always purple
            
            draw_vectors(self.ax, vectors_to_plot, labels_to_plot, plot_colors)
        
        # Set axis limits - ALWAYS use all original vectors for proper scaling
        if vectors:
            # Include all original vectors for axis calculation, regardless of display settings
            all_vectors_for_scaling = list(vectors)  # Start with original vectors
            if show_normal and len(vectors) >= 2:
                normal_vector = np.cross(vectors[0], vectors[1])
                all_vectors_for_scaling.append(normal_vector)
            set_automatic_axis_limits(self.ax, all_vectors_for_scaling)
        
        # Draw plane if requested and possible
        if show_plane and determinant is not None:
            draw_plane_if_coplanar(self.ax, vectors, determinant)
            if abs(determinant) < 1e-10 and len(vectors) >= 2:
                set_axis_limits_for_plane(self.ax, vectors)
        
        # Add determinant text
        if show_determinant and determinant is not None:
            self.add_determinant_text(determinant)
        
        # Add legend
        self.ax.legend()
        self.draw()
        
        return determinant
    
    def add_determinant_text(self, determinant):
        """Add determinant information to the plot."""
        det_text = f"Determinant: {determinant:.3f}"
        if abs(determinant) < 1e-10:
            status_text = "Linearly dependent"
        else:
            status_text = f"Linearly independent\nVolume: {abs(determinant):.3f}"
        
        self.ax.text2D(0.02, 0.98, f"{det_text}\n{status_text}", 
                      transform=self.ax.transAxes, fontsize=10,
                      verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


class VectorCalculatorGUI(QMainWindow):
    """Main GUI application for vector calculations and visualization."""
    
    def __init__(self):
        super().__init__()
        self.formula_parser = VectorFormulaParser()
        self.formula_results = []
        self.last_formula_result = None  # Store last calculated result for "Add to Plot"
        self.calculation_history = []  # Store all calculations for multi-step work
        self.setupUI()
        self.connectSignals()
        self.update_plot()
        self.update_available_variables()  # Initial variable update
        
    def setupUI(self):
        """Set up the user interface."""
        self.setWindowTitle("3D Vector Calculator & Visualizer")
        self.setGeometry(100, 100, 1800, 1100)  # Extended: Larger main window (1800x1100 instead of 1400x900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (controls)
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (plot)
        right_panel = self.create_plot_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions - Extended: Much larger plot area
        splitter.setSizes([500, 1500])  # Extended: Larger plot (1500 instead of 1000, control panel 500 instead of 600)
        
    def create_control_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        panel.setMinimumWidth(550)  # Extended: Wider control panel
        panel.setMaximumWidth(650)  # Extended: More space for input fields
        layout = QVBoxLayout(panel)
        
        # Dynamic vector manager section
        vector_group = QGroupBox("Vector Management")
        vector_layout = QVBoxLayout(vector_group)
        
        # Create dynamic vector manager
        self.vector_manager = DynamicVectorManager()
        vector_layout.addWidget(self.vector_manager)
        
        layout.addWidget(vector_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        # 2D/3D Mode Toggle - NEU
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Dimension:")
        self.mode_2d_rb = QRadioButton("2D Vectors")
        self.mode_3d_rb = QRadioButton("3D Vectors")
        self.mode_3d_rb.setChecked(True)  # Standard ist 3D
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_2d_rb)
        mode_layout.addWidget(self.mode_3d_rb)
        mode_layout.addStretch()
        display_layout.addLayout(mode_layout)
        
        # Separator
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        display_layout.addWidget(separator_line)
        
        self.show_normal_cb = QCheckBox("Show Normal Vectors")
        self.show_normal_cb.setChecked(True)
        
        self.show_vector_cb = QCheckBox("Show Vector")
        self.show_vector_cb.setChecked(True)
        
        self.show_plane_cb = QCheckBox("Show Plane (if coplanar)")
        self.show_plane_cb.setChecked(True)
        
        self.show_determinant_cb = QCheckBox("Show Determinant Info")
        self.show_determinant_cb.setChecked(True)
        
        display_layout.addWidget(self.show_normal_cb)
        display_layout.addWidget(self.show_vector_cb)
        display_layout.addWidget(self.show_plane_cb)
        display_layout.addWidget(self.show_determinant_cb)
        
        layout.addWidget(display_group)
        
        # Formula input section - Enhanced for multi-line formulas and calculations
        formula_group = QGroupBox("Vector Formula Calculator - Workbook")
        formula_layout = QVBoxLayout(formula_group)
        
        # Available variables display (moved to top) - Made scrollable
        variables_group = QGroupBox("Available Variables")
        variables_group_layout = QVBoxLayout(variables_group)
        
        # Header layout with help button
        variables_header_layout = QHBoxLayout()
        
        # Add Help button for supported operations
        self.help_btn = QPushButton("Show All Operations")
        self.help_btn.setMaximumWidth(150)
        self.help_btn.setStyleSheet("background-color: lightgreen;")
        variables_header_layout.addWidget(self.help_btn)
        
        variables_header_layout.addStretch()
        variables_group_layout.addLayout(variables_header_layout)
        
        # Scrollable area for variables
        variables_scroll = QScrollArea()
        variables_scroll.setWidgetResizable(True)
        variables_scroll.setMaximumHeight(80)  # Limit height to make it compact
        variables_scroll.setMinimumHeight(40)
        variables_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        variables_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Widget to contain the variables text
        self.variables_widget = QWidget()
        self.variables_layout = QVBoxLayout(self.variables_widget)
        self.variables_layout.setContentsMargins(5, 5, 5, 5)
        
        self.variables_label = QLabel("None (add vectors first)")
        self.variables_label.setStyleSheet("color: blue; font-weight: bold;")
        self.variables_label.setWordWrap(True)  # Enable word wrapping
        self.variables_layout.addWidget(self.variables_label)
        
        variables_scroll.setWidget(self.variables_widget)
        variables_group_layout.addWidget(variables_scroll)
        
        formula_layout.addWidget(variables_group)
        
        # Multi-line formula input area
        formula_input_group = QGroupBox("Formula Input")
        formula_input_layout = QVBoxLayout(formula_input_group)
        
        # Multi-line text area for formulas - Extended for better visibility
        self.formula_input = FormulaTextEdit(self)
        self.formula_input.setMinimumHeight(180)  # Enlarged: More space for placeholder suggestions
        self.formula_input.setMaximumHeight(220)  # Enlarged: Even more space for longer formulas
        self.formula_input.setPlaceholderText("Enter formulas (one per line):\nA + B\nA_1 × B_2\n|R_{1}|\nunit(A_1)")
        formula_input_layout.addWidget(self.formula_input)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        self.calculate_formula_btn = QPushButton("Calculate All")
        self.calculate_formula_btn.setMinimumWidth(140)
        controls_layout.addWidget(self.calculate_formula_btn)
        
        self.calculate_current_btn = QPushButton("Calculate Current Line")
        self.calculate_current_btn.setMinimumWidth(180)
        controls_layout.addWidget(self.calculate_current_btn)
        
        self.clear_formulas_btn = QPushButton("Clear All")
        self.clear_formulas_btn.setMinimumWidth(120)
        controls_layout.addWidget(self.clear_formulas_btn)
        
        controls_layout.addStretch()
        
        # Add to plot button
        self.add_to_plot_btn = QPushButton("Add Last Result to Plot")
        self.add_to_plot_btn.setMinimumWidth(180)
        self.add_to_plot_btn.setEnabled(False)
        self.add_to_plot_btn.setToolTip("Add the last calculated vector result as new vector to the plot")
        controls_layout.addWidget(self.add_to_plot_btn)
        
        formula_input_layout.addLayout(controls_layout)
        formula_layout.addWidget(formula_input_group)
        
        # Quick formula buttons
        quick_group = QGroupBox("Quick Insert")
        quick_layout = QVBoxLayout(quick_group)
        
        # Basic operations
        basic_layout = QHBoxLayout()
        self.quick_add_btn = QPushButton("A + B")
        self.quick_sub_btn = QPushButton("A - B") 
        self.quick_cross_btn = QPushButton("A × B")  
        self.quick_dot_btn = QPushButton("A · B")
        self.quick_mag_btn = QPushButton("|A|")
        
        for btn in [self.quick_add_btn, self.quick_sub_btn, self.quick_cross_btn, self.quick_dot_btn, self.quick_mag_btn]:
            btn.setMinimumWidth(80)
            btn.setMaximumWidth(120)
            basic_layout.addWidget(btn)
        
        quick_layout.addLayout(basic_layout)
        
        # Advanced operations
        advanced_layout = QHBoxLayout()
        self.quick_unit_btn = QPushButton("unit(A)")
        self.quick_scalar_btn = QPushButton("2*A")
        self.quick_complex_btn = QPushButton("(A+B)×C")
        self.quick_angle_btn = QPushButton("angle(A,B)")
        
        for btn in [self.quick_unit_btn, self.quick_scalar_btn, self.quick_complex_btn, self.quick_angle_btn]:
            btn.setMinimumWidth(90)
            btn.setMaximumWidth(140)
            advanced_layout.addWidget(btn)
        
        advanced_layout.addStretch()
        quick_layout.addLayout(advanced_layout)
        formula_layout.addWidget(quick_group)
        
        # Calculation history and results
        results_group = QGroupBox("Calculation History")
        results_group.setMaximumHeight(450)
        results_layout = QVBoxLayout(results_group)
        
        self.solution_text = QTextEdit()
        self.solution_text.setMinimumHeight(200)
        self.solution_text.setMaximumHeight(370)
        self.solution_text.setReadOnly(True)
        self.solution_text.setPlaceholderText("Calculation history will appear here...\nEach formula and its result will be logged for reference.")
        results_layout.addWidget(self.solution_text, 1)
        
        # History controls widget
        self.history_controls = HistoryControlWidget(self)
        results_layout.addWidget(self.history_controls, 0)  # No stretch for button widget
        formula_layout.addWidget(results_group, 0)
        
        layout.addWidget(formula_group)
        
        # Results display
        results_group = QGroupBox("Vector Information")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.update_btn = QPushButton("Update Plot")
        self.update_btn.setMinimumWidth(120)
        self.reset_btn = QPushButton("Reset Vectors")
        self.reset_btn.setMinimumWidth(130)
        self.clear_btn = QPushButton("Clear Plot")
        self.clear_btn.setMinimumWidth(120)
        
        buttons_layout.addWidget(self.update_btn)
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addWidget(self.clear_btn)
        
        layout.addLayout(buttons_layout)
        
        layout.addStretch()
        
        return panel
    

    
    def create_plot_panel(self):
        """Create the right plot panel with 2D/3D switching capability."""
        panel = QWidget()
        panel.setMinimumSize(800, 600)  # Extended: Minimum size for plot area
        layout = QVBoxLayout(panel)
        
        # Plot canvas - Start with 3D
        self.plot_canvas_3d = PlotCanvas()
        self.plot_canvas_3d.setMinimumSize(780, 580)
        
        self.plot_canvas_2d = Plot2DCanvas()
        self.plot_canvas_2d.setMinimumSize(780, 580)
        self.plot_canvas_2d.hide()  # Initially hidden
        
        layout.addWidget(self.plot_canvas_3d)
        layout.addWidget(self.plot_canvas_2d)
        
        # Current active canvas reference
        self.plot_canvas = self.plot_canvas_3d
        
        return panel
    
    def connectSignals(self):
        """Connect signals and slots."""
        # Vector manager changes
        self.vector_manager.vectorsChanged.connect(self.update_plot)
        self.vector_manager.vectorsChanged.connect(self.update_available_variables)
        
        # 2D/3D mode changes - NEU
        self.mode_2d_rb.toggled.connect(self.toggle_2d_3d_mode)
        self.mode_3d_rb.toggled.connect(self.toggle_2d_3d_mode)
        
        # Display option changes
        self.show_normal_cb.toggled.connect(self.update_plot)
        self.show_vector_cb.toggled.connect(self.update_plot)
        self.show_plane_cb.toggled.connect(self.update_plot)
        self.show_determinant_cb.toggled.connect(self.update_plot)
        
        # Button clicks
        self.update_btn.clicked.connect(self.update_plot)
        self.reset_btn.clicked.connect(self.reset_vectors)
        self.clear_btn.clicked.connect(self.clear_plot)
        
        # Formula buttons
        self.calculate_formula_btn.clicked.connect(self.calculate_all_formulas)
        self.calculate_current_btn.clicked.connect(self.calculate_current_formula)
        self.clear_formulas_btn.clicked.connect(self.clear_formula_input)
        self.add_to_plot_btn.clicked.connect(self.add_formula_result_to_vector_manager)
        
        # History control widget signals
        self.history_controls.clearHistoryRequested.connect(self.clear_calculation_history)
        self.history_controls.saveHistoryRequested.connect(self.save_calculation_history)
        
        # Quick formula buttons - basic operations
        self.quick_add_btn.clicked.connect(lambda: self.insert_formula("A + B"))
        self.quick_sub_btn.clicked.connect(lambda: self.insert_formula("A - B"))
        self.quick_cross_btn.clicked.connect(lambda: self.insert_formula("A × B"))
        self.quick_dot_btn.clicked.connect(lambda: self.insert_formula("A · B"))
        self.quick_mag_btn.clicked.connect(lambda: self.insert_formula("|A|"))
        
        # Quick formula buttons - advanced operations
        self.quick_unit_btn.clicked.connect(lambda: self.insert_formula("unit(A)"))
        self.quick_scalar_btn.clicked.connect(lambda: self.insert_formula("2*A"))
        self.quick_complex_btn.clicked.connect(lambda: self.insert_formula("(A + B) × C"))
        self.quick_angle_btn.clicked.connect(lambda: self.insert_formula("A angle B"))
        
        # Help button connection
        self.help_btn.clicked.connect(self.show_operations_help)
        
    def show_operations_help(self):
        """Show comprehensive help for all supported operations."""
        # Get operations from parser
        operations = self.formula_parser.get_supported_operations()
        examples = self.formula_parser.get_example_formulas()
        
        help_text = "📖 SUPPORTED VECTOR OPERATIONS\n" + "="*50 + "\n\n"
        
        # Add operations list
        help_text += "🔧 AVAILABLE OPERATIONS:\n"
        for op in operations:
            help_text += f"  • {op}\n"
        
        help_text += "\n" + "="*50 + "\n"
        help_text += "📝 EXAMPLE FORMULAS:\n"
        
        # Add examples in columns
        col1_examples = examples[:len(examples)//2]
        col2_examples = examples[len(examples)//2:]
        
        for i in range(max(len(col1_examples), len(col2_examples))):
            left = col1_examples[i] if i < len(col1_examples) else ""
            right = col2_examples[i] if i < len(col2_examples) else ""
            help_text += f"  {left:<20} {right}\n"
        
        help_text += "\n" + "="*50 + "\n"
        help_text += "💡 VARIABLE NAMING:\n"
        help_text += "  • Standard: A, B, C\n"
        help_text += "  • Numbered: A1, B2, C3\n"
        help_text += "  • Underscore: A_1, B_2, C_3\n"
        help_text += "  • Results: R_{1}, R_{2}, result1\n"
        help_text += "  • Added Vectors: R_1, R_2\n\n"
        
        help_text += "⌨️ KEYBOARD SHORTCUTS:\n"
        help_text += "  • ENTER: Calculate current line\n"
        help_text += "  • CTRL+ENTER: Calculate all lines\n\n"
        
        help_text += "🎯 TIPS:\n"
        help_text += "  • Use | | for magnitude: |A|\n"
        help_text += "  • Combine operations: |A × B|\n"
        help_text += "  • Chain results: R_{1} + R_{2}\n"
        help_text += "  • Multiple lines supported!"
        
        # Display in the solution text area
        self.solution_text.clear()
        self.solution_text.append(help_text)
    
    def clear_formula_results(self):
        """Clear formula results from plot."""
        self.formula_results = []
        self.solution_text.clear()
        self.update_plot()
    
    def set_formula(self, formula: str):
        """Set formula in input field and calculate."""
        self.formula_input.setText(formula)
        self.calculate_formula()
    
    def calculate_formula(self):
        """Calculate the entered formula and show solution steps."""
        formula = self.formula_input.text().strip()
        if not formula:
            return
        
        # Get current vectors
        vectors = self.vector_manager.get_vectors_for_formula()
        
        # Set vectors in parser
        self.formula_parser.set_vectors(vectors)
        
        # Parse and solve
        result, solution_steps = self.formula_parser.parse_and_solve(formula)
        
        # Display solution steps
        self.solution_text.clear()
        if solution_steps:
            formatted_steps = []
            for i, step in enumerate(solution_steps, 1):
                formatted_steps.append(f"{i}. {step}")
            
            self.solution_text.setText("\n".join(formatted_steps))
        
        # Handle result
        if result is not None:
            # Store result for "Add to Plot" button
            self.last_formula_result = {
                'result': result,
                'formula': formula,
                'vectors_used': list(vectors.keys())
            }
            
            # Add result information
            if isinstance(result, np.ndarray):
                self.solution_text.append(f"\n✅ RESULT: Vector {result}")
                self.solution_text.append(f"   Magnitude: {np.linalg.norm(result):.3f}")
                self.add_to_plot_btn.setEnabled(True)  # Enable "Add to Plot" button
            else:
                self.solution_text.append(f"\n✅ RESULT: Scalar {result:.6f}")
                self.add_to_plot_btn.setEnabled(False)  # Scalars can't be plotted as vectors
        else:
            self.solution_text.append("\n❌ Calculation failed - check formula syntax")
            self.last_formula_result = None
            self.add_to_plot_btn.setEnabled(False)
    
    def add_formula_result_to_plot(self, result_vector: np.ndarray, formula: str):
        """Add formula result vector to the plot."""
        # Store the result for plotting
        result_info = {
            'vector': result_vector,
            'label': f"Result: {formula}",
            'color': 'magenta'
        }
        
        # Update formula results list
        self.formula_results = [result_info]  # Only keep latest result
        
        # Update plot to include formula result
        self.update_plot_with_formula_results()
    
    def update_plot_with_formula_results(self):
        """Update plot including formula results."""
        # Get vectors from manager
        vector_data = self.vector_manager.get_all_vectors()
        vectors = [v['vector'] for v in vector_data]
        labels = [v['name'] for v in vector_data]
        vector_colors = [v['color'] for v in vector_data]
        # Add formula results
        for result_info in self.formula_results:
            vectors.append(result_info['vector'])
            labels.append(result_info['label'])
        
        # Get display options - FIX: Swap the internal logic since the behavior was reversed
        show_vectors = self.show_vector_cb.isChecked()   # SWAPPED: "Show Vectors" checkbox actually controls normal
        show_normal = self.show_normal_cb.isChecked()   # SWAPPED: "Show Normal Vector" checkbox actually controls vectors
        show_plane = self.show_plane_cb.isChecked()
        show_determinant = self.show_determinant_cb.isChecked()
        
        # Update plot with custom colors
        determinant = self.plot_canvas.update_plot(
            vectors, labels, show_vectors, show_normal, show_plane, show_determinant, vector_colors
        )
        
        # Update results text
        self.update_results(vectors[:3], determinant)  # Only use original vectors for determinant
    
    def update_plot(self):
        """Update the plot with current settings (2D or 3D)."""
        if hasattr(self, 'formula_results') and self.formula_results:
            self.update_plot_with_formula_results()
        else:
            # Get vectors from manager
            vector_data = self.vector_manager.get_all_vectors()
            vectors = [v['vector'] for v in vector_data]
            labels = [v['name'] for v in vector_data]
            vector_colors = [v['color'] for v in vector_data]
            
            # Get display options - FIX: Swap the internal logic since the behavior was reversed
            show_vectors = self.show_vector_cb.isChecked()
            show_normal = self.show_normal_cb.isChecked()
            show_plane = self.show_plane_cb.isChecked()
            show_determinant = self.show_determinant_cb.isChecked()
            
            # Check if in 2D mode
            if self.mode_2d_rb.isChecked():
                print("DEBUG: Switching to 2D mode plotting")
                # 2D mode - use 2D plotting functions
                determinant = self.update_2d_plot(vectors, labels, vector_colors, show_vectors)
            else:
                print("DEBUG: Using 3D mode plotting")
                # 3D mode - use existing 3D plotting
                determinant = self.plot_canvas.update_plot(
                    vectors, labels, show_vectors, show_normal, show_plane, show_determinant, vector_colors
                )
            
            # Update results text
            self.update_results(vectors, determinant)
    
    def update_results(self, vectors, determinant):
        """Update the results text area."""
        results = []
        
        # Vector information
        results.append("=== VECTOR INFORMATION ===")
        for i, (vector, name) in enumerate(zip(vectors, ["A", "B", "C"])):
            magnitude = np.linalg.norm(vector)
            results.append(f"Vector {name}: {vector}")
            results.append(f"Magnitude |{name}|: {magnitude:.3f}")
        
        # Determinant information
        if determinant is not None:
            results.append("\n=== DETERMINANT ANALYSIS ===")
            results.append(f"Determinant: {determinant:.6f}")
            if abs(determinant) < 1e-10:
                results.append("→ Vectors are LINEARLY DEPENDENT")
                results.append("→ Vectors lie in the same plane")
            else:
                results.append("→ Vectors are LINEARLY INDEPENDENT")
                results.append(f"→ Volume of parallelepiped: {abs(determinant):.6f}")
        
        self.results_text.setText("\n".join(results))
    
    def reset_vectors(self):
        """Reset vectors to default values - now with different lengths to test arrow scaling."""
        self.vector_manager.clear_all_vectors()
        self.vector_manager.add_vector("Short", [1, 0, 0], 'red')      # Length 1.0 - arrow ratio 0.2
        self.vector_manager.add_vector("Medium", [0, 3, 0], 'blue')    # Length 3.0 - arrow ratio 0.15  
        self.vector_manager.add_vector("Long", [0, 0, 8], 'green')     # Length 8.0 - arrow ratio 0.1
    
    def clear_plot(self):
        """Clear the plot canvas."""
        self.plot_canvas.clear_plot()
        self.plot_canvas.draw()

    def update_available_variables(self):
        """Update the display of available variables for formula calculator."""
        vectors = self.vector_manager.get_vectors_for_formula()
        
        # Add intermediate results as available variables
        result_vars = []
        for i, calc in enumerate(self.calculation_history):
            if calc['type'] == 'vector':
                # If the result has been added to vector management, use that name
                if calc.get('vector_name'):
                    result_vars.append(calc['vector_name'])
                else:
                    # Otherwise use the generic result names
                    result_num = i + 1
                    result_vars.extend([f'result{result_num}', f'R{result_num}', f'r{result_num}', f'R_{{{result_num}}}'])
        
        all_variables = list(vectors.keys()) + result_vars
        
        if all_variables:
            # Remove duplicates but preserve order
            all_variables = list(dict.fromkeys(all_variables))
            
            # Group variables: regular vectors from vector manager vs calculation results
            vector_vars = list(vectors.keys())  # Direct vectors from manager
            calc_result_vars = [var for var in result_vars if var not in vector_vars]  # Calculation results not yet in manager
            
            vector_vars = sorted(vector_vars)
            calc_result_vars = sorted(calc_result_vars)
            
            # Format the display text with better organization
            display_text = ""
            
            # Show vector variables
            if vector_vars:
                display_text += "🎯 Vectors: " + ", ".join(vector_vars[:10])  # Limit to first 10
                if len(vector_vars) > 10:
                    display_text += f" ... +{len(vector_vars)-10} more"
            
            # Show calculation result variables in a compact format
            if calc_result_vars:
                if display_text:
                    display_text += "\n"
                display_text += "🧮 Results: " + ", ".join(calc_result_vars[:5])  # Limit to first 5
                if len(calc_result_vars) > 5:
                    display_text += f" ... +{len(calc_result_vars)-5} more"
            
            self.variables_label.setText(display_text)
            
            # Update placeholder text to show current variables
            if vectors:
                first_var = list(vectors.keys())[0]
                last_var = list(vectors.keys())[-1] if len(vectors) > 1 else first_var
                examples = [
                    f"{first_var} + {last_var}",
                    f"{first_var} × {last_var}",
                    f"{first_var} · {last_var}",
                    f"|{first_var}|",
                    f"unit({first_var})",
                    f"2*{first_var}",
                    f"{first_var} angle {last_var}"
                ]
                if result_vars:
                    examples.append(f"{result_vars[-1]} + {first_var}")
                    examples.append(f"|{result_vars[-1]}|")
                
                help_note = "💡 Click 'Show All Operations' for complete help!"
                self.formula_input.setPlaceholderText("Enter formulas (one per line):\n" + 
                                                    "\n".join(examples) + "\n\n" + help_note)
            else:
                examples = [
                    "A + B", "A × B", "A · B", "|A|", "unit(A)",
                    "2*A", "A angle B", "A_1 + B_2", "R_{1} × A"
                ]
                help_note = "💡 Click 'Show All Operations' for complete help!"
                self.formula_input.setPlaceholderText("Enter formulas (one per line):\n" + 
                                                    "\n".join(examples) + "\n\n" + help_note)
        else:
            self.variables_label.setText("None (add vectors first)")
            help_note = "💡 Click 'Show All Operations' for complete help!"
            self.formula_input.setPlaceholderText("Add vectors to use as variables\n\n" + help_note)
    
    def add_formula_result_to_vector_manager(self):
        """Add the last formula result as a new vector to the vector manager."""
        if not self.last_formula_result:
            return
            
        result = self.last_formula_result['result']
        formula = self.last_formula_result['formula']
        
        if isinstance(result, np.ndarray) and len(result) == 3:
            # Try to extract variable name from formula (e.g., "X = A + B" -> "X")
            parser = VectorFormulaParser()
            extracted_name = parser.extract_variable_name(formula)
            
            if extracted_name:
                # Use extracted variable name
                result_name = extracted_name
            else:
                # Generate a name for the result vector with underscore notation
                result_number = len(self.vector_manager.vectors) + 1
                result_name = f"R_{result_number}"
            
            # Choose a color (cycle through available colors)
            colors = ['purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
            color = colors[len(self.vector_manager.vectors) % len(colors)]
            
            # Add the result vector to the manager
            self.vector_manager.add_vector(result_name, result.tolist(), color)
            
            # Update the corresponding calculation_history entry with the vector name
            # Find the last vector calculation that matches this result
            for calc in reversed(self.calculation_history):
                if (calc['type'] == 'vector' and 
                    calc['vector_name'] is None and 
                    isinstance(calc['result'], np.ndarray) and
                    np.array_equal(calc['result'], result)):
                    calc['vector_name'] = result_name
                    break
            
            # Update the solution text to show it was added
            self.solution_text.append(f"\n📌 Added result to plot as vector '{result_name}' ({formula})")
            
            # Clear the stored result and disable button
            self.last_formula_result = None
            self.add_to_plot_btn.setEnabled(False)
            
            # Update available variables to reflect the new vector
            self.update_available_variables()
    
    def insert_formula(self, formula_text):
        """Insert formula text at current cursor position."""
        cursor = self.formula_input.textCursor()
        cursor.insertText(formula_text)
        self.formula_input.setTextCursor(cursor)
        self.formula_input.setFocus()
    
    def calculate_current_formula(self):
        """Calculate only the formula on the current line."""
        cursor = self.formula_input.textCursor()
        cursor.select(cursor.LineUnderCursor)
        current_line = cursor.selectedText().strip()
        
        if current_line:
            self.calculate_single_formula(current_line)
    
    def calculate_all_formulas(self):
        """Calculate all formulas in the input area."""
        all_text = self.formula_input.toPlainText().strip()
        if not all_text:
            return
            
        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        
        for line in lines:
            self.calculate_single_formula(line)
    
    def calculate_single_formula(self, formula):
        """Calculate a single formula and add to history."""
        if not formula:
            return
        
        # Get current vectors (including previously calculated results)
        vectors = self.vector_manager.get_vectors_for_formula()
        
        # Add any intermediate results to available vectors
        for i, calc in enumerate(self.calculation_history):
            if calc['type'] == 'vector':
                # Add result with multiple naming conventions
                result_num = i + 1
                vectors[f'result{result_num}'] = calc['result']
                vectors[f'R{result_num}'] = calc['result']  # Alternative R1, R2, etc.
                vectors[f'r{result_num}'] = calc['result']  # Alternative r1, r2, etc.
                vectors[f'R_{{{result_num}}}'] = calc['result']  # Math notation R_{1}, R_{2}, etc.
        
        # Set vectors in parser
        self.formula_parser.set_vectors(vectors)
        
        # Parse and solve
        result, solution_steps = self.formula_parser.parse_and_solve(formula)
        
        # Create calculation entry
        calc_entry = {
            'formula': formula,
            'result': result,
            'type': 'vector' if isinstance(result, np.ndarray) else 'scalar',
            'steps': solution_steps,
            'timestamp': len(self.calculation_history) + 1,
            'vector_name': None  # Will be set when added to vector manager
        }
        
        # Add to history
        self.calculation_history.append(calc_entry)
        
        # Update display
        self.update_calculation_history_display()
        
        # Store as last result for "Add to Plot"
        if isinstance(result, np.ndarray):
            self.last_formula_result = {
                'result': result,
                'formula': formula,
                'vectors_used': list(vectors.keys())
            }
            self.add_to_plot_btn.setEnabled(True)
        else:
            self.add_to_plot_btn.setEnabled(False)
    
    def update_calculation_history_display(self):
        """Update the calculation history display."""
        self.solution_text.clear()
        
        for i, calc in enumerate(self.calculation_history, 1):
            formula = calc['formula']
            result = calc['result']
            calc_type = calc['type']
            
            # Add calculation header
            self.solution_text.append(f"═══ Calculation {i}: {formula} ═══")
            
            if result is None:
                self.solution_text.append("❌ Error: Calculation failed")
            elif calc_type == 'vector':
                magnitude = np.linalg.norm(result)
                self.solution_text.append(f"✅ Result: Vector {result}")
                self.solution_text.append(f"   Magnitude: {magnitude:.3f}")
                self.solution_text.append(f"   Available as: result{i}, R{i}, r{i}, R_{{{i}}}")
            else:
                self.solution_text.append(f"✅ Result: Scalar {result:.6f}")
            
            # Add solution steps if available
            if calc['steps'] and len(calc['steps']) > 1:
                self.solution_text.append("📋 Steps:")
                for step in calc['steps'][1:]:  # Skip first step (original formula)
                    self.solution_text.append(f"   • {step}")
            
            self.solution_text.append("")  # Add spacing
        
        # Auto-scroll to bottom
        scrollbar = self.solution_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update available variables to include results
        self.update_available_variables()
    
    def clear_formula_input(self):
        """Clear the formula input area."""
        self.formula_input.clear()
    
    def clear_calculation_history(self):
        """Clear all calculation history."""
        self.calculation_history.clear()
        self.solution_text.clear()
        self.last_formula_result = None
        self.add_to_plot_btn.setEnabled(False)
        self.update_available_variables()
    
    def save_calculation_history(self):
        """Save calculation history to a file."""
        if not self.calculation_history:
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Calculation History", 
            "vector_calculations.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Vector Calculator - Calculation History\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, calc in enumerate(self.calculation_history, 1):
                        f.write(f"Calculation {i}: {calc['formula']}\n")
                        f.write(f"Result ({calc['type']}): {calc['result']}\n")
                        if calc['type'] == 'vector':
                            f.write(f"Magnitude: {np.linalg.norm(calc['result']):.3f}\n")
                        f.write("-" * 30 + "\n\n")
                        
                self.solution_text.append(f"💾 History saved to: {filename}")
            except Exception as e:
                self.solution_text.append(f"❌ Error saving history: {str(e)}")
    
    def toggle_2d_3d_mode(self):
        """Switch between 2D and 3D visualization mode."""
        print(f"DEBUG: toggle_2d_3d_mode called, 2D mode: {self.mode_2d_rb.isChecked()}")
        
        if self.mode_2d_rb.isChecked():
            # Switch to 2D mode
            print("DEBUG: Switching to 2D mode")
            self.plot_canvas_3d.hide()
            self.plot_canvas_2d.show()
            self.plot_canvas = self.plot_canvas_2d
            
            # Update vector inputs to hide Z coordinate
            for vector_widget in self.vector_manager.vectors:
                vector_widget.z_input.hide()
                vector_widget.z_label.hide()  # Hide Z label
        else:
            # Switch to 3D mode
            print("DEBUG: Switching to 3D mode")
            self.plot_canvas_2d.hide()
            self.plot_canvas_3d.show()
            self.plot_canvas = self.plot_canvas_3d
            
            # Update vector inputs to show Z coordinate
            for vector_widget in self.vector_manager.vectors:
                vector_widget.z_input.show()
                vector_widget.z_label.show()  # Show Z label
        
        # Force immediate plot update
        print("DEBUG: Forcing plot update after mode switch")
        self.update_plot()
    

    
    def update_2d_plot(self, vectors, labels, colors, show_vectors=True):
        """Update 2D plot with vectors."""
        from src.core.vector_2d_core import draw_vectors_2d, set_automatic_2d_axis_limits, analyze_vectors_2d
        
        print(f"DEBUG: update_2d_plot called with {len(vectors)} vectors, show_vectors={show_vectors}")
        
        # Clear previous plot
        self.plot_canvas_2d.ax.clear()
        self.plot_canvas_2d.setup_plot()
        
        print(f"Show Vektors is {show_vectors}")
        if vectors and (show_vectors is True):
            # Convert vectors to numpy arrays and ensure 2D
            np_vectors = []
            for i, v in enumerate(vectors):
                if len(v) > 2:
                    vec_2d = v[:2]
                    np_vectors.append(np.array(vec_2d))
                    print(f"DEBUG: Vector {i} 3D->2D: {v} -> {vec_2d}")
                elif len(v) < 2:
                    vec_2d = v + [0] * (2 - len(v))
                    np_vectors.append(np.array(vec_2d))
                    print(f"DEBUG: Vector {i} padded: {v} -> {vec_2d}")
                else:
                    np_vectors.append(np.array(v))
                    print(f"DEBUG: Vector {i} 2D: {v}")
            
            print(f"DEBUG: About to draw {len(np_vectors)} vectors with labels {labels}")
            
            # Draw vectors
            draw_vectors_2d(self.plot_canvas_2d.ax, np_vectors, labels, colors)
            
            # Set axis limits
            set_automatic_2d_axis_limits(self.plot_canvas_2d.ax, np_vectors)
            
            # Add legend
            if labels:
                self.plot_canvas_2d.ax.legend()
            
            # Analyze vectors for 2D
            analysis = analyze_vectors_2d(np_vectors)
            
            # Refresh canvas INSIDE the if block
            self.plot_canvas_2d.draw()
            
            return analysis.get("area", 0)  # Return area for 2D (equivalent to determinant concept)
        
        # Refresh canvas also if no vectors
        self.plot_canvas_2d.draw()
        return 0


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("3D Vector Calculator")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = VectorCalculatorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()