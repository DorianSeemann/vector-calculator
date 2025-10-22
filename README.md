# 3D Vector Calculator & Visualizer

A comprehensive Python application for 3D vector mathematics with real-time visualization and advanced calculation features.

## Features

### 🎯 **Vector Management**
- Interactive vector input with real-time magnitude calculation
- Dynamic vector addition, deletion, and modification
- Color customization for each vector
- Visibility toggles for individual vectors
- Support for named variables (A, B, C, X_1, etc.)

### 📐 **Mathematical Operations**
- **Basic Operations**: Addition, subtraction, scalar multiplication
- **Advanced Operations**: Cross product, dot product, magnitude, normalization
- **Angle Calculation**: Between any two vectors
- **Determinant**: Automatic calculation for linear independence
- **Plane Detection**: Automatic coplanarity detection and visualization

### 🧮 **Formula Calculator**
- Multi-line formula input with syntax highlighting
- Variable assignment support (`X = A + B`)
- ENTER key for quick calculations
- Comprehensive operator support:
  - `+`, `-` (addition, subtraction)
  - `×`, `x`, `cross()` (cross product)  
  - `·`, `.`, `dot()` (dot product)
  - `|A|`, `mag()`, `norm()` (magnitude)
  - `unit()`, `normalize()` (normalization)
  - `angle()` (angle between vectors)
- Result history with step-by-step solutions
- Automatic result naming and management

### 📊 **Visualization**
- **3D Mode**: Full 3D visualization with interactive rotation
- **2D Mode**: Automatic projection for 2D analysis
- **Display Options**:
  - Show/hide individual vectors
  - Show/hide normal vectors
  - Show/hide coplanar planes
  - Show/hide determinant information
- Automatic axis scaling and centering
- Color-coded vectors with labels

### 💾 **Data Management**
- Save/load calculation history
- Export results to text files
- Persistent vector configurations

## Installation

### Prerequisites
```bash
pip install numpy matplotlib PyQt5
```

### Running the Application
```bash
python main.py
```

## Usage

### Basic Vector Operations
1. **Add Vectors**: Use the vector management panel to input X, Y, Z coordinates
2. **Visualize**: Vectors are automatically plotted in 3D space
3. **Calculate**: Use the formula calculator for complex operations

### Formula Examples
```python
# Basic operations
A + B                    # Vector addition
A - B                    # Vector subtraction  
2*A                      # Scalar multiplication

# Advanced operations
A × B                    # Cross product
A · B                    # Dot product
|A|                      # Magnitude
unit(A)                  # Unit vector
A angle B                # Angle between vectors

# Variable assignment
X = A + B                # Creates vector X
Normal = A × B           # Creates normal vector
```

### Supported Variable Formats
- Simple: `A`, `B`, `C`
- Indexed: `A1`, `B2`, `C3`
- Subscript: `A_1`, `B_2`, `C_3`
- Results: `R1`, `R_2`, `R_{3}`

## Technical Details

### Architecture
- **Core**: `src/core/` - Mathematical operations and plotting functions
- **GUI**: `src/gui/` - User interface components
- **Parsers**: `src/parsers/` - Formula parsing and evaluation

### Key Components
- `vector_math_core.py`: Core mathematical functions
- `vector_gui.py`: Main application interface
- `dynamic_vector_manager.py`: Vector management system
- `vector_formula_parser.py`: Formula parsing engine

### Mathematical Operations
- **Determinant**: Uses column-wise matrix construction for accurate calculation
- **Cross Product**: Right-hand rule implementation
- **Angle Calculation**: Handles edge cases (parallel, antiparallel vectors)
- **Automatic Scaling**: Smart axis adjustment based on vector magnitudes

## Development

### Code Structure
```
src/
├── core/
│   ├── vector_math_core.py     # Mathematical operations
│   └── vector_2d_core.py       # 2D-specific functions
├── gui/
│   ├── vector_gui.py           # Main interface
│   └── dynamic_vector_manager.py # Vector management
└── parsers/
    └── vector_formula_parser.py # Formula evaluation
```

### Key Features Implementation
- **Real-time Updates**: Event-driven UI updates
- **Memory Management**: Efficient vector storage and calculation
- **Error Handling**: Comprehensive input validation
- **Extensibility**: Modular design for easy feature addition

## License

This project is developed for educational purposes in the context of Mathematics 1 for Electrical Engineering.

## Version History

- **v2.0**: Added formula calculator, variable assignment, and comprehensive help system
- **v1.5**: Implemented 2D/3D mode switching and advanced mathematical operations  
- **v1.0**: Basic 3D vector visualization and management

---

**Built with Python, NumPy, Matplotlib, and PyQt5**