from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_determinant(vectors: List[np.ndarray]) -> float:
    """
    Calculate the determinant of three 3D vectors.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of three 3D vectors as NumPy arrays.
        
    Returns
    -------
    float
        The determinant of the 3x3 matrix formed by the vectors.
        
    Examples
    --------
    >>> a = np.array([1, 0, 0])
    >>> b = np.array([0, 1, 0]) 
    >>> c = np.array([0, 0, 1])
    >>> calculate_determinant([a, b, c])
    1.0
    """
    # Build matrix with vectors as columns (not rows)
    matrix = np.column_stack(vectors)  # This is the correct way for 3D vectors
    det = np.linalg.det(matrix)
    return det

def print_determinant_info(determinant: float) -> bool:
    """
    Print information about the determinant and check for linear dependence.
    
    Parameters
    ----------
    determinant : float
        The determinant value.
        
    Returns
    -------
    bool
        True if vectors are linearly dependent (det ≈ 0), False otherwise.
        
    Examples
    --------
    >>> print_determinant_info(0.0)
    Determinant: 0.0
    → Vectors are linearly dependent (lie in a plane)
    True
    """
    print(f"Determinante: {determinant}")
    if abs(determinant) < 1e-10:
        print("→ Vectors are linearly dependent (lie in a plane)")
        return True  # linearly dependent
    else:
        print(f"→ Vectors are linearly independent, volume: {abs(determinant)}")
        return False  # linearly independent

def draw_vectors(ax: Axes3D, vectors: List[np.ndarray], labels: Optional[List[str]] = None, 
                colors: Optional[List[str]] = None) -> None:
    """
    Draw vectors as arrows from the origin in a 3D coordinate system with beautiful arrow heads.
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object for drawing.
    vectors : List[np.ndarray]
        List of 3D vectors as NumPy arrays.
    labels : Optional[List[str]], default=None
        Labels for the vectors. If None, default labels are used.
    colors : Optional[List[str]], default=None
        Colors for the vectors. If None, default colors are used.
        
    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> vectors = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    >>> draw_vectors(ax, vectors, labels=['X-Axis', 'Y-Axis'])
    """
    origin = np.zeros(3)
    default_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    default_labels = [f'Vektor {i+1}' for i in range(len(vectors))]
    
    if colors is None:
        colors = default_colors[:len(vectors)]
    if labels is None:
        labels = default_labels[:len(vectors)]
    
    for i, (vector, color, label) in enumerate(zip(vectors, colors, labels)):
        norm = np.linalg.norm(vector)
        
        # Enhanced arrow styling with beautiful 3D arrow heads
        # Calculate dynamic arrow head size - smaller for longer vectors to prevent huge arrows
        if norm <= 1.0:
            arrow_head_ratio = 0.2  # Standard size for short vectors
        elif norm <= 5.0:
            arrow_head_ratio = 0.15  # Slightly smaller for medium vectors
        elif norm <= 10.0:
            arrow_head_ratio = 0.1   # Smaller for long vectors
        else:
            arrow_head_ratio = 0.08  # Very small for very long vectors
        
        # Special styling for normal vectors (distinguished by label containing "Normal")
        if label and "Normal" in label:
            # Normal vectors get distinctive styling with same arrow size as regular vectors
            ax.quiver(*origin, *vector, color=color, label=label,
                     arrow_length_ratio=arrow_head_ratio,  # Same dynamic arrow head size as regular vectors
                     linewidth=4.0,                       # Extra thick shaft for distinction
                     alpha=0.95,                          # Maximum visibility
                     normalize=False, 
                     pivot='tail',
                     edgecolor='black',                   # Strong black outline
                     linewidths=1.2,                      # Thick outline for emphasis
                     linestyle='--')                      # Dashed line for distinction
        else:
            # Regular vectors with enhanced styling - KORRIGIERT: Verwende direkten Vektor statt direction+length
            ax.quiver(*origin, *vector, color=color, label=label,
                     arrow_length_ratio=arrow_head_ratio,  # Dynamic proportional arrow heads
                     linewidth=3.5,                        # Bold, prominent arrow shafts  
                     alpha=0.9,                           # High visibility
                     normalize=False,                     # Preserve vector magnitude scaling
                     pivot='tail')                        # Arrow originates from origin

def draw_plane_if_coplanar(ax : Axes3D, vectors: List[np.ndarray], determinant: float) -> None:
    """
    Draw a plane if the vectors are coplanar (determinant ≈ 0).
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object for drawing.
    vectors : List[np.ndarray]
        List of 3D vectors as NumPy arrays.
    determinant : float
        The determinant of the vectors.
        
    Notes
    -----
    The plane is only drawn if |determinant| < 1e-10 and at least two vectors 
    are present. The first two vectors are used as basis for the plane 
    parametrization.
    """
    if abs(determinant) < 1e-10 and len(vectors) >= 2:
        print("Drawing plane - vectors are coplanar!")
        
        # Use the first two vectors as basis for the plane
        v1, v2 = vectors[0], vectors[1]
        
        # Larger range for better visibility
        u_range = np.linspace(-2.0, 2.0, 20)
        v_range = np.linspace(-2.0, 2.0, 20)
        U, V = np.meshgrid(u_range, v_range)
        
        # Plane points: P = u*v1 + v*v2
        X = U * v1[0] + V * v2[0]
        Y = U * v1[1] + V * v2[1]  
        Z = U * v1[2] + V * v2[2]
        
        # Improved plane representation in conspicuous color
        ax.plot_surface(X, Y, Z, alpha=0.5, color='yellow', 
                       linewidth=0, antialiased=True, shade=True)
        
        # Additionally: Wireframe for better visibility
        ax.plot_wireframe(X, Y, Z, alpha=0.3, color='orange', linewidth=0.8)
    else:
        print(f"Keine Ebene: determinant={determinant:.6f}, vectors={len(vectors)}")

def set_automatic_axis_limits(ax: Axes3D, vectors: List[np.ndarray]) -> None:
    """
    Set automatic axis limits based on the extreme values of the vectors.
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object.
    vectors : List[np.ndarray]
        List of 3D vectors as NumPy arrays.
        
    Notes
    -----
    The function calculates the minimum and maximum values of all vector components
    and adds a 10% buffer to ensure optimal display.
    """
    if not vectors:
        # Default limits if no vectors
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        return
        
    # Include origin (0,0,0) in calculations to ensure it's always visible
    all_vectors = np.array(vectors)
    origin = np.array([[0, 0, 0]])
    vectors_with_origin = np.vstack([all_vectors, origin])
    
    min_vals = np.min(vectors_with_origin, axis=0)
    max_vals = np.max(vectors_with_origin, axis=0)
    
    # Add buffer (10% of range or at least 0.5)
    ranges = max_vals - min_vals
    buffer = np.maximum(0.1 * ranges, 0.5)  # At least 0.5 buffer for all axes
    
    # Sicherstellen, dass alle Achsen sichtbare Bereiche haben
    for i in range(3):
        if ranges[i] < 1e-10:  # Sehr kleiner oder null Bereich
            min_vals[i] -= 1.0
            max_vals[i] += 1.0
            buffer[i] = 0.1
    
    ax.set_xlim([min_vals[0] - buffer[0], max_vals[0] + buffer[0]])
    ax.set_ylim([min_vals[1] - buffer[1], max_vals[1] + buffer[1]])
    ax.set_zlim([min_vals[2] - buffer[2], max_vals[2] + buffer[2]])

def set_axis_limits_for_plane(ax : Axes3D, vectors: List[np.ndarray]) -> None:
    """
    Expand axis limits to ensure the plane spanned by vectors is fully visible.
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object.
    vectors : List[np.ndarray]
        List of 3D vectors as NumPy arrays.
    """
    if len(vectors) >= 2:
        # Berechne Ebenen-Ausdehnung
        a, b = vectors[0], vectors[1]
        u_range = np.linspace(-2.0, 2.0, 10)
        v_range = np.linspace(-2.0, 2.0, 10)
        U, V = np.meshgrid(u_range, v_range)
        
        # Alle Punkte der Ebene
        X = U * a[0] + V * b[0]
        Y = U * a[1] + V * b[1]
        Z = U * a[2] + V * b[2]
        
        # Aktuelle Grenzen holen
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim() 
        current_zlim = ax.get_zlim()
        
        # Erweiterte Grenzen berechnen
        new_xlim = [min(current_xlim[0], X.min() - 1), max(current_xlim[1], X.max() + 1)]
        new_ylim = [min(current_ylim[0], Y.min() - 1), max(current_ylim[1], Y.max() + 1)]
        new_zlim = [min(current_zlim[0], Z.min() - 1), max(current_zlim[1], Z.max() + 1)]
        
        # Neue Grenzen setzen
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.set_zlim(new_zlim)
        
        print("Extended axes for plane:")
        print(f"X: {new_xlim}")
        print(f"Y: {new_ylim}")
        print(f"Z: {new_zlim}")

def add_determinant_text(ax: Axes3D, determinant: float) -> None:
    """
    Add determinant information as formatted text to the plot.
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object.
    determinant : float
        The determinant value.
        
    Notes
    -----
    The text is positioned in the upper left corner of the plot and contains 
    both the determinant value and an interpretation of linear dependence.
    """
    det_text = f"Determinante: {determinant:.3f}"
    if abs(determinant) < 1e-10:
        status_text = "Linear abhängig"
    else:
        status_text = f"Linear unabhängig\nVolumen: {abs(determinant):.3f}"
    
    ax.text2D(0.02, 0.98, f"{det_text}\n{status_text}", 
              transform=ax.transAxes, fontsize=10,
              verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def plot_3d_vectors_with_names(**kwargs) -> None:
    """
    Plot 3D vectors using variable names as labels automatically.
    
    Parameters
    ----------
    **kwargs : dict
        Named vectors as keyword arguments. The parameter names will be used as labels.
        
    Examples
    --------
    >>> a_ = np.array([1, 4, -2])
    >>> b_ = np.array([-2, 2, 3])
    >>> c_ = np.array([-1, 6, 1])
    >>> plot_3d_vectors_with_names(a_=a_, b_=b_, c_=c_)
    """
    vectors = list(kwargs.values())
    labels = list(kwargs.keys())
    plot_3d_vectors(vectors, labels=labels)

def plot_3d_vectors(vectors: List[np.ndarray], labels: Optional[List[str]] = None, 
                   colors: Optional[List[str]] = None, show_plane: bool = True, 
                   show_determinant: bool = True) -> None:
    """
    Main function for plotting 3D vectors with complete analysis and visualization.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of 3D vectors as NumPy arrays.
    labels : Optional[List[str]], default=None
        Labels for the vectors. If None, default labels are used.
    colors : Optional[List[str]], default=None
        Colors for the vectors. If None, default colors are used.
    show_plane : bool, default=True
        Whether to draw a plane if vectors are coplanar.
    show_determinant : bool, default=True
        Whether to display determinant information.
        
    Examples
    --------
    >>> a = np.array([1, 4, -2])
    >>> b = np.array([-2, 2, 3])
    >>> c = np.array([-1, 6, 1])
    >>> plot_3d_vectors([a, b, c])
    
    >>> # With custom options
    >>> plot_3d_vectors([a, b, c], 
    ...                labels=['Force A', 'Force B', 'Force C'],
    ...                colors=['red', 'green', 'blue'],
    ...                show_plane=False)
                        
    Notes
    -----
    - For exactly 3 vectors, the determinant is automatically calculated
    - For ≥2 vectors, the normal vector (cross product of first two) is added
    - Axis limits are automatically optimized
    - For linear dependence, a plane is optionally displayed
    """
    # Calculate determinant (only for exactly 3 vectors)
    determinant = None
    if len(vectors) == 3:
        determinant = calculate_determinant(vectors)
        if show_determinant:
            print_determinant_info(determinant)
    
    # Calculate normal vector (for first two vectors)
    normal_vector = None
    if len(vectors) >= 2:
        normal_vector = np.cross(vectors[0], vectors[1])
        vectors_to_plot = vectors + [normal_vector]
        
        if labels is None:
            labels = ['a\'', 'b\'', 'c\'', 'Normalenvektor'][:len(vectors_to_plot)]
        else:
            labels = labels + ['Normalenvektor']
        
        if colors is None:
            colors = ['red', 'blue', 'green', 'purple'][:len(vectors_to_plot)]
        else:
            colors = colors + ['purple']
    else:
        vectors_to_plot = vectors
    
    # Plot erstellen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw vectors
    draw_vectors(ax, vectors, labels, colors)
    
    # Automatically adjust axes (before drawing plane!)
    set_automatic_axis_limits(ax, vectors)
    
    # Draw plane (if requested and possible)
    if show_plane and determinant is not None:
        draw_plane_if_coplanar(ax, vectors, determinant)
        # Achsengrenzen nach Ebenen-Zeichnung erweitern
        if abs(determinant) < 1e-10 and len(vectors) >= 2:
            set_axis_limits_for_plane(ax, vectors)
    
    # Achsenbeschriftung
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Add determinant text
    if show_determinant and determinant is not None:
        add_determinant_text(ax, determinant)
    
    plt.show()

# HAUPTPROGRAMM - Einfach zu verwenden!
if __name__ == "__main__":
    # Deine Vektoren hier definieren
    a_ = np.array([1, 4, -2])
    b_ = np.array([-2, 2, 3])
    c_ = np.array([-1, 6, 1])

    b1 = np.array([0, 0, 5])
    b2 = np.array([1, 2, 0])
    b3 = np.array([3, 0, 0])

    det = calculate_determinant([a_, b_, c_])
    print_determinant_info(det)
    
    
    # Mit automatischen Variablennamen als Labels!
    # print("=== Linear unabhängige Vektoren (keine Ebene) ===")
    # plot_3d_vectors_with_names(b1=b1, b2=b2, b3=b3)
    
    # print("\n=== Linearly dependent vectors (with plane) ===")  
    # plot_3d_vectors_with_names(a_=a_, b_=b_, c_=c_)
