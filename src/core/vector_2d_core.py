"""
2D Vector drawing functions for the vector calculator.
Enhanced 2D vector visualization with arrows and styling.
"""

import numpy as np
from typing import List, Optional

def draw_vectors_2d(ax, vectors: List[np.ndarray], labels: Optional[List[str]] = None, 
                   colors: Optional[List[str]] = None) -> None:
    """
    Draw vectors as arrows from the origin in a 2D coordinate system.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 2D axes object for drawing.
    vectors : List[np.ndarray]
        List of 2D vectors as NumPy arrays.
    labels : Optional[List[str]], default=None
        Labels for the vectors. If None, default labels are used.
    colors : Optional[List[str]], default=None
        Colors for the vectors. If None, default colors are used.
    """
    origin = np.zeros(2)
    default_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    default_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    if colors is None:
        colors = default_colors[:len(vectors)]
    if labels is None:
        labels = default_labels[:len(vectors)]
    
    for i, (vector, color, label) in enumerate(zip(vectors, colors, labels)):
        # Convert to 2D if necessary
        if len(vector) > 2:
            vector = vector[:2]
        elif len(vector) < 2:
            vector = np.append(vector, np.zeros(2 - len(vector)))
        
        norm = np.linalg.norm(vector)
        
        if norm > 0:
            # Dynamic arrow head size for 2D
            # Arrow head scaling - smaller for longer vectors
            pass  # Head size is now controlled by headwidth/headlength parameters
            
            # Draw vector arrow
            ax.quiver(*origin, *vector, color=color, label=label,
                     angles='xy', scale_units='xy', scale=1,
                     width=0.005, headwidth=3, headlength=4,
                     alpha=0.9)
            
            # Add vector endpoint label
            ax.annotate(label, (vector[0], vector[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, color=color, weight='bold')

def set_automatic_2d_axis_limits(ax, vectors: List[np.ndarray], buffer_factor: float = 0.3) -> None:
    """
    Set appropriate axis limits for 2D plot based on vector positions.
    Enhanced with smart scaling and origin inclusion.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 2D axes object.
    vectors : List[np.ndarray]
        List of vectors to determine limits from.
    buffer_factor : float, default=0.3
        Buffer factor for axis limits (30% extra space).
    """
    if not vectors:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        return
    
    # Convert all vectors to 2D and include origin
    vectors_2d = [np.array([0, 0])]  # Always include origin
    for v in vectors:
        if len(v) > 2:
            vectors_2d.append(v[:2])
        elif len(v) < 2:
            vectors_2d.append(np.append(v, np.zeros(2 - len(v))))
        else:
            vectors_2d.append(v)
    
    # Find min/max coordinates including origin
    all_coords = np.array(vectors_2d)
    
    if all_coords.size > 0:
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        # Calculate ranges with minimum size for small vectors
        x_range = max(abs(x_max - x_min), 2.0)  # Minimum range of 2
        y_range = max(abs(y_max - y_min), 2.0)
        
        # Smart buffer calculation
        x_buffer = x_range * buffer_factor
        y_buffer = y_range * buffer_factor
        
        # Ensure symmetric limits around origin if vectors are small
        if x_range <= 2.0 and y_range <= 2.0:
            max_extent = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
            limit = max(max_extent * (1 + buffer_factor), 2.0)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
        else:
            # Normal asymmetric limits for larger vectors
            ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
            ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
    else:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

def analyze_vectors_2d(vectors: List[np.ndarray]) -> dict:
    """
    Analyze 2D vector relationships.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of 2D vectors.
        
    Returns
    -------
    dict
        Analysis results including angles, dependencies, etc.
    """
    if len(vectors) < 2:
        return {"analysis": "Need at least 2 vectors for analysis"}
    
    # Convert to 2D
    vectors_2d = []
    for v in vectors:
        if len(v) > 2:
            vectors_2d.append(v[:2])
        elif len(v) < 2:
            vectors_2d.append(np.append(v, np.zeros(2 - len(v))))
        else:
            vectors_2d.append(v)
    
    results = {}
    
    # Check linear independence for 2 vectors
    if len(vectors_2d) >= 2:
        v1, v2 = vectors_2d[0], vectors_2d[1]
        
        # Cross product in 2D (determinant)
        det = v1[0] * v2[1] - v1[1] * v2[0]
        
        if abs(det) < 1e-10:
            results["dependency"] = "Vectors are linearly dependent (parallel)"
        else:
            results["dependency"] = "Vectors are linearly independent"
            results["area"] = abs(det)
        
        # Angle between vectors
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms > 1e-10:
            cos_angle = dot_product / norms
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            results["angle"] = {"radians": angle_rad, "degrees": angle_deg}
    
    return results