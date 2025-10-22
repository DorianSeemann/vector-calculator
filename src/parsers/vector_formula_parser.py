"""
Vector Formula Parser and Solver
Parses and evaluates vector expressions with step-by-step solutions.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional

class VectorFormulaParser:
    """Parser for vector mathematical expressions."""
    
    def __init__(self):
        self.vectors = {}
        self.solution_steps = []
        
    def set_vectors(self, vector_dict: Dict[str, np.ndarray]):
        """Set the available vectors for calculations."""
        self.vectors = vector_dict.copy()
        
    def parse_and_solve(self, formula: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Parse a vector formula and return result with solution steps.
        
        Supported operations:
        - Addition: A + B
        - Subtraction: A - B  
        - Cross product: A × B, A x B, cross(A, B)
        - Dot product: A · B, A . B, dot(A, B)
        - Scalar multiplication: 2*A, 3.5*B
        - Magnitude: |A|, mag(A), norm(A)
        - Normalization: unit(A), normalize(A)
        - Parentheses: (A + B) × C
        """
        self.solution_steps = []
        
        try:
            # Handle variable assignment (X = A + B)
            if '=' in formula:
                var_name = self.extract_variable_name(formula)
                formula = self.get_formula_without_assignment(formula)
                if var_name:
                    self.solution_steps.append(f"Assignment: {var_name} = {formula}")
                    
            # Clean and normalize the formula
            formula = self._normalize_formula(formula)
            self.solution_steps.append(f"Formula to evaluate: {formula}")
            
            # Tokenize and parse
            tokens = self._tokenize(formula)
            self.solution_steps.append(f"Tokenized: {' '.join(tokens)}")
            
            # Evaluate the expression
            result = self._evaluate_expression(tokens)
            
            return result, self.solution_steps
            
        except Exception as e:
            error_msg = f"Error parsing formula: {str(e)}"
            self.solution_steps.append(error_msg)
            return None, self.solution_steps
    
    def extract_variable_name(self, formula: str) -> Optional[str]:
        """
        Extract variable name from formula if it has format 'VAR = expression'.
        Returns the variable name or None if no assignment found.
        """
        formula = formula.strip()
        if '=' in formula:
            parts = formula.split('=', 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                # Validate that it's a valid variable name (letters, numbers, underscore)
                if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', var_name):
                    return var_name
        return None
    
    def get_formula_without_assignment(self, formula: str) -> str:
        """
        Get formula without assignment part (everything after =).
        If no = exists, returns original formula.
        """
        formula = formula.strip()
        if '=' in formula:
            parts = formula.split('=', 1)
            if len(parts) == 2:
                return parts[1].strip()
        return formula
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize the formula string."""
        # Remove extra spaces
        formula = ' '.join(formula.split())
        
        # Replace different symbols with standard ones
        replacements = {
            '×': 'cross',
            'x': 'cross',
            '·': 'dot',
            '.': 'dot',
            '||': 'mag',
        }
        
        for old, new in replacements.items():
            formula = formula.replace(old, new)
        
        return formula
    
    def _tokenize(self, formula: str) -> List[str]:
        """Tokenize the formula into components."""
        # Define token patterns
        patterns = [
            r'R_\{\d+\}',  # Math notation R_{1}, R_{2}, etc.
            r'[A-Z][a-z]*_\d+',  # Variables with underscore (A_1, B_2, Vector_3, etc.)
            r'\d+\.?\d*',  # Numbers (including decimals)
            r'[A-Z][a-z]*\d*',  # Vector names (A, B, C, result1, R1, etc.)
            r'cross|dot|mag|norm|unit|normalize|angle',  # Function names
            r'[+\-*/(),]',  # Operators, parentheses, and comma
            r'\|',  # Magnitude bars
        ]
        
        token_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        tokens = []
        
        for match in re.finditer(token_pattern, formula):
            token = match.group().strip()
            if token:
                tokens.append(token)
        
        return tokens
    
    def _evaluate_expression(self, tokens: List[str]) -> np.ndarray:
        """Evaluate the tokenized expression."""
        # Convert to postfix notation and evaluate
        postfix = self._to_postfix(tokens)
        self.solution_steps.append(f"Postfix notation: {' '.join(postfix)}")
        
        return self._evaluate_postfix(postfix)
    
    def _to_postfix(self, tokens: List[str]) -> List[str]:
        """Convert infix notation to postfix using Shunting Yard algorithm."""
        precedence = {
            '+': 1, '-': 1,
            '*': 2, '/': 2,
            'cross': 3, 'dot': 3,
            'mag': 4, 'norm': 4, 'unit': 4, 'normalize': 4, 'angle': 4,
        }
        
        output = []
        operators = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if self._is_number(token):
                output.append(token)
            elif self._is_vector(token):
                output.append(token)
            elif token in precedence:
                while (operators and operators[-1] != '(' and
                       operators[-1] in precedence and
                       precedence[operators[-1]] >= precedence[token]):
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                if operators:
                    operators.pop()  # Remove the '('
            elif token == '|':
                # Handle magnitude notation |A|
                if i + 2 < len(tokens) and tokens[i + 2] == '|':
                    vector_name = tokens[i + 1]
                    output.append(vector_name)
                    output.append('mag')
                    i += 2  # Skip the vector name and closing |
            
            i += 1
        
        while operators:
            output.append(operators.pop())
        
        return output
    
    def _evaluate_postfix(self, postfix: List[str]) -> np.ndarray:
        """Evaluate postfix expression."""
        stack = []
        
        for token in postfix:
            if self._is_number(token):
                stack.append(float(token))
                self.solution_steps.append(f"Push number: {token}")
                
            elif self._is_vector(token):
                if token in self.vectors:
                    vector = self.vectors[token]
                    stack.append(vector)
                    self.solution_steps.append(f"Push vector {token}: {vector}")
                else:
                    raise ValueError(f"Unknown vector: {token}")
                    
            elif token in ['+', '-', '*', '/', 'cross', 'dot', 'angle']:
                if len(stack) < 2:
                    raise ValueError(f"Not enough operands for {token}")
                    
                b = stack.pop()
                a = stack.pop()
                result = self._apply_binary_operation(a, b, token)
                stack.append(result)
                
            elif token in ['mag', 'norm', 'unit', 'normalize']:
                if len(stack) < 1:
                    raise ValueError(f"Not enough operands for {token}")
                    
                a = stack.pop()
                result = self._apply_unary_operation(a, token)
                stack.append(result)
        
        if len(stack) != 1:
            raise ValueError("Invalid expression")
        
        final_result = stack[0]
        if isinstance(final_result, np.ndarray):
            self.solution_steps.append(f"Final result: {final_result}")
        else:
            self.solution_steps.append(f"Final result (scalar): {final_result}")
            
        return final_result
    
    def _apply_binary_operation(self, a, b, op: str):
        """Apply binary operation between two operands."""
        if op == '+':
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                result = a + b
                self.solution_steps.append(f"Vector addition: {a} + {b} = {result}")
                return result
            else:
                raise ValueError("Addition requires two vectors")
                
        elif op == '-':
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                result = a - b
                self.solution_steps.append(f"Vector subtraction: {a} - {b} = {result}")
                return result
            else:
                raise ValueError("Subtraction requires two vectors")
                
        elif op == '*':
            if isinstance(a, (int, float)) and isinstance(b, np.ndarray):
                result = a * b
                self.solution_steps.append(f"Scalar multiplication: {a} * {b} = {result}")
                return result
            elif isinstance(a, np.ndarray) and isinstance(b, (int, float)):
                result = a * b
                self.solution_steps.append(f"Scalar multiplication: {a} * {b} = {result}")
                return result
            else:
                raise ValueError("Multiplication requires scalar and vector")
                
        elif op == '/':
            if isinstance(a, np.ndarray) and isinstance(b, (int, float)):
                if b == 0:
                    raise ValueError("Division by zero")
                result = a / b
                self.solution_steps.append(f"Scalar division: {a} / {b} = {result}")
                return result
            else:
                raise ValueError("Division requires vector and scalar")
                
        elif op == 'cross':
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if len(a) == 3 and len(b) == 3:
                    result = np.cross(a, b)
                    self.solution_steps.append(f"Cross product: {a} × {b} = {result}")
                    self.solution_steps.append(f"Magnitude of cross product: |{result}| = {np.linalg.norm(result):.3f}")
                    return result
                else:
                    raise ValueError("Cross product requires 3D vectors")
            else:
                raise ValueError("Cross product requires two vectors")
                
        elif op == 'dot':
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                result = np.dot(a, b)
                self.solution_steps.append(f"Dot product: {a} · {b} = {result}")
                # Add angle calculation
                mag_a = np.linalg.norm(a)
                mag_b = np.linalg.norm(b)
                if mag_a > 0 and mag_b > 0:
                    cos_angle = result / (mag_a * mag_b)
                    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
                    angle_deg = np.degrees(angle_rad)
                    self.solution_steps.append(f"Angle between vectors: {angle_deg:.1f}°")
                return result
            else:
                raise ValueError("Dot product requires two vectors")
                
        elif op == 'angle':
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                # Calculate angle between two vectors
                mag_a = np.linalg.norm(a)
                mag_b = np.linalg.norm(b)
                
                if mag_a == 0 or mag_b == 0:
                    raise ValueError("Cannot calculate angle with zero vector")
                
                dot_product = np.dot(a, b)
                cos_angle = dot_product / (mag_a * mag_b)
                
                # Clamp to avoid numerical errors
                cos_angle = np.clip(cos_angle, -1, 1)
                
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                self.solution_steps.append("Angle calculation:")
                self.solution_steps.append(f"  |{a}| = {mag_a:.3f}")
                self.solution_steps.append(f"  |{b}| = {mag_b:.3f}")
                self.solution_steps.append(f"  {a} · {b} = {dot_product:.3f}")
                self.solution_steps.append(f"  cos(θ) = {dot_product:.3f} / ({mag_a:.3f} * {mag_b:.3f}) = {cos_angle:.3f}")
                self.solution_steps.append(f"  θ = arccos({cos_angle:.3f}) = {angle_deg:.1f}°")
                
                return angle_deg
            else:
                raise ValueError("Angle calculation requires two vectors")
        
        else:
            raise ValueError(f"Unknown binary operator: {op}")
    
    def _apply_unary_operation(self, a, op: str):
        """Apply unary operation to an operand."""
        if op in ['mag', 'norm']:
            if isinstance(a, np.ndarray):
                result = np.linalg.norm(a)
                self.solution_steps.append(f"Magnitude: |{a}| = {result}")
                return result
            else:
                raise ValueError("Magnitude requires a vector")
                
        elif op in ['unit', 'normalize']:
            if isinstance(a, np.ndarray):
                magnitude = np.linalg.norm(a)
                if magnitude == 0:
                    raise ValueError("Cannot normalize zero vector")
                result = a / magnitude
                self.solution_steps.append(f"Normalize: {a} / {magnitude} = {result}")
                return result
            else:
                raise ValueError("Normalization requires a vector")
        
        else:
            raise ValueError(f"Unknown unary operator: {op}")
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def _is_vector(self, token: str) -> bool:
        """Check if token is a vector name."""
        # Check for math notation R_{n}
        if token.startswith('R_{') and token.endswith('}'):
            return True
        # Check for underscore variables (A_1, B_2, Vector_3, etc.)
        if '_' in token:
            parts = token.split('_')
            if len(parts) == 2 and parts[0].isalpha() and parts[1].isdigit():
                return True
        # Check for result patterns (result1, R1, r1, etc.)
        if token.lower().startswith('result') or token.lower().startswith('r'):
            return True
        # Check for standard vector names (A, B, C, etc.)
        return token.isalpha() and token.isupper()

    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations with examples."""
        operations = [
            "Vector Addition: A + B, A_1 + B_2",
            "Vector Subtraction: A - B",
            "Scalar Multiplication: 2*A, 3.5*B_1",
            "Cross Product: A × B, A_1 cross B_2",
            "Dot Product: A · B, A_1 dot B_2",
            "Angle between vectors: A angle B",
            "Magnitude: |A|, |A_1|, mag(A)",
            "Normalization: unit(A), normalize(A_1)",
            "Complex expressions: (A_1 + B) × C_2",
            "Scalar division: A / 2",
        ]
        return operations

    def get_example_formulas(self) -> List[str]:
        """Get example formulas for testing."""
        examples = [
            "A + B",
            "A_1 + B_2", 
            "A × B",
            "A_1 · B_2",
            "A angle B",
            "|A_1|",
            "2*A_1",
            "A / 2",
            "unit(A_1)",
            "R_{1} + B_2",
            "(A_1 + B) × C_2",
            "A_1 · (B × C_2)",
            "2*A_1 + 3*B_2",
            "|R_{2} × B_1|",
            "unit(A_1) · unit(B_2)",
        ]
        return examples