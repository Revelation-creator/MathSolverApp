import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
import io
from sympy import symbols, solve, simplify, diff, integrate, sin, cos, tan, asin, acos, atan
from sympy import Matrix as SpMatrix
from sympy.parsing.sympy_parser import parse_expr
import math
import logging

class MathSolver:
    def __init__(self):
        """Initialize the math solver with common symbols"""
        self.x, self.y, self.z = symbols('x y z')
        self.common_symbols = [self.x, self.y, self.z]
    
    def solve_basic_expression(self, expression):
        """Solve basic arithmetic expressions and equations"""
        try:
            # Parse the expression
            expr = parse_expr(expression)
            
            # If it's an equation (contains '='), solve it
            if '=' in expression:
                left, right = expression.split('=')
                left_expr = parse_expr(left.strip())
                right_expr = parse_expr(right.strip())
                equation = left_expr - right_expr
                solutions = solve(equation, self.x)
                
                return {
                    'result': str(solutions) if solutions else 'No solution found',
                    'steps': [
                        f"Original equation: {expression}",
                        f"Rearranged: {equation} = 0",
                        f"Solutions: {solutions}" if solutions else "No solutions found"
                    ],
                    'type': 'equation'
                }
            else:
                # Evaluate the expression
                try:
                    # Try to get numerical value
                    numerical_result = float(expr.evalf())
                    simplified = simplify(expr)
                    
                    return {
                        'result': str(numerical_result) if simplified == expr else f"{simplified} = {numerical_result}",
                        'steps': [
                            f"Original expression: {expression}",
                            f"Simplified: {simplified}",
                            f"Numerical value: {numerical_result}"
                        ],
                        'type': 'expression'
                    }
                except:
                    # If can't evaluate numerically, just simplify
                    simplified = simplify(expr)
                    return {
                        'result': str(simplified),
                        'steps': [
                            f"Original expression: {expression}",
                            f"Simplified: {simplified}"
                        ],
                        'type': 'expression'
                    }
        
        except Exception as e:
            raise Exception(f"Invalid expression: {str(e)}")
    
    def solve_quadratic(self, a, b, c):
        """Solve quadratic equations ax² + bx + c = 0"""
        try:
            # Create the quadratic equation
            expr = a * self.x**2 + b * self.x + c
            solutions = solve(expr, self.x)
            
            # Calculate discriminant for step-by-step solution
            discriminant = b**2 - 4*a*c
            
            steps = [
                f"Quadratic equation: {a}x² + {b}x + {c} = 0",
                f"Using quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)",
                f"a = {a}, b = {b}, c = {c}",
                f"Discriminant = b² - 4ac = {b}² - 4({a})({c}) = {discriminant}"
            ]
            
            if discriminant > 0:
                steps.append("Since discriminant > 0, there are two real solutions")
                steps.append(f"x₁ = ({-b} + √{discriminant}) / {2*a} = {solutions[0]}")
                steps.append(f"x₂ = ({-b} - √{discriminant}) / {2*a} = {solutions[1]}")
            elif discriminant == 0:
                steps.append("Since discriminant = 0, there is one repeated solution")
                steps.append(f"x = {-b} / {2*a} = {solutions[0]}")
            else:
                steps.append("Since discriminant < 0, there are two complex solutions")
                steps.append(f"Solutions: {solutions}")
            
            return {
                'result': str(solutions),
                'steps': steps,
                'discriminant': float(discriminant),
                'type': 'quadratic'
            }
        
        except Exception as e:
            raise Exception(f"Error solving quadratic equation: {str(e)}")
    
    def solve_linear_system(self, equations):
        """Solve system of linear equations"""
        try:
            # Parse equations and create matrix form
            parsed_equations = []
            variables = set()
            
            for eq in equations:
                if '=' not in eq:
                    raise Exception("Each equation must contain '=' sign")
                
                left, right = eq.split('=')
                expr = parse_expr(left.strip()) - parse_expr(right.strip())
                parsed_equations.append(expr)
                variables.update(expr.free_symbols)
            
            variables = sorted(list(variables), key=str)
            solutions = solve(parsed_equations, variables)
            
            steps = [
                "System of equations:",
                *[f"  {eq}" for eq in equations],
                "Solving using substitution/elimination method:",
                f"Variables found: {[str(var) for var in variables]}"
            ]
            
            if solutions:
                steps.append("Solutions:")
                for var, val in solutions.items():
                    steps.append(f"  {var} = {val}")
            else:
                steps.append("No unique solution found (system may be inconsistent or have infinite solutions)")
            
            return {
                'result': str(solutions) if solutions else 'No unique solution',
                'steps': steps,
                'variables': [str(var) for var in variables],
                'type': 'system'
            }
        
        except Exception as e:
            raise Exception(f"Error solving system: {str(e)}")
    
    def simplify_expression(self, expression):
        """Simplify algebraic expressions"""
        try:
            expr = parse_expr(expression)
            simplified = simplify(expr)
            expanded = sp.expand(expr)
            factored = sp.factor(expr)
            
            steps = [
                f"Original expression: {expression}",
                f"Simplified: {simplified}",
                f"Expanded form: {expanded}",
                f"Factored form: {factored}"
            ]
            
            return {
                'result': str(simplified),
                'simplified': str(simplified),
                'expanded': str(expanded),
                'factored': str(factored),
                'steps': steps,
                'type': 'simplification'
            }
        
        except Exception as e:
            raise Exception(f"Error simplifying expression: {str(e)}")
    
    def solve_trigonometric(self, function, value):
        """Solve trigonometric functions"""
        try:
            # Parse the value
            val = parse_expr(value)
            
            # Map function names to sympy functions
            trig_functions = {
                'sin': sin, 'cos': cos, 'tan': tan,
                'asin': asin, 'acos': acos, 'atan': atan
            }
            
            if function not in trig_functions:
                raise Exception(f"Unsupported function: {function}")
            
            func = trig_functions[function]
            
            # If it's an inverse function, just evaluate
            if function.startswith('a'):
                result = func(val)
                steps = [
                    f"Calculating {function}({value})",
                    f"Result: {result}",
                    f"In degrees: {float(result * 180 / sp.pi):.2f}°" if function in ['asin', 'acos', 'atan'] else ""
                ]
            else:
                # For forward functions, evaluate and also solve equations if needed
                result = func(val)
                numerical_result = float(result.evalf())
                steps = [
                    f"Calculating {function}({value})",
                    f"Result: {result}",
                    f"Numerical value: {numerical_result:.6f}"
                ]
            
            return {
                'result': str(result),
                'numerical': float(result.evalf()),
                'steps': [step for step in steps if step],  # Filter empty strings
                'type': 'trigonometric'
            }
        
        except Exception as e:
            raise Exception(f"Error solving trigonometric function: {str(e)}")
    
    def solve_calculus(self, expression, operation, variable='x'):
        """Solve calculus operations (derivatives and integrals)"""
        try:
            expr = parse_expr(expression)
            var = symbols(variable)
            
            if operation == 'derivative':
                result = diff(expr, var)
                steps = [
                    f"Finding derivative of {expression} with respect to {variable}",
                    f"d/d{variable}({expression}) = {result}"
                ]
                
                # Add step-by-step differentiation rules if possible
                if hasattr(expr, 'func'):
                    steps.append(f"Using differentiation rules for {expr.func}")
            
            elif operation == 'integral':
                result = integrate(expr, var)
                steps = [
                    f"Finding indefinite integral of {expression} with respect to {variable}",
                    f"∫{expression} d{variable} = {result} + C"
                ]
                
                # Try definite integral with common limits
                try:
                    definite_result = integrate(expr, (var, 0, 1))
                    steps.append(f"Definite integral from 0 to 1: {definite_result}")
                except:
                    pass
            
            else:
                raise Exception(f"Unsupported calculus operation: {operation}")
            
            return {
                'result': str(result),
                'steps': steps,
                'operation': operation,
                'variable': variable,
                'type': 'calculus'
            }
        
        except Exception as e:
            raise Exception(f"Error solving calculus operation: {str(e)}")
    
    def solve_matrix_operation(self, operation, matrix1, matrix2=None):
        """Solve matrix operations"""
        try:
            # Convert to SymPy matrices
            mat1 = SpMatrix(matrix1)
            
            steps = [f"Matrix A = {mat1}"]
            
            if operation == 'determinant':
                if not mat1.is_square:
                    raise Exception("Matrix must be square to calculate determinant")
                result = mat1.det()
                steps.append(f"det(A) = {result}")
                
            elif operation == 'inverse':
                if not mat1.is_square:
                    raise Exception("Matrix must be square to calculate inverse")
                if mat1.det() == 0:
                    raise Exception("Matrix is singular (determinant = 0), inverse does not exist")
                result = mat1.inv()
                steps.append(f"A⁻¹ = {result}")
                
            elif operation == 'transpose':
                result = mat1.T
                steps.append(f"Aᵀ = {result}")
                
            elif operation in ['addition', 'subtraction', 'multiplication']:
                if not matrix2:
                    raise Exception(f"Second matrix required for {operation}")
                
                mat2 = SpMatrix(matrix2)
                steps.append(f"Matrix B = {mat2}")
                
                if operation == 'addition':
                    if mat1.shape != mat2.shape:
                        raise Exception("Matrices must have same dimensions for addition")
                    result = mat1 + mat2
                    steps.append(f"A + B = {result}")
                    
                elif operation == 'subtraction':
                    if mat1.shape != mat2.shape:
                        raise Exception("Matrices must have same dimensions for subtraction")
                    result = mat1 - mat2
                    steps.append(f"A - B = {result}")
                    
                elif operation == 'multiplication':
                    if mat1.cols != mat2.rows:
                        raise Exception("Number of columns in first matrix must equal number of rows in second matrix")
                    result = mat1 * mat2
                    steps.append(f"A × B = {result}")
            
            else:
                raise Exception(f"Unsupported matrix operation: {operation}")
            
            return {
                'result': str(result),
                'steps': steps,
                'operation': operation,
                'matrix_shape': mat1.shape,
                'type': 'matrix'
            }
        
        except Exception as e:
            raise Exception(f"Error solving matrix operation: {str(e)}")
    
    def plot_function(self, expression, x_min=-10, x_max=10):
        """Generate plot for mathematical functions"""
        try:
            expr = parse_expr(expression)
            
            # Create x values
            x_vals = np.linspace(x_min, x_max, 1000)
            
            # Convert sympy expression to numpy function
            func = sp.lambdify(self.x, expr, 'numpy')
            
            # Calculate y values
            y_vals = func(x_vals)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'y = {expression}')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.title(f'Graph of y = {expression}', fontsize=14, fontweight='bold')
            plt.legend()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            # Find critical points (roots, maxima, minima)
            try:
                roots = solve(expr, self.x)
                derivative = diff(expr, self.x)
                critical_points = solve(derivative, self.x)
                
                analysis = {
                    'roots': [str(root) for root in roots if root.is_real],
                    'critical_points': [str(cp) for cp in critical_points if cp.is_real]
                }
            except:
                analysis = {'roots': [], 'critical_points': []}
            
            return {
                'result': f"Graph generated for y = {expression}",
                'image': img_str,
                'analysis': analysis,
                'x_range': [x_min, x_max],
                'steps': [
                    f"Function: y = {expression}",
                    f"Domain: [{x_min}, {x_max}]",
                    f"Roots: {analysis.get('roots', [])}",
                    f"Critical points: {analysis.get('critical_points', [])}"
                ],
                'type': 'plot'
            }
        
        except Exception as e:
            raise Exception(f"Error plotting function: {str(e)}")
