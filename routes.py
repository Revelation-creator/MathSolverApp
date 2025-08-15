from flask import render_template, request, jsonify, send_file
from app import app
from math_solver import MathSolver
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging

# Initialize the math solver
solver = MathSolver()

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve_expression():
    """Solve basic mathematical expressions"""
    try:
        data = request.get_json()
        expression = data.get('expression', '')
        
        if not expression.strip():
            return jsonify({'error': 'Please enter an expression'})
        
        result = solver.solve_basic_expression(expression)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in solve_expression: {str(e)}")
        return jsonify({'error': f'Error solving expression: {str(e)}'})

@app.route('/quadratic', methods=['POST'])
def solve_quadratic():
    """Solve quadratic equations"""
    try:
        data = request.get_json()
        a = float(data.get('a', 0))
        b = float(data.get('b', 0))
        c = float(data.get('c', 0))
        
        if a == 0:
            return jsonify({'error': 'Coefficient "a" cannot be zero for quadratic equations'})
        
        result = solver.solve_quadratic(a, b, c)
        return jsonify(result)
    
    except ValueError:
        return jsonify({'error': 'Please enter valid numerical coefficients'})
    except Exception as e:
        logging.error(f"Error in solve_quadratic: {str(e)}")
        return jsonify({'error': f'Error solving quadratic equation: {str(e)}'})

@app.route('/system', methods=['POST'])
def solve_system():
    """Solve system of linear equations"""
    try:
        data = request.get_json()
        equations = data.get('equations', [])
        
        if len(equations) < 2:
            return jsonify({'error': 'Please provide at least 2 equations'})
        
        result = solver.solve_linear_system(equations)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in solve_system: {str(e)}")
        return jsonify({'error': f'Error solving system: {str(e)}'})

@app.route('/simplify', methods=['POST'])
def simplify_expression():
    """Simplify algebraic expressions"""
    try:
        data = request.get_json()
        expression = data.get('expression', '')
        
        if not expression.strip():
            return jsonify({'error': 'Please enter an expression to simplify'})
        
        result = solver.simplify_expression(expression)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in simplify_expression: {str(e)}")
        return jsonify({'error': f'Error simplifying expression: {str(e)}'})

@app.route('/trigonometric', methods=['POST'])
def solve_trigonometric():
    """Solve trigonometric functions"""
    try:
        data = request.get_json()
        function = data.get('function', '')
        value = data.get('value', '')
        
        if not function or not value:
            return jsonify({'error': 'Please provide both function and value'})
        
        result = solver.solve_trigonometric(function, value)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in solve_trigonometric: {str(e)}")
        return jsonify({'error': f'Error solving trigonometric function: {str(e)}'})

@app.route('/calculus', methods=['POST'])
def solve_calculus():
    """Solve calculus operations (derivatives and integrals)"""
    try:
        data = request.get_json()
        expression = data.get('expression', '')
        operation = data.get('operation', 'derivative')
        variable = data.get('variable', 'x')
        
        if not expression.strip():
            return jsonify({'error': 'Please enter an expression'})
        
        result = solver.solve_calculus(expression, operation, variable)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in solve_calculus: {str(e)}")
        return jsonify({'error': f'Error solving calculus operation: {str(e)}'})

@app.route('/matrix', methods=['POST'])
def solve_matrix():
    """Solve matrix operations"""
    try:
        data = request.get_json()
        operation = data.get('operation', '')
        matrix1 = data.get('matrix1', [])
        matrix2 = data.get('matrix2', [])
        
        if not matrix1:
            return jsonify({'error': 'Please provide at least one matrix'})
        
        result = solver.solve_matrix_operation(operation, matrix1, matrix2)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in solve_matrix: {str(e)}")
        return jsonify({'error': f'Error solving matrix operation: {str(e)}'})

@app.route('/plot', methods=['POST'])
def plot_function():
    """Generate plot for mathematical functions"""
    try:
        data = request.get_json()
        expression = data.get('expression', '')
        x_min = float(data.get('x_min', -10))
        x_max = float(data.get('x_max', 10))
        
        if not expression.strip():
            return jsonify({'error': 'Please enter an expression to plot'})
        
        result = solver.plot_function(expression, x_min, x_max)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in plot_function: {str(e)}")
        return jsonify({'error': f'Error plotting function: {str(e)}'})

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error occurred'}), 500
