from flask import Flask, request, jsonify, send_file, render_template
import os
from datetime import datetime
import json

# Import your modules
from predict import LCAPredictor, make_prediction
from autofill import autofill_lca_data  # wherever you defined this

from lca_report_generator import generate_ai_lca_report

app = Flask(__name__)

# Initialize the predictor once when app starts
predictor = LCAPredictor(model_dir='ml_Alloyance/model/')

# Create directories for reports
os.makedirs('reports', exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')  # Your frontend

@app.route('/predict', methods=['POST'])
def predict_circularity():
    """
    Endpoint to make LCA predictions based on user input
    """
    try:
        # Get JSON data from request
        user_input = request.json
        
        if not user_input:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'Raw Material Type',
            'Process Stage',
            'Technology'
        ]
        
        missing_fields = [field for field in required_fields if field not in user_input]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # 1. Get raw input from user
        raw_input = user_input  # or request.get_json(force=True)

        # 2. Autofill and preprocess data
        df_imputed = autofill_lca_data(raw_input)

        # 3. # Call make_prediction from predict.py
        predictions = make_prediction(predictor, df_imputed)

        # Calculate overall circularity score
        circularity_score = (
            predictions['recycled_content'] + 
            predictions['reuse_potential'] + 
            predictions['recovery_rate']
        ) / 3
        
        # Return predictions
        response = {
            'predictions': predictions,
            'circularity_score': circularity_score,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'predict_circularity : error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """
    Endpoint to generate PDF report with LCA results
    """
    try:
        # Get input data
        data = request.json
        user_input = data.get('input_data', {})
        
        if not user_input:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Make predictions
        predictions = predictor.predict(user_input)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"lca_report_{timestamp}.pdf"
        report_path = os.path.join('static/reports', report_filename)
        
        # Generate the AI-powered report
        generate_ai_lca_report(
            model_predictions=predictions,
            input_parameters=user_input,
            output_file=report_path
        )
        
        # Return success response with download link
        response = {
            'status': 'success',
            'predictions': predictions,
            'report_url': f'/download_report/{report_filename}',
            'report_filename': report_filename
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_ai_report', methods=['POST'])
def generate_ai_report():
    """
    Endpoint to generate AI-powered PDF report with advanced analytics
    """
    try:
        # Get input data
        data = request.json
        user_input = data.get('input_data', {})
        
        if not user_input:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Make predictions
        predictions = predictor.predict(user_input)
        
        # Generate unique filename for AI report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"ai_lca_report_{timestamp}.pdf"
        report_path = os.path.join('static/reports', report_filename)
        
        # Generate the AI-powered report with advanced analytics
        generate_ai_lca_report(
            model_predictions=predictions,
            input_parameters=user_input,
            output_file=report_path
        )
        
        # Calculate overall circularity score for response
        circularity_score = (
            predictions['recycled_content'] + 
            predictions['reuse_potential'] + 
            predictions['recovery_rate']
        ) / 3
        
        # Return success response with download link
        response = {
            'status': 'success',
            'report_type': 'AI-Powered Advanced Analytics',
            'predictions': predictions,
            'circularity_score': round(circularity_score, 2),
            'report_url': f'/download_report/{report_filename}',
            'report_filename': report_filename,
            'features': [
                'Multi-agent AI analytics',
                'Dynamic benchmarking',
                'Professional terminology',
                'Strategic recommendations',
                'Trend projections'
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report/<filename>')
def download_report(filename):
    """
    Endpoint to download generated PDF reports
    """
    try:
        file_path = os.path.join('static/reports', filename)
        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name=filename,
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Report file not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint for batch predictions (multiple inputs at once)
    """
    try:
        # Get list of inputs
        input_list = request.json.get('inputs', [])
        
        if not input_list:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Make batch predictions
        results = predictor.batch_predict(input_list)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """
    Endpoint to get information about loaded models
    """
    try:
        info = {
            'models_loaded': list(predictor.models.keys()),
            'categorical_columns': predictor.categorical_cols,
            'model_directory': predictor.model_dir,
            'status': 'Models loaded successfully'
        }
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report_options')
def report_options():
    """
    Endpoint to get information about available report types
    """
    try:
        options = {
            'available_reports': {
                'standard_report': {
                    'endpoint': '/generate_report',
                    'description': 'Standard LCA report with comprehensive analysis',
                    'features': [
                        'Input parameter analysis',
                        'LCA results with visualizations',
                        'Environmental impact assessment',
                        'Strategic recommendations'
                    ]
                },
                'ai_powered_report': {
                    'endpoint': '/generate_ai_report',
                    'description': 'AI-powered report with advanced multi-agent analytics',
                    'features': [
                        'Multi-agent AI analytics',
                        'Dynamic industry benchmarking',
                        'Professional terminology optimization',
                        'Intelligent narrative generation',
                        'Advanced performance visualizations',
                        'Trend analysis and projections',
                        'Strategic roadmap generation'
                    ]
                }
            },
            'recommendation': 'Use /generate_ai_report for the most advanced analytics and insights'
        }
        
        return jsonify(options)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(predictor.models) == 3
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting LCA Prediction Flask App...")
    print(f"Models loaded: {list(predictor.models.keys())}")
    app.run(debug=True, host='0.0.0.0', port=5000)