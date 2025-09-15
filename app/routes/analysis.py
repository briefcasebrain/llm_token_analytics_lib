"""
Analysis endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import logging

from llm_token_analytics import TokenAnalyzer

analysis_bp = Blueprint('analysis', __name__)
logger = logging.getLogger(__name__)


@analysis_bp.route('/analyze', methods=['POST'])
def analyze_distributions():
    """
    Analyze token distributions.

    Request body:
    {
        "collection_id": "collection_20250101_120000" or
        "data": {
            "input_tokens": [...],
            "output_tokens": [...]
        }
    }
    """
    try:
        data = request.json

        # Load data
        if 'collection_id' in data:
            collection = current_app.storage.get_result(data['collection_id'])
            if not collection:
                return jsonify({
                    'error': 'Collection not found'
                }), 404

            df = current_app.storage.get_collection_data(data['collection_id'])
        elif 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            return jsonify({
                'error': 'No data provided'
            }), 400

        # Analyze
        analyzer = TokenAnalyzer(df)

        # Distribution analysis
        dist_results = analyzer.analyze_distributions()

        # Correlation analysis
        corr_results = analyzer.analyze_correlations()

        # Prepare response
        response = {
            'success': True,
            'distributions': {},
            'correlations': {}
        }

        # Format distribution results
        for token_type, results in dist_results.items():
            response['distributions'][token_type] = {
                'best_fit': results['distribution']['type'],
                'parameters': results['distribution']['params'],
                'validation': {
                    'ks_pvalue': results['validation']['ks_pvalue'],
                    'is_valid': results['validation']['is_valid']
                },
                'statistics': results['statistics']
            }

        # Format correlation results
        response['correlations'] = {
            'pearson': corr_results['correlations']['linear']['pearson']['value'],
            'spearman': corr_results['correlations']['linear']['spearman']['value'],
            'kendall': corr_results['correlations']['linear']['kendall']['value'],
            'tail_correlation': corr_results['correlations']['tail_correlation']
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500
