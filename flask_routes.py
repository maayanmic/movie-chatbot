"""Flask routes and web interface for movie recommendation system."""

from flask import render_template, request, jsonify, session
import uuid
from datetime import datetime

def setup_routes(app, recommender, conversation_memory):
    """Setup Flask routes."""
    
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/get-user-id', methods=['GET'])
    def get_user_id():
        """Get or create user session ID"""
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        return jsonify({'user_id': session['user_id']})

    def save_conversation(user_id, user_query, response):
        """Save conversation to memory"""
        if user_id not in conversation_memory:
            conversation_memory[user_id] = []
        
        conversation_memory[user_id].append({
            'user_query': user_query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 3 conversations to avoid context overload
        if len(conversation_memory[user_id]) > 3:
            conversation_memory[user_id] = conversation_memory[user_id][-3:]

    def get_conversation_context(user_id):
        """Get recent conversation context for user"""
        if user_id not in conversation_memory:
            return ""

        if not conversation_memory[user_id]:
            return ""

        # For better context quality, return only the last conversation
        # This prevents confusion between different topics/genres
        context = "Previous conversation:\n"
        last_conv = conversation_memory[user_id][-1]  # Only last conversation
        context += f"User: {last_conv['user_query']}\n"
        context += f"Assistant: {last_conv['response'][:150]}...\n\n"

        return context

    @app.route('/recommend', methods=['POST'])
    def recommend():
        try:
            data = request.get_json()
            query = data.get('query', '')

            if not query:
                return jsonify({'error': 'No query provided'}), 400

            user_id = get_user_id().get_json()['user_id']

            # Handle reset conversation request
            if query == '__RESET_CONVERSATION__':
                if user_id in conversation_memory:
                    conversation_memory[user_id] = []
                return jsonify({'recommendation': 'Conversation reset successfully'})

            # Get conversation context for better understanding
            context = get_conversation_context(user_id)

            # Get movie recommendation with context
            recommendation = recommender.get_recommendation(query, context)

            # Save conversation to memory
            save_conversation(user_id, query, recommendation)

            return jsonify({'recommendation': recommendation})

        except Exception as e:
            print(f"Error in recommendation: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500