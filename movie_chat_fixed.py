import pandas as pd
import google.generativeai as genai
import os
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from flask import Flask, render_template, request, jsonify, session
import uuid

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'movie_recommendation_secret_key_2024'

# Global variables
movies_df = None
recommender = None
conversation_memory = {}

class MovieRecommender:
    def __init__(self, csv_file_path):
        """Initialize the movie recommender with CSV data and Gemini client."""
        self.model = None
        
        # Initialize Gemini API
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("Gemini API initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini API: {e}")
                print("Continuing with basic functionality...")
        else:
            print("Warning: GEMINI_API_KEY not found. Using basic functionality only.")
        
        # Load movie data
        self.movies_df = self.load_movies(csv_file_path)
        print(f"Successfully loaded {len(self.movies_df)} movies")

    def load_movies(self, csv_file_path):
        """Load and validate movie data from CSV."""
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file_path, encoding='latin-1')
                print("Successfully loaded CSV with latin-1 encoding")
            except Exception as e:
                print(f"Error loading CSV: {e}")
                return pd.DataFrame()
        
        # Validate required columns
        required_columns = ['name', 'genre', 'released', 'popular']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        return df

    def is_analytical_question(self, query):
        """Check if the query is asking for analysis rather than recommendations using Gemini."""
        try:
            if self.model:
                prompt = f"""Is this question asking me to analyze/pick from movies I already showed, or search for completely new movies?

Query: "{query}"

Examples:
- "which one you recommend?" = ANALYSIS (pick from shown movies)
- "what's the best one?" = ANALYSIS (analyze shown movies)  
- "about this movie?" = ANALYSIS (analyze specific movie)
- "action movies" = SEARCH (find new movies)

Answer: ANALYSIS or SEARCH"""
                response = self.model.generate_content(prompt)
                print(f"DEBUG: Analysis check - Query: '{query}' -> Gemini response: '{response.text}'")
                return "ANALYSIS" in response.text.upper()
            else:
                # Basic fallback using general patterns
                query_lower = query.lower()
                question_patterns = ['which', 'what', 'how', 'are they', 'is it', 'tell me', 'pick', 'choose', 'recommend', 'suggest']
                analysis_context = ['one', 'best', 'better', 'rating', 'suitable', 'good', 'about']
                
                has_question = any(pattern in query_lower for pattern in question_patterns)
                has_context = any(context in query_lower for context in analysis_context)
                
                return has_question and has_context
        except Exception as e:
            query_lower = query.lower()
            return any(word in query_lower for word in ['which', 'pick', 'recommend', 'best', 'rating', 'suitable'])

    def extract_query_parameters(self, user_query, conversation_context=""):
        """Use Gemini to extract parameters from natural language query."""
        try:
            if self.model:
                prompt = f"""Extract movie search parameters from this query. Include conversation context if relevant.

Query: "{user_query}"
Context: {conversation_context}

Extract: genre, year_range, actor, director, country, age_group, description_keywords, intent

Respond in JSON format only."""
                
                response = self.model.generate_content(prompt)
                try:
                    params = json.loads(response.text)
                    print(f"DEBUG: Gemini extracted parameters: {params}")
                    return params
                except json.JSONDecodeError:
                    print("DEBUG: Gemini response was not valid JSON, using fallback")
                    return self.basic_parameter_extraction(user_query, conversation_context)
            else:
                return self.basic_parameter_extraction(user_query, conversation_context)
                
        except Exception as e:
            print(f"DEBUG: Error in parameter extraction: {e}")
            return self.basic_parameter_extraction(user_query, conversation_context)

    def basic_parameter_extraction(self, query, conversation_context=""):
        """Basic fallback parameter extraction without AI - uses general algorithmic approach."""
        params = {
            'genre': None, 'year_range': None, 'actor': None, 'director': None,
            'country': None, 'age_group': None, 'popular': None, 'runtime': None,
            'runtime_operator': None, 'description_keywords': [], 'intent': 'recommend'
        }
        
        # Use description keywords to let the filter handle the search algorithmically
        # This avoids hardcoded keyword lists and uses the actual movie data
        query_words = query.lower().split()
        
        # Extract meaningful words (not stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'what', 'are', 'is', 'movies', 'movie', 'good', 'best', 'find', 'show', 'me'}
        
        meaningful_words = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        if meaningful_words:
            params['description_keywords'] = meaningful_words
        
        return params

    def filter_movies(self, params):
        """Filter movies based on extracted parameters using algorithmic approach."""
        print(f"DEBUG: Starting with {len(self.movies_df)} movies")
        print(f"DEBUG: Parameters: {params}")
        
        filtered = self.movies_df.copy()
        
        # Description keywords filtering - algorithmic search through actual data
        if params.get('description_keywords'):
            keywords = params['description_keywords']
            print(f"DEBUG: Filtering by description keywords: {keywords}")
            
            # Create a combined text field for searching
            if 'name' in filtered.columns and 'genre' in filtered.columns:
                # Search in movie names and genres algorithmically
                search_text = filtered['name'].fillna('') + ' ' + filtered['genre'].fillna('')
                
                # Apply each keyword as a filter
                for keyword in keywords:
                    print(f"DEBUG: Searching for keyword '{keyword}'")
                    mask = search_text.str.contains(keyword, case=False, na=False)
                    keyword_matches = filtered[mask]
                    print(f"DEBUG: Found {len(keyword_matches)} movies with '{keyword}'")
                    
                    if not keyword_matches.empty:
                        filtered = keyword_matches
                        break  # Use first successful keyword match
                
                print(f"DEBUG: After description filtering: {len(filtered)} movies")
        
        # Sort by popularity if available
        if 'popular' in filtered.columns:
            filtered = filtered.sort_values('popular', ascending=False)
        
        return filtered

    def generate_analytical_response(self, filtered_movies, query, conversation_context=""):
        """Generate analytical response using Gemini."""
        if filtered_movies.empty:
            return "I don't have any movies to analyze based on your previous search."
        
        try:
            if self.model:
                movies_data = []
                for _, movie in filtered_movies.head(10).iterrows():
                    year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
                    genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
                    rating = movie['popular'] if pd.notna(movie['popular']) else 'Unknown'
                    movies_data.append({
                        'title': movie['name'],
                        'year': year,
                        'genre': genre,
                        'popularity_rating': rating
                    })
                
                context_info = ""
                if conversation_context:
                    context_info = f"\nConversation history:\n{conversation_context}\n"
                
                prompt = f"""You are a movie recommendation chatbot. The user is asking: "{query}"
{context_info}
Movies available for analysis:
{movies_data}

If the user asks about "this movie" or similar, refer to the specific movie you mentioned in the conversation history.

Give a SHORT, conversational response (2-3 sentences max). Be friendly but CONCISE."""

                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return self.generate_basic_analytical_response(filtered_movies, query)
                
        except Exception as e:
            return self.generate_basic_analytical_response(filtered_movies, query)

    def generate_basic_analytical_response(self, filtered_movies, query):
        """Generate basic analytical response without AI - simple fallback."""
        if len(filtered_movies) == 0:
            return "I couldn't find any movies matching your criteria."
        
        query_lower = query.lower()
        if 'which' in query_lower or 'recommend' in query_lower:
            top_movie = filtered_movies.iloc[0]
            return f"I'd recommend \"{top_movie['name']}.\" It has a good rating and fits your criteria!"
        elif 'best' in query_lower:
            return f"The highest-rated movie from your search is \"{filtered_movies.iloc[0]['name']}.\""
        else:
            return f"I found {len(filtered_movies)} movies that match your search."

    def generate_fallback_response(self, filtered_movies, params=None):
        """Generate a basic response without AI."""
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Try a different search!"
        
        movies_list = []
        for _, movie in filtered_movies.head(6).iterrows():
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
            movies_list.append(f"• {movie['name']} ({year}) - {genre}")
        
        return "Here are some movie recommendations:\n" + "\n".join(movies_list)

    def get_recommendation(self, user_query, conversation_context=""):
        """Main method to get movie recommendations based on user query."""
        print(f"DEBUG: Processing query: {user_query}")
        print(f"DEBUG: Conversation context: {conversation_context[:100]}...")
        
        # Check if this is an analytical question first
        if conversation_context and self.is_analytical_question(user_query):
            # For analytical questions, use the previous search results
            params = self.extract_query_parameters(conversation_context, "")
            filtered_movies = self.filter_movies(params)
            return self.generate_analytical_response(filtered_movies, user_query, conversation_context)

        # Regular search query
        params = self.extract_query_parameters(user_query, conversation_context)
        filtered_movies = self.filter_movies(params)
        
        print(f"DEBUG: Final results summary:")
        for i, (_, movie) in enumerate(filtered_movies.head(6).iterrows(), 1):
            rating = movie['popular'] if pd.notna(movie['popular']) else 'N/A'
            print(f"  {i}. {movie['name']} - Popularity: {rating}")
        
        # Generate response
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Try a different search!"
        
        # Use AI to generate response if available
        try:
            if self.model:
                movies_data = []
                for _, movie in filtered_movies.head(6).iterrows():
                    year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
                    genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
                    rating = movie['popular'] if pd.notna(movie['popular']) else 'Unknown'
                    movies_data.append(f"{movie['name']} ({year}) - {genre}")
                
                prompt = f"""List these movies in a friendly way. Keep it short and simple.

Movies: {movies_data}

Format as a bullet list. Be conversational but brief."""
                
                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return self.generate_fallback_response(filtered_movies, params)
                
        except Exception as e:
            return self.generate_fallback_response(filtered_movies, params)

def initialize_system():
    """Initialize the movie recommendation system."""
    global recommender, movies_df
    
    print("Initializing Movie Recommendation System...")
    
    # Check API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        print(f"API Key exists: True")
        print(f"API Key starts with: {api_key[:10]}...")
    else:
        print("API Key exists: False")
    
    # Initialize recommender
    csv_path = 'attached_assets/MergeAndCleaned_Movies.csv'
    recommender = MovieRecommender(csv_path)
    movies_df = recommender.movies_df
    
    print("System initialized successfully!")

def setup_routes():
    """Setup Flask routes."""
    
    @app.route('/')
    def index():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Movie Recommendation Chat</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .chat-container { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; background-color: #fafafa; border-radius: 5px; }
                .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
                .user-message { background-color: #007bff; color: white; text-align: right; margin-left: 20%; }
                .bot-message { background-color: #e9ecef; color: #333; margin-right: 20%; }
                .input-container { display: flex; gap: 10px; }
                .input-container input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
                .input-container button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
                .input-container button:hover { background-color: #0056b3; }
                .reset-btn { background-color: #dc3545 !important; margin-left: 5px; }
                .reset-btn:hover { background-color: #c82333 !important; }
                h1 { text-align: center; color: #333; margin-bottom: 30px; }
                .loading { color: #666; font-style: italic; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎬 Movie Recommendation Chat</h1>
                <div id="chat-container" class="chat-container"></div>
                <div class="input-container">
                    <input type="text" id="user-input" placeholder="Ask me about movies...">
                    <button onclick="sendMessage()">Send</button>
                    <button class="reset-btn" onclick="resetChat()">Reset</button>
                </div>
            </div>

            <script>
                function addMessage(message, isUser) {
                    const chatContainer = document.getElementById('chat-container');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
                    messageDiv.innerHTML = message.replace(/\\n/g, '<br>');
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                function sendMessage() {
                    const input = document.getElementById('user-input');
                    const message = input.value.trim();
                    
                    if (!message) return;
                    
                    addMessage(message, true);
                    input.value = '';
                    
                    addMessage('<span class="loading">Thinking...</span>', false);
                    
                    fetch('/recommend', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({query: message})
                    })
                    .then(response => response.json())
                    .then(data => {
                        const chatContainer = document.getElementById('chat-container');
                        chatContainer.removeChild(chatContainer.lastChild);
                        
                        if (data.error) {
                            addMessage('Error: ' + data.error, false);
                        } else {
                            addMessage(data.response, false);
                        }
                    })
                    .catch(error => {
                        const chatContainer = document.getElementById('chat-container');
                        chatContainer.removeChild(chatContainer.lastChild);
                        addMessage('Error: Failed to get response', false);
                    });
                }

                function resetChat() {
                    document.getElementById('chat-container').innerHTML = '';
                    fetch('/recommend', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: 'RESET_CONVERSATION'})
                    });
                }

                document.getElementById('user-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });

                addMessage('Hi! I can help you find movies. Try asking: "What movies are good for kids?" or "I want action movies from 2020"', false);
            </script>
        </body>
        </html>
        '''

    def get_user_id():
        """Get or create user session ID"""
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        return session['user_id']

    def save_conversation(user_id, user_query, response):
        """Save conversation to memory"""
        if user_id not in conversation_memory:
            conversation_memory[user_id] = []

        conversation_memory[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'response': response
        })

        # Keep only last 10 conversations to avoid memory overflow
        if len(conversation_memory[user_id]) > 10:
            conversation_memory[user_id] = conversation_memory[user_id][-10:]

    def get_conversation_context(user_id):
        """Get recent conversation context for user"""
        if user_id not in conversation_memory:
            return ""

        if not conversation_memory[user_id]:
            return ""

        # For better context quality, return only the last conversation
        context = "Previous conversation:\n"
        last_conv = conversation_memory[user_id][-1]
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

            user_id = get_user_id()

            # Handle reset command
            if query == 'RESET_CONVERSATION':
                if user_id in conversation_memory:
                    conversation_memory[user_id] = []
                return jsonify({'response': 'Conversation reset!'})

            # Get conversation context
            conversation_context = get_conversation_context(user_id)

            # Get recommendation
            if recommender:
                response = recommender.get_recommendation(query, conversation_context)
                
                # Save this conversation
                save_conversation(user_id, query, response)
                
                return jsonify({'response': response})
            else:
                return jsonify({'error': 'Movie recommendation system not initialized'}), 500

        except Exception as e:
            print(f"Error in recommend route: {e}")
            return jsonify({'error': 'Internal server error'}), 500

def start_server():
    """Start the Flask development server."""
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

def main():
    """Main function to run the application."""
    print("=" * 50)
    print("Movie Recommendation Chatbot")
    print("=" * 50)

    # Initialize the system
    initialize_system()

    # Setup Flask routes
    setup_routes()

    # Start the server
    start_server()

if __name__ == '__main__':
    main()