from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import json
import os
import google.generativeai as genai
import re
import uuid
from datetime import datetime

print(f"API Key exists: {bool(os.environ.get('GEMINI_API_KEY'))}")
print(f"API Key starts with: {os.environ.get('GEMINI_API_KEY', 'NOT_FOUND')[:10]}...")

# Global variables
app = Flask(__name__, static_folder='.', template_folder='.')
app.secret_key = 'movie_recommender_secret_key_2024'
conversation_memory = {}
recommender = None

class MovieRecommender:
    def __init__(self, csv_file_path):
        """Initialize the movie recommender with CSV data and Gemini client."""
        try:
            # Initialize Gemini
            api_key = os.environ.get('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("Gemini API initialized successfully")
            else:
                print("Warning: GEMINI_API_KEY not found. Using basic mode.")
                self.model = None

            # Load movie data
            self.movies = self.load_movies(csv_file_path)
            print(f"Successfully loaded {len(self.movies)} movies")

        except Exception as e:
            print(f"Error initializing MovieRecommender: {str(e)}")
            self.model = None
            self.movies = pd.DataFrame()

    def load_movies(self, csv_file_path):
        """Load and validate movie data from CSV."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    movies = pd.read_csv(csv_file_path, encoding=encoding)
                    print(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read CSV file with any encoding")

            # Validate required columns
            required_columns = ['name', 'genre', 'released', 'popular']
            missing_columns = [col for col in required_columns if col not in movies.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Clean and validate data
            movies = movies.dropna(subset=['name'])
            movies['released'] = pd.to_numeric(movies['released'], errors='coerce')
            movies['popular'] = pd.to_numeric(movies['popular'], errors='coerce')

            return movies

        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise Exception(f"Error loading CSV: {str(e)}")

    def fuzzy_match(self, word, target_words, threshold=0.7):
        """Check if word matches any target word with fuzzy matching."""
        import difflib

        word_lower = word.lower().strip()

        for target in target_words:
            target_lower = target.lower()

            # Exact match
            if word_lower == target_lower:
                return True

            # Check if word contains target or target contains word
            if word_lower in target_lower or target_lower in word_lower:
                return True

            # Fuzzy matching using sequence similarity
            similarity = difflib.SequenceMatcher(None, word_lower, target_lower).ratio()
            if similarity >= threshold:
                return True

        return False

    def extract_query_parameters(self, user_query, conversation_context=""):
        """Use Gemini to extract parameters from natural language query."""

        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries.

IMPORTANT: Handle English text and typos. Be extremely flexible with genre recognition and spelling variations.

GENRE VARIATIONS TO RECOGNIZE (including common typos):
- Romance/Romantic: "romance", "romantic", "rommantic", "rommntic", "rommance", "romence", "romanc"
- Action: "action", "actoin", "akshen"
- Comedy: "comedy", "comdy", "komedia", "funny"
- Drama: "drama", "drame"
- Horror: "horror", "scary", "horer", "horor"
- Thriller: "thriller", "thriler", "suspense"
- Sci-Fi: "sci-fi", "science fiction", "scifi", "sci fi"
- Fantasy: "fantasy", "fantesy"
- Documentary: "documentary", "documentry", "docu"
- Animation: "animation", "animated", "animtion", "cartoon"

Extract the following information from the user's query and return as JSON:
- age_group: target age group ONLY if explicitly mentioned (Kids, Teens, Young Adults, Adults) - if not mentioned, use null
- genre: specific genre ONLY if explicitly mentioned - be extremely flexible with spelling variations - if not mentioned, use null
- year_range: [min_year, max_year] ONLY if years are explicitly mentioned - if not mentioned, use null
- country: specific country ONLY if explicitly mentioned - if not mentioned, use null
- popular: ONLY if user explicitly asks for popular/top movies (high, medium, low) OR specific rating (1, 2, 3, 4, 5) - if not mentioned, use null
- actor: actor/actress name ONLY if explicitly mentioned - if not mentioned, use null
- director: director name ONLY if explicitly mentioned - if not mentioned, use null
- runtime: ONLY if duration is explicitly mentioned, convert to minutes (e.g., "two hours" = 120, "90 minutes" = 90, "hour and half" = 90) - if not mentioned, use null
- runtime_operator: ONLY if runtime comparison is mentioned: "greater_than", "less_than", "equal_to" or "between" - if not mentioned, use null
- description_keywords: array of keywords describing plot/story elements (e.g., for "movie about a missing doctor" extract ["missing", "doctor"]) - if no plot description, use null
- intent: the main intent (recommend, check_suitability, filter, general_movie_question, off_topic)

Return JSON format only."""

        try:
            if self.model:
                prompt = f"{system_prompt}\n\nUser query: {user_query}"
                response = self.model.generate_content(prompt)

                # Extract JSON from response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()

                params = json.loads(response_text)
                print(f"DEBUG: Gemini extracted parameters: {params}")
                return params
            else:
                return self.basic_parameter_extraction(user_query, conversation_context)

        except Exception as e:
            print(f"DEBUG: Gemini failed, using fallback: {str(e)}")
            return self.basic_parameter_extraction(user_query, conversation_context)

    def basic_parameter_extraction(self, query, conversation_context=""):
        """Basic fallback parameter extraction without AI."""
        params = {
            'age_group': None,
            'genre': None,
            'year_range': None,
            'country': None,
            'popular': None,
            'actor': None,
            'director': None,
            'runtime': None,
            'runtime_operator': None,
            'description_keywords': None,
            'intent': 'recommend'
        }
        
        # Check if this is a follow-up query based on conversation context
        if conversation_context:
            if self.is_followup_query(query, conversation_context):
                context_params = self.extract_context_parameters(conversation_context)
                params.update(context_params)
        
        # Extract additional parameters from the current query itself 
        query_lower = query.lower()
        
        # Kids/Family detection
        kids_indicators = [
            'for kids', 'for children', 'suitable for kids', 'family friendly',
            'children can watch', 'kids can watch'
        ]
        
        for indicator in kids_indicators:
            if indicator in query_lower:
                params['age_group'] = 'Kids'
                print(f"DEBUG: Detected kids request from current query: {indicator}")
                break

        # Genre detection using fuzzy matching
        genre_keywords = {
            'romance': ['romance', 'romantic'],
            'action': ['action'],
            'comedy': ['comedy', 'comedies', 'funny'],
            'drama': ['drama'],
            'horror': ['horror', 'scary'],
            'thriller': ['thriller', 'suspense'],
            'sci-fi': ['sci-fi', 'science fiction', 'scifi'],
            'fantasy': ['fantasy'],
            'documentary': ['documentary', 'docu'],
            'animation': ['animation', 'animated', 'cartoon']
        }

        # Split query into words for fuzzy matching
        query_words = query.lower().split()

        for genre, keywords in genre_keywords.items():
            for word in query_words:
                # Use higher threshold for more accurate matching
                if len(word) >= 3 and self.fuzzy_match(word, keywords, threshold=0.8):
                    params['genre'] = genre.title()
                    break
            if params['genre']:
                break

        # Extract year from current query if present
        import re
        year_match = re.search(r'\b(19|20)(\d{2})\b', query)
        if year_match:
            year = int(year_match.group(0))
            params['year_range'] = [year, year]

        # Handle temporal keywords (recent, latest, new, etc.)
        if any(word in query_lower for word in ['recent', 'latest', 'new', 'newer']):
            # Set to recent years (2019-2021)
            params['year_range'] = [2019, 2021]

        return params

    def is_followup_query(self, query, context):
        """Use Gemini to determine if query is a follow-up or new topic."""
        if not context.strip():
            return False
            
        try:
            if self.model:
                prompt = f"""
                Based on the conversation context and the new query, determine if the user is:
                1. Making a FOLLOWUP request (adding filters to previous recommendations)
                2. Starting a NEW_TOPIC (completely different request)
                
                Context: {context}
                New query: {query}
                
                Examples:
                - "only romantic" after asking for kids movies = FOLLOWUP
                - "only from 2019" after previous recommendations = FOLLOWUP
                - "what about action movies?" after drama recommendations = NEW_TOPIC
                - "recommend horror movies" after comedy recommendations = NEW_TOPIC
                
                Answer with only: FOLLOWUP or NEW_TOPIC
                """
                
                response = self.model.generate_content(prompt)
                result = response.text.strip().upper()
                
                if "FOLLOWUP" in result:
                    print(f"DEBUG: Gemini detected FOLLOWUP")
                    return True
                elif "NEW_TOPIC" in result:
                    print(f"DEBUG: Gemini detected NEW_TOPIC")
                    return False
                else:
                    print(f"DEBUG: Gemini unclear response: {result}, assuming followup")
                    return True
                    
        except Exception as e:
            print(f"DEBUG: Gemini failed: {str(e)}, using simple fallback")
            # Simple fallback: short queries are usually follow-ups
            return len(query.split()) <= 3


    def extract_context_parameters(self, context):
        """Extract relevant parameters from conversation context."""
        context_params = {}
        
        try:
            # Extract previous parameters using basic text analysis
            context_lower = context.lower()
            
            # Look for age group mentions
            if any(word in context_lower for word in ['kids', 'children', 'family']):
                context_params['age_group'] = 'Kids'
                print(f"DEBUG: Inherited age_group: Kids")
            
            # Look for genre mentions
            genre_keywords = {
                'Drama': ['drama', 'dramas'],
                'Comedy': ['comedy', 'comedies'],
                'Action': ['action'],
                'Romance': ['romance', 'romantic'],
                'Horror': ['horror'],
                'Thriller': ['thriller'],
                'Animation': ['animation', 'animated']
            }
            
            for genre, keywords in genre_keywords.items():
                if any(keyword in context_lower for keyword in keywords):
                    context_params['genre'] = genre
                    print(f"DEBUG: Inherited genre: {genre}")
                    break
            
            # Look for year mentions
            year_match = re.search(r'\b(19|20)(\d{2})\b', context)
            if year_match:
                year = int(year_match.group(0))
                context_params['year_range'] = [year, year]
                print(f"DEBUG: Inherited year: {year}")
                
        except Exception as e:
            print(f"DEBUG: Error extracting context parameters: {e}")
            
        return context_params

    def filter_movies(self, params):
        """Filter movies based on extracted parameters."""
        print(f"DEBUG: Starting with {len(self.movies)} movies")
        print(f"DEBUG: Parameters: {params}")
        
        filtered = self.movies.copy()
        is_specific_search = bool(params.get('description_keywords'))

        # Genre filtering with mapping to actual CSV genres
        if params.get('genre') and params['genre'] != 'Unknown':
            genre = params['genre'].lower()

            # Map user genres to actual CSV genre formats
            genre_mapping = {
                'romance': 'romantic',
                'romantic': 'romantic',
                'action': 'action',
                'comedy': 'comedies',
                'drama': 'dramas',
                'horror': 'horror',
                'thriller': 'thriller',
                'sci-fi': 'sci-fi',
                'fantasy': 'fantasy',
                'documentary': 'documentaries',
                'animation': 'children'
            }

            # Use mapped genre or original if no mapping exists
            search_genre = genre_mapping.get(genre, genre)
            genre_mask = filtered['genre'].str.contains(search_genre, case=False, na=False)
            filtered = filtered[genre_mask]
            print(f"DEBUG: After genre filter ({search_genre}): {len(filtered)} movies")

        # Year range filtering
        if params.get('year_range'):
            year_range = params['year_range']
            if isinstance(year_range, list) and len(year_range) == 2:
                min_year, max_year = year_range
                year_mask = (filtered['released'] >= min_year) & (filtered['released'] <= max_year)
                filtered = filtered[year_mask]
                print(f"DEBUG: After year filter ({min_year}-{max_year}): {len(filtered)} movies")

        # Age group filtering
        if params.get('age_group'):
            age_mask = filtered['age_group'] == params['age_group']
            filtered = filtered[age_mask]
            print(f"DEBUG: After age group filter ({params['age_group']}): {len(filtered)} movies")

        # Runtime filtering
        if params.get('runtime') and params.get('runtime_operator'):
            runtime = params['runtime']
            operator = params['runtime_operator']

            if operator == 'less_than':
                runtime_mask = filtered['runtime'] <= runtime
            elif operator == 'greater_than':
                runtime_mask = filtered['runtime'] >= runtime
            elif operator == 'equal_to':
                runtime_mask = (filtered['runtime'] >= runtime - 10) & (filtered['runtime'] <= runtime + 10)
            elif operator == 'between' and isinstance(runtime, list) and len(runtime) == 2:
                runtime_mask = (filtered['runtime'] >= runtime[0]) & (filtered['runtime'] <= runtime[1])
            else:
                runtime_mask = pd.Series([True] * len(filtered))

            filtered = filtered[runtime_mask]

        # Actor filtering
        if params.get('actor'):
            actor = params['actor']
            actor_mask = filtered['cast'].str.contains(actor, case=False, na=False)
            filtered = filtered[actor_mask]

        # Director filtering
        if params.get('director'):
            director = params['director']
            director_mask = filtered['director'].str.contains(director, case=False, na=False)
            filtered = filtered[director_mask]

        # Country filtering
        if params.get('country'):
            country = params['country']
            country_mask = filtered['country'].str.contains(country, case=False, na=False)
            filtered = filtered[country_mask]

        # Description keyword filtering with relevance scoring
        if params.get('description_keywords'):
            keywords = params['description_keywords']
            filtered['keyword_score'] = 0

            for keyword in keywords:
                if len(keyword) > 2:
                    keyword_mask = filtered['description'].str.contains(keyword, case=False, na=False)
                    filtered.loc[keyword_mask, 'keyword_score'] += 1

            keyword_filtered = filtered[filtered['keyword_score'] > 0]
            if not keyword_filtered.empty:
                filtered = keyword_filtered

        # Ensure we have results before sorting
        if filtered.empty:
            return filtered

        # Only randomize for general searches, not specific ones
        if not is_specific_search:
            import random
            random.seed()
            filtered = filtered.sample(frac=1).reset_index(drop=True)

        # Smart sorting logic based on search type
        if is_specific_search:
            if 'keyword_score' in filtered.columns:
                filtered = filtered.sort_values(['keyword_score', 'popular', 'released'], ascending=[False, False, False])
            else:
                filtered = filtered.sort_values(['popular', 'released'], ascending=[False, False])
        else:
            filtered['combined_score'] = (0.7 * filtered['popular']) + (0.3 * (filtered['released'] - 2000) / 24)
            filtered = filtered.sort_values('combined_score', ascending=False)

        result = filtered.head(6)
        
        # Debug output
        if not result.empty:
            print("DEBUG: Final results summary:")
            for i, (_, movie) in enumerate(result.iterrows(), 1):
                print(f"  {i}. {movie.get('name', 'Unknown')} - Popularity: {movie.get('popular', 'N/A')}")
        
        return result

    def get_recommendation(self, user_query, conversation_context=""):
        """Main method to get movie recommendations based on user query."""
        try:
            print(f"DEBUG: Processing query: {user_query}")
            print(f"DEBUG: Conversation context: {conversation_context[:100]}...")
            
            # Check for off-topic queries first
            off_topic_keywords = ['weather', 'politics', 'cooking', 'sports', 'news', 'health', 'travel']
            if any(keyword in user_query.lower() for keyword in off_topic_keywords):
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about movie recommendations, actors, directors, or anything related to films!"

            # Extract parameters from query
            params = self.extract_query_parameters(user_query, conversation_context)
            
            # Check if context was used for follow-up
            if conversation_context and self.is_followup_query(user_query, conversation_context):
                print("DEBUG: Context was provided - checking if parameters inherited correctly")

            # Handle off-topic intent
            if params.get('intent') == 'off_topic':
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about movie recommendations, actors, directors, or anything related to films!"

            # Filter movies based on parameters
            filtered_movies = self.filter_movies(params)

            # Generate response
            if not filtered_movies.empty:
                return self.generate_response(filtered_movies, params, user_query)
            else:
                return self.suggest_alternatives(conversation_context, user_query)

        except Exception as e:
            print(f"DEBUG: Error in get_recommendation: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."

    def suggest_alternatives(self, conversation_context, user_query):
        """Suggest alternative genres when user doesn't like previous recommendations."""
        try:
            if self.model and conversation_context:
                prompt = f"""
                Based on the conversation context and user's current request, suggest alternative movie genres.
                
                Context: {conversation_context}
                Current request: {user_query}
                
                Provide 3 alternative genre suggestions in a friendly, helpful tone.
                Focus on genres that might appeal to someone who didn't like the previous recommendations.
                """
                
                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return "I couldn't find any movies matching your specific criteria. Try asking for different genres like action, comedy, drama, or sci-fi movies!"
                
        except Exception as e:
            return "I couldn't find any movies matching your specific criteria. Try broadening your search or asking for different genres, years, or actors."

    def generate_response(self, filtered_movies, params, original_query):
        """Generate a natural language response using Gemini."""
        try:
            if self.model and len(filtered_movies) > 0:
                # Prepare movie data for Gemini
                movie_list = []
                for _, movie in filtered_movies.iterrows():
                    movie_info = {
                        'name': movie.get('name', 'Unknown'),
                        'year': movie.get('released', 'Unknown'),
                        'genre': movie.get('genre', 'Unknown'),
                        'popularity': movie.get('popular', 'Unknown'),
                        'description': movie.get('description', 'No description available')[:200]
                    }
                    movie_list.append(movie_info)

                prompt = f"""
                Generate a friendly, engaging response for movie recommendations.
                
                User's original query: {original_query}
                Extracted parameters: {params}
                
                Recommended movies: {json.dumps(movie_list, indent=2)}
                
                Guidelines:
                - Be conversational and enthusiastic
                - Mention why these movies match their criteria
                - Include brief descriptions or highlights
                - Keep it concise but informative
                - End with an invitation for more questions
                """

                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return self.generate_fallback_response(filtered_movies, params)

        except Exception as e:
            print(f"DEBUG: Gemini failed for response generation: {str(e)}")
            return self.generate_fallback_response(filtered_movies, params)

    def generate_fallback_response(self, filtered_movies, params=None):
        """Generate a basic response without AI."""
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Try asking for different genres or removing some filters."

        response_parts = ["Here are some great movie recommendations for you:\n\n"]
        
        for i, (_, movie) in enumerate(filtered_movies.iterrows(), 1):
            movie_name = movie.get('name', 'Unknown Movie')
            movie_year = movie.get('released', 'Unknown')
            movie_genre = movie.get('genre', 'Unknown')
            
            response_parts.append(f"{i}. **{movie_name}** ({movie_year}) - {movie_genre}")
        
        response_parts.append("\n\nWould you like more recommendations or information about any of these movies?")
        
        return "\n".join(response_parts)


def initialize_system():
    """Initialize the movie recommendation system."""
    global recommender
    
    print("=" * 50)
    print("Movie Recommendation Chatbot")
    print("=" * 50)
    print("Initializing Movie Recommendation System...")
    
    try:
        csv_path = 'attached_assets/MergeAndCleaned_Movies.csv'
        recommender = MovieRecommender(csv_path)
        print("System initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize system: {str(e)}")
        return False


def setup_routes():
    """Setup Flask routes."""
    
    @app.route('/')
    def index():
        return render_template('index.html')

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
        
        # Keep only last 10 conversations
        conversation_memory[user_id] = conversation_memory[user_id][-10:]

    def get_conversation_context(user_id):
        """Get recent conversation context for user"""
        if user_id not in conversation_memory:
            return ""
        
        # Get last 3 conversations for context
        recent_conversations = conversation_memory[user_id][-3:]
        context_parts = []
        
        for conv in recent_conversations:
            context_parts.append(f"User: {conv['user_query']}")
            context_parts.append(f"Assistant: {conv['response'][:200]}...")
        
        return "Previous conversation:\n" + "\n".join(context_parts)

    @app.route('/recommend', methods=['POST'])
    def recommend():
        try:
            user_query = request.json.get('query', '').strip()
            if not user_query:
                return jsonify({'error': 'Please provide a query'}), 400

            user_id = get_user_id()
            conversation_context = get_conversation_context(user_id)
            
            if recommender:
                response = recommender.get_recommendation(user_query, conversation_context)
                save_conversation(user_id, user_query, response)
                return jsonify({'response': response})
            else:
                return jsonify({'error': 'Recommendation system not initialized'}), 500

        except Exception as e:
            print(f"Error in recommend route: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500


def start_server():
    """Start the Flask development server."""
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)


def main():
    """Main function to run the application."""
    if initialize_system():
        setup_routes()
        start_server()
    else:
        print("Failed to start the application.")


if __name__ == '__main__':
    main()