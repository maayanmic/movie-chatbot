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
                self.model = genai.GenerativeModel('gemini-pro')
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
        print(f"DEBUG: Processing query: {user_query}")
        print(f"DEBUG: Conversation context: {conversation_context[:200]}...")

        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries.

IMPORTANT: Handle Hebrew text, English text, mixed Hebrew-English, and typos. Be extremely flexible with genre recognition and spelling variations.

CONTEXT HANDLING: When analyzing the current query, consider the previous conversation context to understand:
- Follow-up questions (e.g., "only from 2019" after asking for kids movies)
- Refinements (e.g., "something newer" or "more recent")
- Continuations (e.g., "and also" or "what about")
- References to previous recommendations

CRITICAL: If the user's query seems to be refining or building upon a previous request, you MUST:
1. Extract parameters from the CURRENT query
2. Inherit relevant parameters from the PREVIOUS query context
3. Combine them into a complete parameter set

For example, if previous query was "movies for kids" and current is "only from 2019", you should return:
- age_group: "Kids" (from previous context)
- year_range: [2019, 2019] (from current query)

EXAMPLES:
Previous: "romantic movies" → Current: "from 2020" → Return: genre: "romantic", year_range: [2020, 2020]
Previous: "action movies" → Current: "with Tom Cruise" → Return: genre: "action", actor: "Tom Cruise"

GENRE VARIATIONS TO RECOGNIZE (including common typos):
- Romance/Romantic: "romance", "romantic", "rommantic", "rommntic", "rommance", "romence", "romanc", "רומנטי", "רומנטית", "אהבה"
- Action: "action", "actoin", "akshen", "אקשן", "פעולה"
- Comedy: "comedy", "comdy", "komedia", "funny", "קומדיה"
- Drama: "drama", "drame", "דרמה"
- Horror: "horror", "scary", "horer", "horor", "אימה"
- Thriller: "thriller", "thriler", "suspense", "מתח"
- Sci-Fi: "sci-fi", "science fiction", "scifi", "sci fi", "מדע בדיוני"
- Fantasy: "fantasy", "fantesy", "פנטזיה"
- Documentary: "documentary", "documentry", "docu", "תיעודי"
- Animation: "animation", "animated", "animtion", "cartoon", "אנימציה"

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
                context_info = ""
                if conversation_context:
                    context_info = f"\n\nPrevious conversation context:\n{conversation_context}\n"
                
                prompt = f"{system_prompt}{context_info}\nUser query: {user_query}"
                response = self.model.generate_content(prompt)

                # Extract JSON from response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()

                params = json.loads(response_text)
                print(f"DEBUG: Gemini extracted parameters: {params}")
                if conversation_context:
                    print(f"DEBUG: Context was provided - checking if parameters inherited correctly")
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
            'description_keywords': None,
            'intent': 'recommend'
        }
        
        # Check if this is a follow-up query based on conversation context
        if conversation_context and self.is_followup_query(query, conversation_context):
            print("DEBUG: Detected follow-up query, extracting context parameters")
            context_params = self.extract_context_parameters(conversation_context)
            params.update(context_params)
        
        # Also extract kids-related parameters from the current query itself
        kids_indicators = [
            'for kids', 'for children', 'suitable for kids', 'suit to kids', 
            'that will suit', 'appropriate for kids', 'family friendly',
            'children can watch', 'kids can watch', 'child friendly'
        ]
        
        query_lower = query.lower()
        for indicator in kids_indicators:
            if indicator in query_lower:
                params['age_group'] = 'Kids'
                print(f"DEBUG: Detected kids request from current query: {indicator}")
                break

        # Genre detection using fuzzy matching
        genre_keywords = {
            'romance': ['romance', 'romantic', 'רומנטי', 'רומנטית', 'אהבה'],
            'action': ['action', 'אקשן', 'פעולה'],
            'comedy': ['comedy', 'קומדיה', 'funny'],
            'drama': ['drama', 'דרמה'],
            'horror': ['horror', 'אימה', 'scary'],
            'thriller': ['thriller', 'מתח', 'suspense'],
            'sci-fi': ['sci-fi', 'science fiction', 'מדע בדיוני', 'scifi'],
            'fantasy': ['fantasy', 'פנטזיה'],
            'documentary': ['documentary', 'תיעודי', 'docu'],
            'animation': ['animation', 'animated', 'אנימציה', 'cartoon']
        }

        # Split query into words for fuzzy matching
        query_words = query.lower().split()

        for genre, keywords in genre_keywords.items():
            for word in query_words:
                if self.fuzzy_match(word, keywords, threshold=0.6):
                    params['genre'] = genre.title()
                    break
            if params['genre']:
                break

        # Extract keywords for description search - improved to catch more patterns
        description_keywords = []
        query_lower = query.lower()
        
        # Look for story patterns
        story_indicators = ['about', 'movie about', 'film about', 'story of', 'follows', 'centers on']
        
        for indicator in story_indicators:
            if indicator in query_lower:
                start_pos = query_lower.find(indicator) + len(indicator)
                remaining_text = query[start_pos:].strip()
                words = remaining_text.split()
                meaningful_words = [word for word in words if len(word) > 2 and  # Changed from 3 to 2
                                  word.lower() not in ['that', 'with', 'from', 'where', 'when', 'what', 'which', 'and', 'the', 'his', 'her']]
                description_keywords.extend(meaningful_words[:7])  # Increased from 5 to 7
                break
        
        # Also look for specific key terms in the query
        key_terms = ['doctor', 'psychiatrist', 'missing', 'wife', 'treat', 'medical', 'condition', 'patient']
        for term in key_terms:
            if term in query_lower and term not in description_keywords:
                description_keywords.append(term)

        if description_keywords:
            params['description_keywords'] = description_keywords[:7]  # Limit to 7 terms

        # Extract year from current query if present
        import re
        year_match = re.search(r'\b(19|20)(\d{2})\b', query)
        if year_match:
            year = int(year_match.group(0))
            params['year_range'] = [year, year]
            print(f"DEBUG: Extracted year {year} from query")

        return params

    def is_followup_query(self, query, context):
        """Check if the current query is a follow-up to previous conversation."""
        followup_indicators = [
            'only', 'just', 'from', 'in', 'with', 'by', 'after', 'before',
            'newer', 'older', 'recent', 'latest', 'also', 'too', 'and',
            'but', 'however', 'except', 'without', 'plus', 'for'
        ]
        
        query_lower = query.lower().strip()
        
        # Short queries with follow-up indicators are likely follow-ups
        if len(query_lower.split()) <= 4:
            for indicator in followup_indicators:
                # Check if query starts with the indicator OR contains it as a separate word
                if query_lower.startswith(indicator) or f' {indicator} ' in f' {query_lower} ':
                    return True
        
        # Check for year patterns (2019, 2020, etc.)
        import re
        if re.search(r'\b(19|20)\d{2}\b', query):
            return True
        
        # Check for kids-related follow-up queries
        kids_patterns = [
            'for kids', 'for children', 'suitable for kids', 'suit to kids', 
            'that will suit', 'appropriate for kids', 'family friendly',
            'children can watch', 'kids can watch'
        ]
        
        for pattern in kids_patterns:
            if pattern in query_lower:
                return True
            
        return False

    def extract_context_parameters(self, context):
        """Extract relevant parameters from conversation context."""
        params = {}
        
        # Look for kids/children mentions in context
        if any(word in context.lower() for word in ['kids', 'children', 'child', 'family']):
            params['age_group'] = 'Kids'
            
        # Look for genre mentions in context  
        genre_patterns = {
            'romance': ['romantic', 'romance', 'love', 'romantic movies'],
            'action': ['action'],
            'comedy': ['comedy', 'funny', 'comedies'],
            'drama': ['drama', 'dramas'],
            'horror': ['horror', 'scary'],
            'thriller': ['thriller', 'suspense']
        }
        
        context_lower = context.lower()
        for genre, keywords in genre_patterns.items():
            if any(keyword in context_lower for keyword in keywords):
                params['genre'] = genre.title()
                break
        
        # Extract the most recent genre from the last user query in context
        lines = context.split('\n')
        last_user_query = None
        for line in lines:
            if line.startswith('User: '):
                last_user_query = line[6:].lower()  # Remove 'User: ' prefix
        
        if last_user_query:
            for genre, keywords in genre_patterns.items():
                if any(keyword in last_user_query for keyword in keywords):
                    params['genre'] = genre.title()
                    print(f"DEBUG: Extracted genre '{genre.title()}' from last query: {last_user_query}")
                    break
                
        return params

    def filter_movies(self, params):
        """Filter movies based on extracted parameters."""
        filtered = self.movies.copy()
        is_specific_search = bool(params.get('description_keywords'))
        print(f"DEBUG: Starting with {len(filtered)} movies")
        print(f"DEBUG: Parameters: {params}")

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

        # Year range filtering
        if params.get('year_range'):
            year_range = params['year_range']
            if isinstance(year_range, list) and len(year_range) == 2:
                min_year, max_year = year_range
                year_mask = (filtered['released'] >= min_year) & (filtered['released'] <= max_year)
                filtered = filtered[year_mask]

        # Age group filtering
        if params.get('age_group'):
            age_mask = filtered['age_group'] == params['age_group']
            filtered = filtered[age_mask]

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
            print(f"DEBUG: Filtering by description keywords: {keywords}")

            for keyword in keywords:
                if len(keyword) > 2:
                    # Clean keyword from punctuation
                    clean_keyword = keyword.strip('.,!?;:')
                    print(f"DEBUG: Searching for keyword '{clean_keyword}'")
                    keyword_mask = filtered['description'].str.contains(clean_keyword, case=False, na=False)
                    matches = filtered[keyword_mask]
                    print(f"DEBUG: Found {len(matches)} movies with '{clean_keyword}'")
                    if clean_keyword == 'missing' and len(matches) > 0:
                        print(f"DEBUG: Movies with 'missing': {matches['name'].head(5).tolist()}")
                        # Check specifically for the movie with the exact description
                        exact_match = filtered[filtered['description'].str.contains("When a doctor goes missing, his psychiatrist wife treats", case=False, na=False)]
                        if not exact_match.empty:
                            movie_name = exact_match.iloc[0]['name']
                            print(f"DEBUG: Found exact match movie: '{movie_name}'")
                            if movie_name not in matches['name'].values:
                                print(f"DEBUG: But '{movie_name}' was NOT found in 'missing' search!")
                        else:
                            print("DEBUG: Exact description match not found")
                    filtered.loc[keyword_mask, 'keyword_score'] += 1

            keyword_filtered = filtered[filtered['keyword_score'] > 0]
            print(f"DEBUG: After description filtering: {len(keyword_filtered)} movies")
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
        print(f"DEBUG: Final results summary:")
        for i, (_, movie) in enumerate(result.iterrows(), 1):
            print(f"  {i}. {movie['name']} - Popularity: {movie['popular']}")
        return result

    def get_recommendation(self, user_query, conversation_context=""):
        """Main method to get movie recommendations based on user query."""
        try:
            # Check for off-topic queries first
            off_topic_keywords = ['weather', 'politics', 'cooking', 'sports', 'news', 'health', 'travel']
            if any(keyword in user_query.lower() for keyword in off_topic_keywords):
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about movie recommendations, actors, directors, or anything related to films!"

            # Extract parameters from query
            params = self.extract_query_parameters(user_query, conversation_context)

            # Handle off-topic intent
            if params.get('intent') == 'off_topic':
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about movie recommendations, actors, directors, or anything related to films!"

            # Filter movies based on parameters
            filtered_movies = self.filter_movies(params)

            # Generate response
            if not filtered_movies.empty:
                return self.generate_response(filtered_movies, params, user_query)
            else:
                return "I couldn't find any movies matching your specific criteria. Try broadening your search or asking for different genres, years, or actors."

        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a different query."

    def generate_response(self, filtered_movies, params, original_query):
        """Generate a natural language response using Gemini."""
        try:
            if self.model and not filtered_movies.empty:
                movies_text = ""
                for _, movie in filtered_movies.iterrows():
                    year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
                    genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
                    movies_text += f"• {movie['name']} ({year}) - {genre}\n"

                prompt = f"""Based on the user's query: "{original_query}"

Here are the most relevant movies from our database:
{movies_text}

Generate a helpful response in English. Start with a brief introduction, then list the movies with their details. Keep it conversational and informative."""

                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return self.generate_fallback_response(filtered_movies)

        except Exception as e:
            return self.generate_fallback_response(filtered_movies)

    def generate_fallback_response(self, filtered_movies):
        """Generate a basic response without AI."""
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Please try a different search."

        response = "Here are some movie recommendations for you:\n\n"
        for _, movie in filtered_movies.iterrows():
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown genre'
            response += f"• {movie['name']} ({year}) - {genre}\n"

        return response

def initialize_system():
    """Initialize the movie recommendation system."""
    global recommender
    print("Initializing Movie Recommendation System...")

    # Initialize the recommender
    csv_file = "attached_assets/MergeAndCleaned_Movies.csv"
    recommender = MovieRecommender(csv_file)

    print("System initialized successfully!")

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

        # Keep only last 10 conversations to avoid memory overflow
        if len(conversation_memory[user_id]) > 10:
            conversation_memory[user_id] = conversation_memory[user_id][-10:]

    def get_conversation_context(user_id):
        """Get recent conversation context for user"""
        if user_id not in conversation_memory:
            return ""

        context = "Previous conversation:\n"
        for conv in conversation_memory[user_id][-3:]:  # Last 3 conversations
            context += f"User: {conv['user_query']}\n"
            context += f"Assistant: {conv['response'][:150]}...\n\n"

        return context

    @app.route('/recommend', methods=['POST'])
    def recommend():
        try:
            data = request.get_json()
            query = data.get('query', '')

            if not query:
                return jsonify({'error': 'No query provided'}), 400

            user_id = get_user_id()

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
            print(f"Error in recommend endpoint: {str(e)}")
            return jsonify({'error': 'An error occurred processing your request'}), 500

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