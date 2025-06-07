from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import json
import os
import google.generativeai as genai
import re
import uuid
from datetime import datetime

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
        print(f"DEBUG: Processing query: {user_query}")
        print(f"DEBUG: Conversation context: {conversation_context[:200]}...")

        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries.

IMPORTANT: Handle Hebrew text, English text, mixed Hebrew-English, and typos. Be extremely flexible with genre recognition and spelling variations.

CONTEXT HANDLING: When analyzing the current query, consider the previous conversation context to understand:
- Follow-up questions (e.g., "only from 2019" after asking for kids movies)
- Refinements (e.g., "something newer" or "more recent")
- Continuations (e.g., "and also" or "what about")
- References to previous recommendations

CRITICAL INHERITANCE RULE: When conversation context is provided, you MUST ALWAYS extract parameters from BOTH the context AND the current query. Never ignore context parameters.

MANDATORY EXAMPLES:
Context: "User: give me drama movies" → Current: "for adults only"
MUST RETURN: genre: "Drama", age_group: "Adults"

Context: "User: drama movies" → Current: "what released in 2019?"  
MUST RETURN: genre: "Drama", year_range: [2019, 2019]

Context: "User: romantic movies" → Current: "from 2020"
MUST RETURN: genre: "Romance", year_range: [2020, 2020]

RULE: If the context mentions ANY parameter (genre, age_group, actor, etc.), include it in your response even if the current query doesn't mention it.

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

SPECIAL HANDLING FOR MOVIE REFERENCES:
If the user asks about "this movie", "that movie", "the movie", or similar references without naming it specifically, check the conversation context for any movie titles mentioned in previous assistant responses. If found, set description_keywords to search for that specific movie title.

Examples:
- Context shows assistant mentioned "My Octopus Teacher" → User asks "about what this movie?" → description_keywords: ["My Octopus Teacher"]
- Context shows assistant mentioned "Charming" → User asks "tell me about that movie" → description_keywords: ["Charming"]

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
        if conversation_context:
            if self.is_followup_query(query, conversation_context):
                context_params = self.extract_context_parameters(conversation_context)
                # Merge context parameters with current params (context has priority for missing values)
                for key, value in context_params.items():
                    if value is not None and params.get(key) is None:
                        params[key] = value
                        print(f"DEBUG: Inherited {key} from context: {value}")

        # Extract additional parameters from the current query itself
        # (beyond what was extracted from context)
        query_lower = query.lower()

        # Kids/Family detection
        kids_indicators = [
            'for kids', 'for children', 'suitable for kids', 'suit to kids',
            'that will suit', 'appropriate for kids', 'family friendly',
            'children can watch', 'kids can watch', 'child friendly'
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
                    params['genre'] = genre
                    print(f"DEBUG: Detected genre '{genre}' from word '{word}'")
                    break
            if params['genre']:
                break

        # Year range detection
        import re
        
        # Look for patterns like "2014-2016", "from 2014 to 2016", "between 2014 and 2016"
        range_patterns = [
            r'(\d{4})\s*-\s*(\d{4})',  # 2014-2016
            r'from\s+(\d{4})\s+to\s+(\d{4})',  # from 2014 to 2016
            r'between\s+(\d{4})\s+and\s+(\d{4})'  # between 2014 and 2016
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                start_year = int(match.group(1))
                end_year = int(match.group(2))
                params['year_range'] = [start_year, end_year]
                print(f"DEBUG: Detected year range: {start_year}-{end_year}")
                break
        
        # If no range found, look for single year
        if not params.get('year_range'):
            year_match = re.search(r'\b(19|20)(\d{2})\b', query)
            if year_match:
                year = int(year_match.group(0))
                params['year_range'] = [year, year]
                print(f"DEBUG: Detected single year: {year}")

        return params

    def is_followup_query(self, query, context):
        """Use Gemini to determine if query is a follow-up or new topic."""
        if not context.strip():
            return False

        if not self.model:
            # Simple fallback if no AI available
            query_words = query.lower().split()
            followup_indicators = ['only', 'just', 'from', 'in', 'with', 'but', 'and', 'or', 'also']
            return len(query_words) <= 4 or any(word in followup_indicators for word in query_words[:2])

        try:
            prompt = f"""
Analyze if the user's current query is a follow-up to the previous conversation or a completely new movie request.

Previous conversation:
{context}

Current query: {query}

Return "FOLLOWUP" if:
- The query refines/filters previous results (e.g., "only from 2019", "but romantic", "with Tom Cruise")
- It's a short query that builds on context (e.g., "only romantic", "from 2020", "for adults")
- Uses words like "only", "just", "but", "and", "also", "with", "from"

Return "NEW" if:
- It's a completely different movie request
- Asks about a different genre/topic without reference to previous conversation
- Is a long, complete sentence starting fresh

Response (just FOLLOWUP or NEW):"""

            response = self.model.generate_content(prompt)
            result = response.text.strip().upper()
            is_followup = "FOLLOWUP" in result
            print(f"DEBUG: Followup analysis for '{query}': {is_followup}")
            return is_followup

        except Exception as e:
            print(f"DEBUG: Error in followup detection: {e}")
            # Fallback to simple heuristic
            query_words = query.lower().split()
            followup_indicators = ['only', 'just', 'from', 'in', 'with', 'but', 'and', 'or', 'also']
            return len(query_words) <= 4 or any(word in followup_indicators for word in query_words[:2])

    def extract_context_parameters(self, context):
        """Extract relevant parameters from conversation context."""
        params = {}
        
        # Simple extraction from context
        context_lower = context.lower()
        
        # Look for age group mentions in context
        if 'kids' in context_lower or 'children' in context_lower:
            params['age_group'] = 'Kids'
            print(f"DEBUG: Inherited age_group 'Kids' from context")
        elif 'adults' in context_lower:
            params['age_group'] = 'Adults'
            print(f"DEBUG: Inherited age_group 'Adults' from context")
        elif 'teens' in context_lower:
            params['age_group'] = 'Teens'
            print(f"DEBUG: Inherited age_group 'Teens' from context")
            
        # Look for genre mentions in context
        genre_patterns = {
            'romance': ['romantic', 'romance'],
            'action': ['action'],
            'comedy': ['comedy', 'comedies', 'funny'],
            'drama': ['drama'],
            'horror': ['horror', 'scary'],
            'thriller': ['thriller'],
            'sci-fi': ['sci-fi', 'science fiction'],
            'fantasy': ['fantasy'],
            'documentary': ['documentary'],
            'animation': ['animation', 'animated']
        }
        
        for genre, keywords in genre_patterns.items():
            for keyword in keywords:
                if keyword in context_lower:
                    params['genre'] = genre
                    print(f"DEBUG: Inherited genre '{genre}' from context (found '{keyword}')")
                    break
            if params.get('genre'):
                break
                
        return params

    def filter_movies(self, params):
        """Filter movies based on extracted parameters using algorithmic approach."""
        print(f"DEBUG: Starting with {len(self.movies)} movies")
        print(f"DEBUG: Parameters: {params}")
        
        filtered = self.movies.copy()
        
        # Genre filtering - simple mapping to actual dataset values
        if params.get('genre'):
            user_genre = params['genre'].lower()
            print(f"DEBUG: User requested genre: {user_genre}")
            
            # Simple mapping to actual genre names in dataset
            genre_mapping = {
                'comedy': 'Comedies',
                'action': 'Action & Adventure', 
                'drama': 'Dramas',
                'horror': 'Horror Movies',
                'thriller': 'Thrillers',
                'romance': 'Romantic Movies',
                'romantic': 'Romantic Movies',
                'sci-fi': 'Sci-Fi & Fantasy',
                'fantasy': 'Sci-Fi & Fantasy',
                'documentary': 'Documentaries',
                'animation': 'Anime Features'
            }
            
            # Get the actual genre name from dataset
            actual_genre = genre_mapping.get(user_genre, user_genre.title())
            genre_filter = filtered['genre'].str.contains(actual_genre, case=False, na=False)
            print(f"DEBUG: Looking for genre: {actual_genre}")
            
            filtered = filtered[genre_filter]
            print(f"DEBUG: After genre filtering: {len(filtered)} movies")

        # Year range filtering
        if params.get('year_range'):
            year_range = params['year_range']
            print(f"DEBUG: Filtering by year range: {year_range}")
            
            if len(year_range) == 2:
                start_year, end_year = year_range
                year_filter = (filtered['released'] >= start_year) & (filtered['released'] <= end_year)
                filtered = filtered[year_filter]
                print(f"DEBUG: After year filtering ({start_year}-{end_year}): {len(filtered)} movies")

        # Age group filtering - use actual age_group column data
        if params.get('age_group'):
            age_group = params['age_group']
            print(f"DEBUG: Filtering by age_group: {age_group}")
            
            if 'age_group' in filtered.columns:
                age_filter = filtered['age_group'] == age_group
                filtered = filtered[age_filter]
                print(f"DEBUG: After age_group filtering: {len(filtered)} movies for {age_group}")
            else:
                print(f"DEBUG: No age_group column found in dataset")

        # Sort by popularity (higher is better)
        if not filtered.empty and 'popular' in filtered.columns:
            filtered = filtered.sort_values('popular', ascending=False)
            
        return filtered

    def get_recommendation(self, user_query, conversation_context=""):
        """Main method to get movie recommendations based on user query."""
        print(f"DEBUG: Processing query: {user_query}")
        print(f"DEBUG: Conversation context: {conversation_context[:100]}...")
        
        # Extract parameters from query and context
        params = self.extract_query_parameters(user_query, conversation_context)
        filtered_movies = self.filter_movies(params)
        
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Try adjusting your search parameters."
        
        # Generate personalized response with proper formatting
        return self.generate_personalized_intro(params) + self.format_movie_list(filtered_movies.head(10))

    def generate_personalized_intro(self, params):
        """Generate personalized introduction based on search parameters."""
        intro_parts = []
        
        if params.get('genre'):
            intro_parts.append(f"{params['genre']} movies")
        else:
            intro_parts.append("movies")
            
        if params.get('age_group'):
            intro_parts.append(f"that are suitable for {params['age_group'].lower()}")
            
        if params.get('year_range'):
            year_range = params['year_range']
            if year_range[0] == year_range[1]:
                intro_parts.append(f"from {year_range[0]}")
            else:
                intro_parts.append(f"from {year_range[0]}-{year_range[1]}")
        
        intro = "Here are " + " ".join(intro_parts) + ":"
        return intro
        
    def format_movie_list(self, movies):
        """Format movie list for display."""
        if movies.empty:
            return ""
            
        result = []
        for _, movie in movies.head(10).iterrows():
            movie_info = f"• {movie['name']}"
            if pd.notna(movie['released']):
                movie_info += f" ({int(movie['released'])})"
            if pd.notna(movie['genre']):
                movie_info += f" - {movie['genre']}"
            result.append(movie_info)
        
        return "\n" + "\n".join(result)


def initialize_system():
    """Initialize the movie recommendation system."""
    global recommender
    print("=" * 50)
    print("Movie Recommendation Chatbot")
    print("=" * 50)
    print("Initializing Movie Recommendation System...")
    
    try:
        # Initialize the recommender with the CSV file
        csv_path = "./attached_assets/MergeAndCleaned_Movies.csv"
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
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
            'user': user_query,
            'assistant': response
        })
        
        # Keep only last 10 exchanges per user
        if len(conversation_memory[user_id]) > 10:
            conversation_memory[user_id] = conversation_memory[user_id][-10:]

    def get_conversation_context(user_id):
        """Get recent conversation context for user"""
        if user_id not in conversation_memory:
            return ""
        
        # Get last 3 exchanges for context
        recent_conversations = conversation_memory[user_id][-3:]
        context_parts = []
        
        for conv in recent_conversations:
            context_parts.append(f"User: {conv['user']}")
            context_parts.append(f"Assistant: {conv['assistant'][:200]}")  # Limit assistant response length
        
        return "\n".join(context_parts)

    @app.route('/recommend', methods=['POST'])
    def recommend():
        try:
            data = request.json
            user_query = data.get('query', '').strip()
            
            if not user_query:
                return jsonify({'error': 'Please provide a valid query'}), 400
            
            user_id = get_user_id()
            conversation_context = get_conversation_context(user_id)
            
            # Get recommendation
            if recommender:
                response = recommender.get_recommendation(user_query, conversation_context)
            else:
                response = "System not initialized. Please try again later."
            
            # Save conversation
            save_conversation(user_id, user_query, response)
            
            return jsonify({
                'response': response,
                'success': True
            })
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return jsonify({'error': 'An error occurred processing your request'}), 500


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


if __name__ == "__main__":
    main()