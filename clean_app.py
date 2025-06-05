from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import json
import os
import google.generativeai as genai
import re
import uuid
from datetime import datetime

app = Flask(__name__, static_folder='.', template_folder='.')
app.secret_key = 'movie_recommender_secret_key_2024'

# Global conversation memory for session continuity
conversation_memory = {}

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
            
            print(f"DEBUG: CSV columns: {movies.columns.tolist()}")
            
            return movies
            
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def extract_query_parameters(self, user_query):
        """Use Gemini to extract parameters from natural language query."""
        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries.

IMPORTANT: Only extract parameters that are EXPLICITLY mentioned in the query. Do not infer or assume parameters.

Extract the following information from the user's query and return as JSON:
- age_group: target age group ONLY if explicitly mentioned (Kids, Teens, Young Adults, Adults) - if not mentioned, use null
- genre: specific genre ONLY if explicitly mentioned (e.g., Horror, Action, Drama, Comedy) - if not mentioned, use null
- year_range: [min_year, max_year] ONLY if years are explicitly mentioned - if not mentioned, use null
- country: specific country ONLY if explicitly mentioned - if not mentioned, use null
- popular: ONLY if user explicitly asks for popular/top movies (high, medium, low) OR specific rating (1, 2, 3, 4, 5) - if not mentioned, use null
- actor: actor/actress name ONLY if explicitly mentioned - if not mentioned, use null
- director: director name ONLY if explicitly mentioned - if not mentioned, use null
- runtime: ONLY if duration is explicitly mentioned, convert to minutes (e.g., "two hours" = 120, "90 minutes" = 90, "hour and half" = 90) - if not mentioned, use null
- runtime_operator: ONLY if runtime comparison is mentioned: "greater_than", "less_than", "equal_to" or "between" - if not mentioned, use null
- description_keywords: array of keywords describing plot/story elements (e.g., for "movie about a missing doctor" extract ["missing", "doctor"]) - if no plot description, use null
- intent: the main intent (recommend, check_suitability, filter, general_movie_question, off_topic)

IMPORTANT: Always convert time references to minutes:
- "two hours" = 120 minutes
- "one hour" = 60 minutes  
- "hour and half" = 90 minutes
- "90 minutes" = 90 minutes

Note: Age groups are: Kids (up to 7), Teens (8-13), Young Adults (14-17), Adults (18+)

Intent guidelines:
- recommend: User wants movie recommendations or lists
- check_suitability: User asks if specific movies match criteria
- general_movie_question: User asks general questions about movies, actors, directors (like "Is X also a drama movie?", "Who directed Y?", "What year was Z released?")
- off_topic: User asks about non-movie topics (politics, weather, cooking, etc.)

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
                return self.basic_parameter_extraction(user_query)
            
        except Exception as e:
            # Fallback to basic parameter extraction
            print(f"DEBUG: Gemini failed, using basic extraction: {str(e)}")
            params = self.basic_parameter_extraction(user_query)
            print(f"DEBUG: Basic extraction result: {params}")
            return params
    
    def basic_parameter_extraction(self, query):
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
        
        # Basic age group detection
        query_lower = query.lower()
        if any(word in query_lower for word in ['kid', 'child', 'children']):
            params['age_group'] = 'Kids'
        elif any(word in query_lower for word in ['teen', 'teenager']):
            params['age_group'] = 'Teens'
        elif any(word in query_lower for word in ['young adult', 'college']):
            params['age_group'] = 'Young Adults'
        elif any(word in query_lower for word in ['adult', 'mature']):
            params['age_group'] = 'Adults'
        
        # Basic genre detection
        genres = ['horror', 'action', 'drama', 'comedy', 'romance', 'sci-fi', 'fantasy', 'documentary', 'thriller']
        for genre in genres:
            if genre in query_lower:
                params['genre'] = genre.title()
                break
        
        # Extract keywords for description search
        description_keywords = []
        story_indicators = ['about', 'movie about', 'film about', 'story of', 'follows', 'centers on']
        
        for indicator in story_indicators:
            if indicator in query_lower:
                # Extract text after the indicator
                start_pos = query_lower.find(indicator) + len(indicator)
                remaining_text = query[start_pos:].strip()
                # Extract meaningful words (skip common words)
                words = remaining_text.split()
                meaningful_words = [word for word in words if len(word) > 3 and 
                                  word.lower() not in ['that', 'with', 'from', 'where', 'when', 'what', 'which']]
                description_keywords.extend(meaningful_words[:5])  # Take first 5 meaningful words
                break
        
        if description_keywords:
            params['description_keywords'] = description_keywords
        
        return params
    
    def filter_movies(self, params):
        """Filter movies based on extracted parameters."""
        filtered = self.movies.copy()
        print(f"DEBUG: Starting with {len(filtered)} movies")
        print(f"DEBUG: Parameters: {params}")
        
        # Different handling for specific vs general searches
        is_specific_search = bool(params.get('description_keywords'))
        
        # Age group filtering - skip if we have description keywords (prioritize content over demographics)
        if params.get('age_group') and params['age_group'] != 'Unknown' and not params.get('description_keywords'):
            age_group = params['age_group']
            print(f"DEBUG: Filtering by age group: {age_group}")
            # Try exact match first
            exact_match = filtered[filtered['age_group'].str.lower() == age_group.lower()]
            if not exact_match.empty:
                filtered = exact_match
            else:
                # Fallback to partial match
                age_mask = filtered['age_group'].str.contains(age_group, case=False, na=False)
                filtered = filtered[age_mask]
            print(f"DEBUG: After age filtering: {len(filtered)} movies")
        elif params.get('description_keywords'):
            print(f"DEBUG: Skipping age filter due to description search priority")
        
        # Genre filtering with better mapping (skip if Unknown)
        if params.get('genre') and params['genre'] != 'Unknown':
            genre = params['genre'].lower()
            print(f"DEBUG: Filtering by genre: {genre}")
            
            # Genre mapping for better matching
            genre_mappings = {
                'horror': ['horror', 'scary'],
                'action': ['action', 'adventure'],
                'drama': ['drama', 'dramatic'],
                'comedy': ['comedy', 'comedies', 'funny'],
                'romance': ['romance', 'romantic', 'love'],
                'sci-fi': ['sci-fi', 'science fiction', 'sci fi', 'scifi'],
                'fantasy': ['fantasy', 'magical'],
                'documentary': ['documentary', 'documentaries'],
                'thriller': ['thriller', 'suspense']
            }
            
            search_terms = genre_mappings.get(genre, [genre])
            genre_mask = pd.Series([False] * len(filtered), index=filtered.index)
            
            for term in search_terms:
                term_mask = filtered['genre'].str.contains(term, case=False, na=False)
                genre_mask = genre_mask | term_mask
            
            filtered = filtered[genre_mask]
            print(f"DEBUG: After genre filtering: {len(filtered)} movies")
        
        # Year range filtering
        if params.get('year_range'):
            year_range = params['year_range']
            if isinstance(year_range, list) and len(year_range) == 2:
                min_year, max_year = year_range
                print(f"DEBUG: Filtering by year range: {min_year}-{max_year}")
                year_mask = (filtered['released'] >= min_year) & (filtered['released'] <= max_year)
                filtered = filtered[year_mask]
                print(f"DEBUG: After year filtering: {len(filtered)} movies")
        
        # Country filtering
        if params.get('country'):
            country = params['country']
            print(f"DEBUG: Filtering by country: {country}")
            if 'country' in filtered.columns:
                country_mask = filtered['country'].str.contains(country, case=False, na=False)
                filtered = filtered[country_mask]
                print(f"DEBUG: After country filtering: {len(filtered)} movies")
        
        # Description keyword filtering with relevance scoring
        if params.get('description_keywords'):
            keywords = params['description_keywords']
            print(f"DEBUG: Filtering by description keywords: {keywords}")
            
            # Create a relevance score for each movie based on keyword matches
            filtered['keyword_score'] = 0
            
            for keyword in keywords:
                if len(keyword) > 2:  # Only search meaningful keywords
                    keyword_mask = filtered['description'].str.contains(keyword, case=False, na=False)
                    # Add points for each keyword match
                    filtered.loc[keyword_mask, 'keyword_score'] += 1
            
            # Only keep movies that have at least one keyword match
            keyword_filtered = filtered[filtered['keyword_score'] > 0]
            
            if not keyword_filtered.empty:
                filtered = keyword_filtered
                print(f"DEBUG: After description filtering: {len(filtered)} movies")
                
                # Debug: Check if movie "706" is in the results
                movie_706 = filtered[filtered['name'] == '706']
                if not movie_706.empty:
                    score = movie_706.iloc[0]['keyword_score']
                    print(f"DEBUG: Found movie '706' with popularity {movie_706.iloc[0]['popular']} and keyword score {score}")
                else:
                    print(f"DEBUG: Movie '706' not found in filtered results")
                    
            else:
                print(f"DEBUG: No movies found matching description keywords")

        # Ensure we have results before sorting
        if filtered.empty:
            return filtered
            
        # Only randomize for general searches, not specific ones
        if not is_specific_search:
            import random
            random.seed()  # Use current time as seed for true randomization
            filtered = filtered.sample(frac=1).reset_index(drop=True)
        
        # Popularity filtering - handle specific numbers and ranges
        if params.get('popular'):
            popular_param = params['popular']
            if isinstance(popular_param, (int, float)):
                # Filter by exact popularity rating
                filtered = filtered[filtered['popular'] == popular_param]
                print(f"DEBUG: Filtered by popularity = {popular_param}, found {len(filtered)} movies")
            elif popular_param == 'high':
                # Filter for high popularity (4-5)
                filtered = filtered[filtered['popular'] >= 4]
                filtered = filtered.sort_values('popular', ascending=False)
            elif popular_param == 'medium':
                # Filter for medium popularity (2-3)
                filtered = filtered[filtered['popular'].between(2, 3)]
            elif popular_param == 'low':
                # Filter for low popularity (1-2)
                filtered = filtered[filtered['popular'] <= 2]
        
        # Smart sorting logic based on search type
        if not params.get('popular') or params.get('popular') not in ['high', 'medium', 'low']:
            if is_specific_search:
                # For specific searches: prioritize relevance (keyword matches) over everything else
                if 'keyword_score' in filtered.columns:
                    filtered = filtered.sort_values(['keyword_score', 'popular', 'released'], ascending=[False, False, False])
                    print(f"DEBUG: SPECIFIC SEARCH - Sorted by relevance. Top movie: {filtered.iloc[0]['name'] if not filtered.empty else 'None'} (score: {filtered.iloc[0]['keyword_score'] if not filtered.empty else 0})")
                else:
                    # Even without keyword scores, don't randomize specific searches
                    filtered = filtered.sort_values(['popular', 'released'], ascending=[False, False])
            # For general searches: Sort by a combination of popularity and recency with randomness
            elif 'popular' in filtered.columns and 'released' in filtered.columns:
                # Smart sorting with weighted combination
                filtered['combined_score'] = (0.7 * filtered['popular']) + (0.3 * (filtered['released'] - 2000) / 24)
                filtered = filtered.sort_values('combined_score', ascending=False)
        
        # Debug: Print final results summary
        if not filtered.empty:
            print(f"DEBUG: Final results summary:")
            for i, (_, movie) in enumerate(filtered.head(6).iterrows()):
                print(f"  {i+1}. {movie['name']} - Popularity: {movie['popular']}")
        
        return filtered.head(6)  # Return top 6 results

    def get_recommendation(self, user_query):
        """Main method to get movie recommendations based on user query."""
        try:
            # Check for off-topic queries first
            off_topic_keywords = ['weather', 'politics', 'cooking', 'sports', 'news', 'health', 'travel']
            if any(keyword in user_query.lower() for keyword in off_topic_keywords):
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about movie recommendations, actors, directors, or anything related to films!"
            
            # Extract parameters from query
            params = self.extract_query_parameters(user_query)
            
            # Handle off-topic intent
            if params.get('intent') == 'off_topic':
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about movie recommendations, actors, directors, or anything related to films!"
            
            # Filter movies based on parameters
            filtered_movies = self.filter_movies(params)
            
            # Generate response
            if not filtered_movies.empty:
                return self.generate_response(filtered_movies, params, user_query)
            else:
                return self.generate_no_results_response(params)
                
        except Exception as e:
            print(f"Error in get_recommendation: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a different query."

    def generate_response(self, filtered_movies, params, original_query):
        """Generate a natural language response using Gemini."""
        try:
            if self.model and not filtered_movies.empty:
                # Prepare movie data for Gemini
                movies_info = []
                for _, movie in filtered_movies.iterrows():
                    movies_info.append({
                        'name': movie['name'],
                        'year': int(movie['released']) if pd.notna(movie['released']) else 'Unknown',
                        'genre': movie['genre'] if pd.notna(movie['genre']) else 'Unknown',
                        'popularity': int(movie['popular']) if pd.notna(movie['popular']) else 'Unknown'
                    })
                
                print(f"DEBUG: Movies being sent to Gemini for response generation:")
                for i, movie_info in enumerate(movies_info):
                    print(f"  {i+1}. {movie_info['name']} - Popularity: {movie_info['popularity']}")
                
                # Create prompt for response generation
                movies_text = ""
                for movie_info in movies_info:
                    movies_text += f"• {movie_info['name']} ({movie_info['year']}) - {movie_info['genre']}\n"
                
                prompt = f"""Based on the user's query: "{original_query}"

Here are the most relevant movies from our database:
{movies_text}

Generate a helpful response in English. Start with a brief introduction, then list the movies with their details. Keep it conversational and informative."""
                
                response = self.model.generate_content(prompt)
                gemini_response = response.text.strip()
                
                print(f"DEBUG: Gemini's actual response: {gemini_response}")
                return gemini_response
            else:
                return self.generate_fallback_response(filtered_movies, params)
                
        except Exception as e:
            print(f"Error generating response with Gemini: {str(e)}")
            return self.generate_fallback_response(filtered_movies, params)
    
    def generate_no_results_response(self, params):
        """Generate response when no movies match the criteria."""
        return "I couldn't find any movies matching your specific criteria. Try broadening your search or asking for different genres, years, or actors."
    
    def generate_fallback_response(self, filtered_movies, params):
        """Generate a basic response without AI."""
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Please try a different search."
        
        response = "Here are some movie recommendations for you:\n\n"
        for _, movie in filtered_movies.iterrows():
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown genre'
            response += f"• {movie['name']} ({year}) - {genre}\n"
        
        return response

# Initialize the recommender
recommender = MovieRecommender("MergeAndCleaned_Movies.csv")

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
        
        # Get conversation context
        context = get_conversation_context(user_id)
        
        # Enhance query with context if available
        enhanced_query = query
        if context:
            enhanced_query = f"Previous context: {context}\n\nCurrent question: {query}"
        
        # Regular movie recommendation
        recommendation = recommender.get_recommendation(enhanced_query)
        
        # Save conversation to memory
        save_conversation(user_id, query, recommendation)
        
        return jsonify({'recommendation': recommendation})
        
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)