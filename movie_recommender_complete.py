from flask import Flask, render_template, request, jsonify, session, render_template_string
import pandas as pd
import json
import os
import google.generativeai as genai
import re
import uuid
from datetime import datetime
from difflib import SequenceMatcher
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Global variables
app = Flask(__name__, static_folder='.', template_folder='templates')
app.secret_key = 'movie_recommender_secret_key_2024'
conversation_memory = {}
recommender = None

# Utility functions
def fuzzy_match(word, target_words, threshold=0.7):
    """Check if word matches any target word with fuzzy matching."""
    if not word or not target_words:
        return False
    
    word_lower = word.lower().strip()
    for target in target_words:
        if not target:
            continue
        target_lower = target.lower().strip()
        similarity = SequenceMatcher(None, word_lower, target_lower).ratio()
        if similarity >= threshold:
            return True
    return False

def extract_runtime_from_text(text):
    """Extract runtime in minutes from text."""
    text = text.lower()
    
    # Pattern for "X hours Y minutes" or "X hours and Y minutes"
    hours_minutes_pattern = r'(\d+)\s*(?:hours?|hrs?)\s*(?:and\s*)?(\d+)\s*(?:minutes?|mins?)'
    match = re.search(hours_minutes_pattern, text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours * 60 + minutes
    
    # Pattern for just hours
    hours_pattern = r'(\d+)\s*(?:hours?|hrs?)'
    match = re.search(hours_pattern, text)
    if match:
        return int(match.group(1)) * 60
    
    # Pattern for just minutes
    minutes_pattern = r'(\d+)\s*(?:minutes?|mins?)'
    match = re.search(minutes_pattern, text)
    if match:
        return int(match.group(1))
    
    # Special cases
    if 'hour and half' in text or 'hour and a half' in text:
        return 90
    if 'two hours' in text:
        return 120
    
    return None

def format_runtime_display(runtime_minutes):
    """Format runtime for display."""
    if pd.isna(runtime_minutes) or runtime_minutes <= 0:
        return ""
    
    runtime_minutes = int(runtime_minutes)
    hours = runtime_minutes // 60
    minutes = runtime_minutes % 60
    
    if hours > 0 and minutes > 0:
        return f" • {hours}h {minutes}m"
    elif hours > 0:
        return f" • {hours}h"
    else:
        return f" • {minutes}m"

def clean_movie_data(movies_df):
    """Clean and prepare movie data."""
    # Fill missing values
    movies_df['genre'] = movies_df['genre'].fillna('Unknown')
    movies_df['country'] = movies_df['country'].fillna('Unknown')
    movies_df['age_group'] = movies_df['age_group'].fillna('General')
    movies_df['popular'] = movies_df['popular'].fillna(3.0)
    movies_df['released'] = movies_df['released'].fillna(2000)
    movies_df['runtime'] = movies_df['runtime'].fillna(100)
    
    return movies_df

def is_off_topic_query(query):
    """Check if query is off-topic (not about movies)."""
    off_topic_keywords = [
        'weather', 'politics', 'cooking', 'sports', 'news', 
        'health', 'travel', 'music', 'books', 'recipes'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in off_topic_keywords)

def setup_movie_clustering(movies_df):
    """Setup K-Means clustering for movie recommendations."""
    try:
        print("Setting up K-Means clustering...")
        
        # Encode categorical features
        le_genre = LabelEncoder()
        le_country = LabelEncoder()
        le_age = LabelEncoder()
        
        # Create feature matrix
        movies_df['genre_encoded'] = le_genre.fit_transform(movies_df['genre'].fillna('Unknown'))
        movies_df['country_encoded'] = le_country.fit_transform(movies_df['country'].fillna('Unknown'))
        movies_df['age_group_encoded'] = le_age.fit_transform(movies_df['age_group'].fillna('General'))
        
        feature_matrix = np.column_stack([
            movies_df['genre_encoded'].values,
            movies_df['country_encoded'].values,
            movies_df['age_group_encoded'].values,
            movies_df['released'].fillna(2000).values,
            movies_df['popular'].fillna(3.0).values,
            movies_df['runtime'].fillna(100).values
        ])
        
        # Scale and cluster
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        n_clusters = min(15, len(movies_df) // 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        movies_df['cluster'] = kmeans.fit_predict(feature_matrix_scaled)
        
        print(f"Clustering completed with {n_clusters} clusters")
        return True
        
    except Exception as e:
        print(f"Clustering setup failed: {e}")
        movies_df['cluster'] = 0
        return False

def get_cluster_recommendations(movies_df, filtered_movies, limit=6):
    """Enhance recommendations using cluster information."""
    try:
        if 'cluster' not in movies_df.columns:
            return filtered_movies.head(limit)
        
        # Get cluster distribution from filtered movies
        cluster_counts = filtered_movies['cluster'].value_counts()
        
        # Get diverse recommendations from top clusters
        recommendations = []
        for cluster_id in cluster_counts.index[:3]:  # Top 3 clusters
            cluster_movies = filtered_movies[filtered_movies['cluster'] == cluster_id]
            recommendations.append(cluster_movies.head(2))
        
        # Combine and return
        if recommendations:
            result = pd.concat(recommendations).drop_duplicates(subset=['name']).head(limit)
            return result if not result.empty else filtered_movies.head(limit)
        else:
            return filtered_movies.head(limit)
            
    except Exception as e:
        print(f"Cluster recommendation failed: {e}")
        return filtered_movies.head(limit)


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
            
            # Setup clustering
            setup_movie_clustering(self.movies)
            
        except Exception as e:
            print(f"Error initializing MovieRecommender: {str(e)}")
            raise Exception(f"Error initializing MovieRecommender: {str(e)}")

    def load_movies(self, csv_file_path):
        """Load and validate movie data from CSV."""
        try:
            # Try different encodings
            encodings = ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']
            movies = None
            
            for encoding in encodings:
                try:
                    movies = pd.read_csv(csv_file_path, encoding=encoding)
                    print(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if movies is None:
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
        try:
            if self.model:
                prompt = f"""Extract movie search parameters from this query. Return ONLY valid JSON.

Context from conversation: {conversation_context}
User query: "{user_query}"

Extract these parameters:
- age_group: ONLY if explicitly mentioned (Kids, Teens, Young Adults, Adults) - if not mentioned, use null
- genre: specific genre ONLY if explicitly mentioned - be extremely flexible with spelling variations - if not mentioned, use null
- year_range: [min_year, max_year] ONLY if years are explicitly mentioned - if not mentioned, use null
- country: specific country ONLY if explicitly mentioned - if not mentioned, use null
- popular: ONLY if user explicitly asks for popular/top movies (high, medium, low) OR specific rating (1, 2, 3, 4, 5) - if not mentioned, use null
- actor: actor/actress name ONLY if explicitly mentioned - if not mentioned, use null
- director: director name ONLY if explicitly mentioned - if not mentioned, use null
- runtime: ONLY if duration is explicitly mentioned, convert to minutes (e.g., "two hours" = 120, "90 minutes" = 90, "hour and half" = 90) - if not mentioned, use null
- runtime_operator: ONLY if runtime comparison is mentioned: "greater_than", "less_than", "equal_to" or "between" - if not mentioned, use null
- description_keywords: array of keywords describing plot/story elements (e.g., for "movie about a missing doctor" extract ["missing", "doctor"]) - if no plot description, use null
- intent: the main intent (recommend, check_suitability, filter, general)

Return only the JSON object."""

                response = self.model.generate_content(prompt)
                result = json.loads(response.text.strip())
                
                # Age detection for teens - handle numeric ages
                if 'teen' in user_query.lower() or any(age in user_query for age in ['13', '14', '15', '16', '17', '18']):
                    result['age_group'] = 'Teens'
                
                return result
            else:
                return self.basic_parameter_extraction(user_query, conversation_context)
        except Exception as e:
            print(f"Error with Gemini extraction: {e}")
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

        query_lower = query.lower()

        # Age group detection
        if any(word in query_lower for word in ['kid', 'child', 'children']):
            params['age_group'] = 'Kids'
        elif any(word in query_lower for word in ['teen', 'teenager', 'adolescent']):
            params['age_group'] = 'Teens'
        elif any(word in query_lower for word in ['young adult', 'youth']):
            params['age_group'] = 'Young Adults'

        # Genre detection
        genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller', 'sci-fi', 'fantasy', 'documentary']
        for genre in genres:
            if genre in query_lower:
                params['genre'] = genre.title()
                break

        # Year detection
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        if len(years) == 1:
            year = int(years[0])
            params['year_range'] = [year, year]
        elif len(years) == 2:
            params['year_range'] = [int(min(years)), int(max(years))]

        # Country detection (basic)
        countries = ['usa', 'america', 'uk', 'france', 'germany', 'japan', 'korea', 'india', 'china']
        for country in countries:
            if country in query_lower:
                params['country'] = country.title()
                break

        # Quick popularity detection
        if any(w in query_lower for w in ['popular', 'top', 'best']):
            params['popular'] = 'high'

        return params

    def is_followup_query(self, query, context):
        """Use Gemini to determine if query is a follow-up or new topic."""
        if not context.strip():
            return False

        if not self.model:
            # Simple fallback if no AI available
            # Enhanced fallback for follow-up detection when AI is unavailable
            query_lower = query.lower()
            
            # Age appropriateness questions
            age_questions = ['is it for', 'suitable for', 'appropriate for', 'good for', 'ok for']
            if any(phrase in query_lower for phrase in age_questions):
                return True
            
            # Filtering/refinement phrases  
            filter_phrases = ['only', 'just', 'but', 'except', 'without', 'more', 'other', 'different']
            if any(word in query_lower for word in filter_phrases):
                return True
                
            # Recommendation requests about existing results
            recommendation_phrases = ['which one', 'what do you recommend', 'recommend one', 'pick one', 'which do you']
            if any(phrase in query_lower for phrase in recommendation_phrases):
                return True
            
            # Short queries are likely follow-ups
            if len(query.split()) <= 3:
                return True
            
            return False

        try:
            prompt = f"""Is this a follow-up question to the previous conversation or a completely new topic?

Previous context: {context[:200]}...
Current query: "{query}"

Answer with just "FOLLOWUP" or "NEW_TOPIC".

FOLLOWUP examples:
- "only from 2020" (after showing movies)
- "are they suitable for kids?" (asking about previous results)
- "what about horror movies?" (refinement)
- "which one you recommend?" (asking about previous list)

NEW_TOPIC examples:
- "show me action movies" (completely new request)
- "I want romantic films" (new genre request)"""

            response = self.model.generate_content(prompt)
            return "FOLLOWUP" in response.text.upper()
        except:
            return False

    def extract_context_parameters(self, context):
        """Extract relevant parameters from conversation context."""
        try:
            # Extract the most recent query from context
            lines = context.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('User:'):
                    recent_query = line.replace('User:', '').strip()
                    return self.extract_query_parameters(recent_query)
            return {}
        except:
            return {}

    def filter_movies(self, params):
        """Filter movies based on extracted parameters."""
        filtered = self.movies.copy()
        is_specific_search = bool(params.get('description_keywords'))
        print(f"DEBUG: Starting with {len(filtered)} movies")
        print(f"DEBUG: Parameters: {params}")

        # Genre filtering with mapping to actual CSV genres
        if params.get('genre') and params['genre'] != 'Unknown':
            # Handle both string and list formats from Gemini
            genre_value = params['genre']
            if isinstance(genre_value, list) and len(genre_value) > 0:
                genre = genre_value[0].lower()
            elif isinstance(genre_value, str):
                genre = genre_value.lower()
            else:
                genre = None

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
                        # Check if exact search works
                        exact_match = self.movies[self.movies['description'].str.contains("When a doctor goes missing, his psychiatrist wife treats", case=False, na=False)]
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
                filtered = filtered.sort_values(['keyword_score', 'popular', 'released'],
                                                ascending=[False, False, False])
            else:
                filtered = filtered.sort_values(['popular', 'released'], ascending=[False, False])
        else:
            filtered['combined_score'] = (0.7 * filtered['popular']) + (0.3 * (filtered['released'] - 2000) / 24)
            filtered = filtered.sort_values('combined_score', ascending=False)

        # Remove duplicates before returning results
        filtered = filtered.drop_duplicates(subset=['name'], keep='first')
        
        # Apply K-Means clustering for better recommendations
        try:
            result = get_cluster_recommendations(self.movies, filtered, 6)
        except:
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
                return "I'm sorry, but I specialize only in the world of movies. Ask me anything about movies and I'll be happy to help!"

            print(f"DEBUG: Processing query: {user_query}")
            print(f"DEBUG: Conversation context: {conversation_context[:100]}...")

            # Determine if this is a follow-up query
            is_followup = self.is_followup_query(user_query, conversation_context)
            
            if is_followup:
                print("DEBUG: Gemini detected FOLLOWUP")
                # For follow-up queries, combine context with current query
                context_params = self.extract_context_parameters(conversation_context)
                current_params = self.extract_query_parameters(user_query, conversation_context)
                
                # Merge parameters, giving priority to current query
                merged_params = context_params.copy()
                for key, value in current_params.items():
                    if value is not None:
                        merged_params[key] = value
                
                # Debug output for inherited parameters
                for key, value in context_params.items():
                    if value is not None and key in merged_params and merged_params[key] == value and current_params.get(key) is None:
                        if key == 'age_group' and 'teen' in conversation_context.lower():
                            print(f"DEBUG: Detected teen age (15 year), setting to Teens")
                            merged_params['age_group'] = 'Teens'
                        print(f"DEBUG: Inherited {key}='{value}' from context query: {user_query}")
                
                params = merged_params
                print(f"DEBUG: Context was provided - checking if parameters inherited correctly")
            else:
                print("DEBUG: Gemini detected NEW_TOPIC")
                # For new queries, extract fresh parameters
                params = self.extract_query_parameters(user_query, conversation_context)

            # Filter movies based on parameters
            filtered_movies = self.filter_movies(params)

            # Handle empty results with suggestions
            if filtered_movies.empty:
                return self.suggest_alternatives(conversation_context, user_query)

            # Generate natural language response
            return self.generate_response(filtered_movies, params, user_query)

        except Exception as e:
            print(f"Error in get_recommendation: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question."

    def suggest_alternatives(self, conversation_context, user_query):
        """Suggest alternative genres when user doesn't like previous recommendations."""
        # Try to extract the genre they didn't like from context
        previous_genre = None
        if 'genre' in conversation_context.lower():
            # Simple extraction - could be improved
            for genre in ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller']:
                if genre in conversation_context.lower():
                    previous_genre = genre.title()
                    break

        # Suggest alternative genres based on what they didn't like
        genre_alternatives = {
            'Action': ['Thriller', 'Adventure', 'Sci-Fi'],
            'Comedy': ['Romance', 'Family', 'Animation'],
            'Drama': ['Thriller', 'Biography', 'History'],
            'Horror': ['Thriller', 'Mystery', 'Sci-Fi'],
            'Thriller': ['Action', 'Crime', 'Mystery'],
            'Sci-Fi': ['Fantasy', 'Adventure', 'Action'],
            'Fantasy': ['Adventure', 'Family', 'Animation'],
            'Documentary': ['Biography', 'History', 'Crime'],
            'Animation': ['Family', 'Comedy', 'Adventure']
        }

        alternatives = genre_alternatives.get(previous_genre, ['Comedy', 'Drama', 'Action'])

        # Get sample movies from alternative genres
        response_text = f"I understand you didn't like the {previous_genre.lower() if previous_genre else 'previous'} recommendations. Let me suggest some alternatives:\n\n"

        for alt_genre in alternatives:
            # Get a few movies from this alternative genre
            alt_params = {'genre': alt_genre, 'intent': 'recommend'}
            alt_movies = self.filter_movies(alt_params)

            if not alt_movies.empty:
                top_movie = alt_movies.iloc[0]
                response_text += f"• **{alt_genre}**: Try \"{top_movie['title']}\" ({top_movie['release_year']})\n"

        response_text += "\nJust let me know which genre interests you, or ask for something completely different!"

        return response_text

    def generate_response(self, filtered_movies, params, original_query):
        """Generate a natural language response - either analysis or movie list."""
        # Check if this is an analytical question about the results
        if self.is_analytical_question(original_query):
            return self.generate_analytical_response(filtered_movies, original_query)
        else:
            # Basic movie list response
            return self.generate_fallback_response(filtered_movies, params)

    def is_analytical_question(self, query):
        """Check if the query is asking for analysis rather than recommendations using Gemini."""
        try:
            if self.model:
                prompt = f"""Is this query asking for analysis/conversation about existing results, or asking for new movie search?

Query: "{query}"

Respond with just "ANALYSIS" or "SEARCH".

ANALYSIS examples:
- "which one is the best?"
- "pick one for me"
- "are they suitable for kids?"
- "what about the ratings?"
- "tell me more about..."
- "which do you recommend?"
- "which one you recommend?"
- "what do you recommend?"
- "recommend one"
- "is it from the list?"
- "it is from the list you gave me?"
- "from the list you showed?"
- "from your recommendations?"

SEARCH examples:
- "show me action movies"
- "find comedy from 2020"
- "I want romantic films"
- "get me thriller movies"
"""
                response = self.model.generate_content(prompt)
                return "ANALYSIS" in response.text.upper()
            else:
                # Basic fallback using general patterns
                query_lower = query.lower()
                # Check for question words that typically indicate analysis
                question_patterns = ['which', 'what', 'how', 'are they', 'is it', 'tell me', 'pick', 'choose',
                                     'recommend', 'suggest', 'from the list', 'from your']
                # Check for analysis context words
                analysis_context = ['one', 'best', 'better', 'rating', 'suitable', 'good', 'about', 'list', 'gave me']

                has_question = any(pattern in query_lower for pattern in question_patterns)
                has_context = any(context in query_lower for context in analysis_context)

                return has_question and has_context
        except Exception as e:
            # Basic fallback
            query_lower = query.lower()
            return any(word in query_lower for word in ['which', 'pick', 'recommend', 'best', 'rating', 'suitable'])

    def generate_analytical_response(self, filtered_movies, query):
        """Generate analytical response using Gemini."""
        try:
            if self.model and not filtered_movies.empty:
                # Prepare movie data for analysis
                movies_data = ""
                for i, (_, movie) in enumerate(filtered_movies.head(6).iterrows(), 1):
                    runtime_display = format_runtime_display(movie.get('runtime', 0))
                    movies_data += f"{i}. {movie['name']} ({movie.get('released', 'N/A')}) - "
                    movies_data += f"Rating: {movie.get('popular', 'N/A')}/5, "
                    movies_data += f"Genre: {movie.get('genre', 'Unknown')}, "
                    movies_data += f"Age: {movie.get('age_group', 'General')}{runtime_display}\n"

                prompt = f"""You are a movie recommendation chatbot. The user is asking: "{query}"

Here are the movies from their previous search:
{movies_data}

CRITICAL: If user asks for a recommendation from the list (like "give me recommend from your list", "which one you recommend", "recommend from the list"), pick EXACTLY ONE movie from the list above and explain why it's the best choice. Do NOT generate a new search.

Give a SHORT, conversational response (2-3 sentences max). Examples:

User asks "which one you recommend?" → "I'd recommend 'Charming'. It's highly rated and family-friendly."
User asks "give me recommend from your list" → "I suggest 'Dad Wanted' - it's both funny and heartwarming."
User asks "recommend from the list" → Pick ONE specific movie and explain why
User asks "are they suitable for kids?" → Simple yes/no with quick reason
User asks about ratings → Brief comparison or explanation

Always pick ONE specific movie when asked for recommendations. Be friendly but CONCISE."""

                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return self.generate_basic_analytical_response(filtered_movies, query)
        except Exception as e:
            print(f"Error generating analytical response: {e}")
            return self.generate_basic_analytical_response(filtered_movies, query)

    def generate_basic_analytical_response(self, filtered_movies, query):
        """Generate basic analytical response without AI - simple fallback."""
        if filtered_movies.empty:
            return "I don't have any movies to analyze from your previous search."

        query_lower = query.lower()
        
        if any(word in query_lower for word in ['recommend', 'suggest', 'pick', 'choose', 'best']):
            # Pick the top-rated movie
            top_movie = filtered_movies.iloc[0]
            return f"I'd recommend '{top_movie['name']}' - it has a high rating and seems like a great choice!"
        
        elif any(word in query_lower for word in ['kid', 'child', 'family']):
            family_movies = filtered_movies[filtered_movies['age_group'].isin(['Kids', 'Family', 'General'])]
            if not family_movies.empty:
                return f"Yes, most of these are family-friendly. I'd especially recommend '{family_movies.iloc[0]['name']}'."
            else:
                return "These movies might be better suited for older audiences."
        
        elif 'rating' in query_lower:
            avg_rating = filtered_movies['popular'].mean()
            return f"The average rating of these movies is {avg_rating:.1f}/5. The highest rated is '{filtered_movies.iloc[0]['name']}'."
        
        else:
            return f"I found {len(filtered_movies)} movies matching your criteria. Would you like me to recommend one?"

    def generate_fallback_response(self, filtered_movies, params=None):
        """Generate a basic response without AI."""
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Try adjusting your preferences!"

        response_parts = []
        
        # Create context-aware introduction
        if params:
            criteria = []
            if params.get('genre'):
                criteria.append(f"{params['genre'].lower()}")
            if params.get('age_group'):
                criteria.append(f"for {params['age_group'].lower()}")
            if params.get('year_range'):
                if len(params['year_range']) == 2 and params['year_range'][0] == params['year_range'][1]:
                    criteria.append(f"from {params['year_range'][0]}")
                else:
                    criteria.append(f"from {params['year_range'][0]}-{params['year_range'][1]}")
            if params.get('country'):
                criteria.append(f"from {params['country']}")
            
            if criteria:
                intro = f"Here are some movie recommendations {' '.join(criteria)}:"
            else:
                intro = "Here are some movie recommendations:"
        else:
            intro = "Here are some movie recommendations:"
        
        response_parts.append(intro)

        # Add movie list
        for i, (_, movie) in enumerate(filtered_movies.head(6).iterrows(), 1):
            movie_line = f"• <span class='movie-title'>{movie['name']}</span>"
            
            if pd.notna(movie.get('released')):
                movie_line += f" <span class='movie-year'>({int(movie['released'])})</span>"
            
            if pd.notna(movie.get('genre')):
                movie_line += f" <span class='movie-genre'>{movie['genre']}</span>"
            
            if pd.notna(movie.get('age_group')):
                movie_line += f" <span class='movie-age'>{movie['age_group']}</span>"
            
            runtime_display = format_runtime_display(movie.get('runtime', 0))
            if runtime_display:
                movie_line += f" <span class='movie-runtime'>{runtime_display}</span>"
            
            response_parts.append(movie_line)

        return '\n'.join(response_parts)


def initialize_system():
    """Initialize the movie recommendation system."""
    try:
        csv_file_path = 'attached_assets/MergeAndCleaned_Movies.csv'
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        global recommender
        recommender = MovieRecommender(csv_file_path)
        print(f"Successfully loaded {len(recommender.movies)} movies")
        print("System initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize system: {str(e)}")
        return False


# Flask routes
@app.route('/')
def index():
    return render_template_string(open('index.html', 'r', encoding='utf-8').read())

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
        'timestamp': datetime.now(),
        'user': user_query,
        'assistant': response
    })
    
    # Keep only last 10 exchanges
    if len(conversation_memory[user_id]) > 10:
        conversation_memory[user_id] = conversation_memory[user_id][-10:]

def get_conversation_context(user_id):
    """Get recent conversation context for user"""
    if user_id not in conversation_memory:
        return ""
    
    context_parts = []
    for entry in conversation_memory[user_id][-5:]:  # Last 5 exchanges
        context_parts.append(f"User: {entry['user']}")
        context_parts.append(f"Assistant: {entry['assistant'][:100]}...")
    
    return "\n".join(context_parts)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

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
    app.run(host='0.0.0.0', port=5000, debug=True)

def main():
    """Main function to run the application."""
    print("=" * 50)
    print("Movie Recommendation Chatbot")
    print("=" * 50)
    
    print("Initializing Movie Recommendation System...")
    if initialize_system():
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        start_server()
    else:
        print("Failed to initialize the system. Please check the error messages above.")

if __name__ == "__main__":
    main()