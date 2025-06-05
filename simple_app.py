from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import json
import os
import google.generativeai as genai
import re
import uuid
from watchlist_db import WatchlistDB

app = Flask(__name__, static_folder='.', template_folder='.')
app.secret_key = 'movie_watchlist_secret_key_2024'

# Initialize watchlist database
watchlist_db = WatchlistDB()

class MovieRecommender:
    def __init__(self, csv_file_path):
        """Initialize the movie recommender with CSV data and Gemini client."""
        self.movies = self.load_movies(csv_file_path)
        # Configure Gemini AI
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
        
    def load_movies(self, csv_file_path):
        """Load and validate movie data from CSV."""
        try:
            # Try different encodings to handle special characters
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not read CSV file with any encoding")
            
            # Clean and validate data
            df = df.dropna(subset=['name'])
            df['released'] = pd.to_numeric(df['released'], errors='coerce')
            df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
            df['popular'] = pd.to_numeric(df['popular'], errors='coerce')
            df['popular'] = df['popular'].fillna(1)
            
            # Ensure all text columns are strings
            df['name'] = df['name'].astype(str)
            df['genre'] = df['genre'].astype(str).fillna('')
            df['age_group'] = df['age_group'].astype(str).fillna('')
            
            print(f"Successfully loaded {len(df)} movies")
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def extract_query_parameters(self, user_query):
        """Use Gemini to extract parameters from natural language query."""
        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries.

Extract the following information from the user's query and return as JSON:
- age_group: target age group if mentioned (Kids, Teens, Young Adults, Adults, Unknown)
- genre: specific genre if mentioned (e.g., Horror, Action, Drama, Comedy)
- year_range: [min_year, max_year] if mentioned
- country: specific country if mentioned
- popular: if asking for popular/top movies (high, medium, low)
- actor: actor/actress name if mentioned
- director: director name if mentioned
- intent: the main intent (recommend, check_suitability, filter, general)

Note: Age groups are: Kids (up to 7), Teens (8-13), Young Adults (14-17), Adults (18+)

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
        
        # Enhanced actor detection - look for capitalized names
        import re
        # Look for patterns like "Artiwara Kongmalai" (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        potential_names = re.findall(name_pattern, query)
        
        # Also check for specific actor-related keywords
        actor_indicators = ['actor', 'actress', 'starring', 'cast', 'with', 'movies by', 'films with']
        for indicator in actor_indicators:
            if indicator in query_lower:
                # Extract text after the indicator
                idx = query_lower.find(indicator)
                after_indicator = query[idx + len(indicator):].strip()
                # Look for capitalized names in the text after indicator
                names_after = re.findall(name_pattern, after_indicator)
                potential_names.extend(names_after)
        
        # If we found potential actor names, use the first one
        if potential_names:
            # Check if this might be a director query instead
            director_indicators = ['director', 'directed', 'filme by', 'made by']
            is_director_query = any(indicator in query_lower for indicator in director_indicators)
            
            if is_director_query:
                params['director'] = potential_names[0]
            else:
                params['actor'] = potential_names[0]
        
        return params
    
    def filter_movies(self, params):
        """Filter movies based on extracted parameters."""
        filtered = self.movies.copy()
        
        # Age group filtering - handle both exact and partial matches
        if params.get('age_group'):
            age_group = params['age_group']
            # Try exact match first
            exact_match = filtered[filtered['age_group'].str.lower() == age_group.lower()]
            if not exact_match.empty:
                filtered = exact_match
            else:
                # Fallback to partial match
                age_mask = filtered['age_group'].str.contains(age_group, case=False, na=False)
                filtered = filtered[age_mask]
        
        # Genre filtering with better mapping
        if params.get('genre'):
            genre = params['genre'].lower()
            # Map common genre requests to actual genre names in data
            genre_mappings = {
                'comedy': 'Comedies',
                'horror': 'Horror Movies', 
                'action': 'Action & Adventure',
                'drama': 'Dramas',
                'romance': 'Romantic Movies',
                'sci-fi': 'Sci-Fi & Fantasy',
                'thriller': 'Thrillers',
                'documentary': 'Documentaries',
                'kids': 'Children & Family Movies'
            }
            
            search_genre = genre_mappings.get(genre, params['genre'])
            genre_mask = filtered['genre'].str.contains(search_genre, case=False, na=False)
            filtered = filtered[genre_mask]
        
        # Year range filtering
        if params.get('year_range'):
            min_year, max_year = params['year_range']
            filtered = filtered[
                (filtered['released'] >= min_year) & 
                (filtered['released'] <= max_year)
            ]
        
        # Country filtering
        if params.get('country'):
            country_mask = filtered['country'].str.contains(params['country'], case=False, na=False)
            filtered = filtered[country_mask]
        
        # Actor filtering - search in cast column
        if params.get('actor'):
            actor_name = params['actor']
            print(f"DEBUG: Searching for actor: '{actor_name}'")
            
            # Try multiple search strategies
            actor_mask = filtered['cast'].str.contains(actor_name, case=False, na=False)
            actor_results = filtered[actor_mask]
            
            # If no results, try searching for individual parts of the name
            if actor_results.empty and ' ' in actor_name:
                name_parts = actor_name.split()
                for part in name_parts:
                    if len(part) > 2:  # Only search meaningful name parts
                        part_mask = filtered['cast'].str.contains(part, case=False, na=False)
                        part_results = filtered[part_mask]
                        if not part_results.empty:
                            actor_results = part_results
                            print(f"DEBUG: Found movies using name part '{part}'")
                            break
            
            filtered = actor_results
            print(f"DEBUG: Found {len(filtered)} movies with actor search")
        
        # Director filtering
        if params.get('director'):
            director_name = params['director']
            print(f"DEBUG: Searching for director: '{director_name}'")
            
            # Try multiple search strategies like we do for actors
            director_mask = filtered['director'].str.contains(director_name, case=False, na=False)
            director_results = filtered[director_mask]
            
            # If no results, try searching for individual parts of the name
            if director_results.empty and ' ' in director_name:
                name_parts = director_name.split()
                for part in name_parts:
                    if len(part) > 2:  # Only search meaningful name parts
                        part_mask = filtered['director'].str.contains(part, case=False, na=False)
                        part_results = filtered[part_mask]
                        if not part_results.empty:
                            director_results = part_results
                            print(f"DEBUG: Found movies using director name part '{part}'")
                            break
            
            filtered = director_results
            print(f"DEBUG: Found {len(filtered)} movies with director search")
        
        # Ensure we have results before sorting
        if filtered.empty:
            return filtered
            
        # Add randomization to avoid always showing the same movies
        import random
        random.seed()  # Use current time as seed for true randomization
        filtered = filtered.sample(frac=1).reset_index(drop=True)
        
        # Smart sorting logic
        if params.get('popular') == 'high':
            # Sort by popularity only for explicit popular requests
            filtered = filtered.sort_values('popular', ascending=False)
        else:
            # Sort by a combination of popularity and recency with some randomness
            if 'popular' in filtered.columns and 'released' in filtered.columns:
                # Normalize released year (focus on 2000+ movies)
                max_year = filtered['released'].max()
                min_year = max(filtered['released'].min(), 2000)
                
                if max_year > min_year:
                    filtered['recency_score'] = (filtered['released'] - min_year) / (max_year - min_year)
                else:
                    filtered['recency_score'] = 1
                
                # Normalize popularity (1-5 → 0-1)
                filtered['pop_score'] = (filtered['popular'] - 1) / 4
                
                # Combined smart score with more weight on popularity
                filtered['smart_score'] = (filtered['pop_score'] * 0.7) + (filtered['recency_score'] * 0.3)
                filtered = filtered.sort_values('smart_score', ascending=False)
        
        return filtered.head(6)
    
    def generate_response(self, filtered_movies, params, original_query):
        """Generate a natural language response using Gemini."""
        if filtered_movies.empty:
            return self.generate_alternative_suggestions(params, original_query)
        
        # Prepare movie data for the prompt
        movie_list = []
        for _, movie in filtered_movies.iterrows():
            movie_info = {
                'name': movie['name'],
                'genre': movie['genre'],
                'released': int(movie['released']) if pd.notna(movie['released']) else 'Unknown',
                'popular': int(movie['popular']) if pd.notna(movie['popular']) else 0,
                'age_group': movie['age_group'] if pd.notna(movie['age_group']) else 'N/A',
                'runtime': int(movie['runtime']) if pd.notna(movie['runtime']) else 'N/A',
                'country': movie['country'] if pd.notna(movie['country']) else 'N/A',
                'description': movie['description'] if pd.notna(movie['description']) else 'No description available'
            }
            movie_list.append(movie_info)
        
        system_prompt = f"""You are a helpful movie recommendation assistant. Respond in English.

The user asked: "{original_query}"

ALWAYS format your response as a clean list:
1. One brief intro sentence
2. Then list each movie exactly like this:
   • Movie Name (Year) - Genre
   • Movie Name (Year) - Genre
   • Movie Name (Year) - Genre

Use bullet points (•) for each movie. Keep it simple - just name, year, and genre. No descriptions or extra text."""

        try:
            if self.model:
                prompt = f"{system_prompt}\n\nHere are the movies I found: {json.dumps(movie_list, ensure_ascii=False)}"
                response = self.model.generate_content(prompt)
                return response.text
            else:
                return self.generate_fallback_response(filtered_movies, params)
            
        except Exception as e:
            return self.generate_fallback_response(filtered_movies, params)
    
    def generate_alternative_suggestions(self, params, original_query):
        """Generate alternative suggestions when no exact matches are found."""
        # Try to find alternatives by relaxing constraints
        alternatives = []
        
        # If looking for specific age group + genre, try just the genre
        if params.get('age_group') and params.get('genre'):
            genre_only = {'genre': params['genre']}
            alt_movies = self.filter_movies(genre_only)
            if not alt_movies.empty:
                alternatives.append(f"I found {params['genre']} movies for other age groups")
        
        # If looking for specific genre + age, try just the age group
        elif params.get('genre') and params.get('age_group'):
            age_only = {'age_group': params['age_group']}
            alt_movies = self.filter_movies(age_only)
            if not alt_movies.empty:
                alternatives.append(f"I found movies suitable for {params['age_group']} in other genres")
        
        # If looking for specific genre, suggest similar genres
        elif params.get('genre'):
            genre = params['genre'].lower()
            similar_genres = {
                'horror': ['thriller', 'drama'],
                'comedy': ['family', 'romance'],
                'action': ['adventure', 'thriller'],
                'drama': ['thriller', 'romance'],
                'family': ['comedy', 'animation']
            }
            
            for similar in similar_genres.get(genre, []):
                similar_params = params.copy()
                similar_params['genre'] = similar
                alt_movies = self.filter_movies(similar_params)
                if not alt_movies.empty:
                    alternatives.append(f"I found {similar} movies that might interest you")
                    break
        
        # Generate response with alternatives
        if alternatives:
            alt_params = params.copy()
            # Remove one constraint to find alternatives
            if 'age_group' in alt_params and 'genre' in alt_params:
                del alt_params['age_group']  # Keep genre, remove age restriction
            
            alt_movies = self.filter_movies(alt_params)
            if not alt_movies.empty:
                # Create alternative response with smart explanation
                movie_list = []
                for _, movie in alt_movies.iterrows():
                    movie_info = {
                        'name': movie['name'],
                        'genre': movie['genre'],
                        'released': int(movie['released']) if pd.notna(movie['released']) else 'Unknown',
                        'popular': int(movie['popular']) if pd.notna(movie['popular']) else 0,
                        'age_group': movie['age_group'] if pd.notna(movie['age_group']) else 'N/A',
                        'runtime': int(movie['runtime']) if pd.notna(movie['runtime']) else 'N/A',
                        'country': movie['country'] if pd.notna(movie['country']) else 'N/A',
                        'description': movie['description'] if pd.notna(movie['description']) else 'No description available'
                    }
                    movie_list.append(movie_info)
                
                # Use Gemini to create smart alternative response
                if self.model:
                    try:
                        alt_prompt = f"""The user asked: "{original_query}"

I don't have exact matches for this request, but I found related alternatives. Create a short, clear response that:
1. First says clearly "I don't have [specific request], but here are some alternatives:"
2. Lists 3-4 movies maximum in this simple format:
   - Movie Name (Year) - Brief reason why it's relevant
3. Keep it short and conversational

Movies available: {json.dumps(movie_list[:4], ensure_ascii=False)}

Keep the response under 150 words."""
                        
                        response = self.model.generate_content(alt_prompt)
                        return response.text
                    except:
                        pass
        
        # If still no alternatives, suggest popular movies
        popular_movies = self.movies.head(5)
        return self.generate_fallback_response(popular_movies, params)
    
    def generate_fallback_response(self, filtered_movies, params):
        """Generate a basic response without AI."""
        response = "Here are some recommendations:\n\n"
        count = 0
        for _, movie in filtered_movies.iterrows():
            if count >= 6:  # Limit to 6 movies
                break
            response += f"• {movie['name']} ({movie['released']}) - {movie['genre']}\n"
            count += 1
        
        return response
    
    def get_recommendation(self, user_query):
        """Main method to get movie recommendations based on user query."""
        try:
            # Extract parameters from user query
            params = self.extract_query_parameters(user_query)
            
            # Filter movies based on parameters
            filtered_movies = self.filter_movies(params)
            
            # Generate natural language response
            response = self.generate_response(filtered_movies, params, user_query)
            
            return response
            
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a different query."

# Initialize the recommender
recommender = MovieRecommender("movies_clean_utf8.csv")

@app.route('/')
def index():
    return render_template('index.html')

def get_user_id():
    """Get or create user session ID"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        # Create a simple username for the user
        username = f"user_{session['user_id'][:8]}"
        watchlist_db.add_user(session['user_id'], username)
    return session['user_id']

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Check for watchlist-related queries
        query_lower = query.lower()
        user_id = get_user_id()
        
        if 'my watchlist' in query_lower or 'show watchlist' in query_lower:
            watchlist = watchlist_db.get_watchlist(user_id)
            if not watchlist:
                return jsonify({'recommendation': 'Your watchlist is empty. Start adding movies by saying "add [movie name] to watchlist"!'})
            
            response = "📋 Your Watchlist:\n\n"
            for movie in watchlist:
                status_emoji = {"want_to_watch": "⏳", "watched": "✅", "watching": "🎬"}.get(movie['status'], "⏳")
                response += f"{status_emoji} {movie['title']} ({movie['year'] or 'Unknown'}) - {movie['genre'] or 'Unknown genre'}\n"
            
            return jsonify({'recommendation': response})
        
        elif 'add to watchlist' in query_lower or 'save to watchlist' in query_lower:
            # Extract movie name from query
            import re
            match = re.search(r'add\s+"?([^"]+?)"?\s+to\s+watchlist', query_lower)
            if not match:
                match = re.search(r'save\s+"?([^"]+?)"?\s+to\s+watchlist', query_lower)
            
            if match:
                movie_title = match.group(1).strip()
                # Try to find the movie in our database
                movie_found = recommender.movies[recommender.movies['name'].str.lower().str.contains(movie_title.lower(), na=False)]
                
                if not movie_found.empty:
                    movie = movie_found.iloc[0]
                    success = watchlist_db.add_to_watchlist(
                        user_id, 
                        movie['name'], 
                        movie['released'] if pd.notna(movie['released']) else None,
                        movie['genre'] if pd.notna(movie['genre']) else None
                    )
                    
                    if success:
                        return jsonify({'recommendation': f'✅ Added "{movie["name"]}" to your watchlist!'})
                    else:
                        return jsonify({'recommendation': f'"{movie["name"]}" is already in your watchlist.'})
                else:
                    return jsonify({'recommendation': f'Sorry, I couldn\'t find "{movie_title}" in our movie database.'})
            else:
                return jsonify({'recommendation': 'Please specify which movie to add. Example: "add Inception to watchlist"'})
        
        # Regular movie recommendation
        recommendation = recommender.get_recommendation(query)
        return jsonify({'recommendation': recommendation})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/watchlist', methods=['GET'])
def get_watchlist():
    """Get user's complete watchlist"""
    try:
        user_id = get_user_id()
        watchlist = watchlist_db.get_watchlist(user_id)
        return jsonify({'watchlist': watchlist})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/watchlist/add', methods=['POST'])
def add_to_watchlist():
    """Add movie to watchlist"""
    try:
        data = request.get_json()
        movie_title = data.get('title')
        movie_year = data.get('year')
        genre = data.get('genre')
        
        if not movie_title:
            return jsonify({'error': 'Movie title is required'}), 400
        
        user_id = get_user_id()
        success = watchlist_db.add_to_watchlist(user_id, movie_title, movie_year, genre)
        
        if success:
            return jsonify({'message': 'Movie added to watchlist successfully'})
        else:
            return jsonify({'error': 'Movie already in watchlist'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/watchlist/update', methods=['POST'])
def update_watchlist():
    """Update movie status in watchlist"""
    try:
        data = request.get_json()
        movie_title = data.get('title')
        movie_year = data.get('year')
        status = data.get('status')  # want_to_watch, watching, watched
        rating = data.get('rating')
        notes = data.get('notes')
        
        if not movie_title or not status:
            return jsonify({'error': 'Movie title and status are required'}), 400
        
        user_id = get_user_id()
        success = watchlist_db.update_movie_status(user_id, movie_title, movie_year, status, rating, notes)
        
        if success:
            return jsonify({'message': 'Movie status updated successfully'})
        else:
            return jsonify({'error': 'Movie not found in watchlist'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/watchlist/remove', methods=['POST'])
def remove_from_watchlist():
    """Remove movie from watchlist"""
    try:
        data = request.get_json()
        movie_title = data.get('title')
        movie_year = data.get('year')
        
        if not movie_title:
            return jsonify({'error': 'Movie title is required'}), 400
        
        user_id = get_user_id()
        success = watchlist_db.remove_from_watchlist(user_id, movie_title, movie_year)
        
        if success:
            return jsonify({'message': 'Movie removed from watchlist successfully'})
        else:
            return jsonify({'error': 'Movie not found in watchlist'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)