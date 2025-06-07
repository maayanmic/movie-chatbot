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
        self.movies = self.load_movies(csv_file_path)
        # Configure Gemini AI
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Use the correct model name for Gemini 1.5
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini model: {e}")
                self.model = None
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
            print(f"DEBUG: CSV columns: {list(df.columns)}")
            
            # Debug specific movie data
            if 'Stuck Apart' in df['name'].values:
                stuck_apart_data = df[df['name'] == 'Stuck Apart']
                print(f"DEBUG: Stuck Apart data in CSV: {stuck_apart_data[['name', 'popular']].iloc[0].to_dict()}")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        except Exception as e:
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
        
        # Enhanced actor/director detection - look for capitalized names
        import re
        # Look for patterns like "Artiwara Kongmalai" (capitalized words) or single names like "Onir"
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, query)
        
        # Check for director-specific keywords first
        director_indicators = ['director', 'directed', 'filme by', 'made by']
        is_director_query = any(indicator in query_lower for indicator in director_indicators)
        
        # Also check for actor-related keywords
        actor_indicators = ['actor', 'actress', 'starring', 'cast', 'with', 'movies by', 'films with']
        
        # Extract names after specific keywords
        all_indicators = director_indicators + actor_indicators
        for indicator in all_indicators:
            if indicator in query_lower:
                # Extract text after the indicator
                idx = query_lower.find(indicator)
                after_indicator = query[idx + len(indicator):].strip()
                # Look for capitalized names in the text after indicator
                names_after = re.findall(name_pattern, after_indicator)
                potential_names.extend(names_after)
        
        # Filter out common words that aren't names
        common_words = ['The', 'And', 'Of', 'In', 'On', 'At', 'To', 'For', 'With', 'By', 'From', 'Movie', 'Film', 'Movies', 'Films']
        filtered_names = [name for name in potential_names if name not in common_words]
        
        # If we found potential names, use the most relevant one
        if filtered_names:
            if is_director_query:
                params['director'] = filtered_names[0]
            else:
                params['actor'] = filtered_names[0]
        
        # Extract keywords for description search
        # Look for descriptive words that might appear in movie plots
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
        
        # Country filtering (skip if Unknown)
        if params.get('country') and params['country'] != 'Unknown':
            print(f"DEBUG: Filtering by country: {params['country']}")
            country_mask = filtered['country'].str.contains(params['country'], case=False, na=False)
            filtered = filtered[country_mask]
            print(f"DEBUG: After country filtering: {len(filtered)} movies")
        
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
        
        # Director filtering with improved search
        if params.get('director'):
            director_name = params['director']
            print(f"DEBUG: Searching for director: '{director_name}'")
            print(f"DEBUG: DataFrame shape before director search: {filtered.shape}")
            print(f"DEBUG: Sample directors in filtered data:")
            sample_directors = filtered['director'].dropna().head(5).tolist()
            print(f"DEBUG: {sample_directors}")
            
            # Try exact match first
            try:
                director_mask = filtered['director'].str.contains(director_name, case=False, na=False)
                director_results = filtered[director_mask]
                print(f"DEBUG: Exact search found {len(director_results)} movies")
                
                # If still no results, try a simple test
                if director_results.empty:
                    print(f"DEBUG: Testing if 'Onir' exists in any director name...")
                    test_mask = filtered['director'].str.contains('Onir', case=False, na=False)
                    test_results = filtered[test_mask]
                    print(f"DEBUG: Test search for 'Onir' found {len(test_results)} movies")
                    
            except Exception as e:
                print(f"DEBUG: Error in exact search: {e}")
                director_results = filtered.iloc[0:0]  # Empty DataFrame
            
            # If no results, try searching for individual parts of the name
            if director_results.empty and ' ' in director_name:
                name_parts = director_name.split()
                for part in name_parts:
                    if len(part) > 2:  # Only search meaningful name parts
                        print(f"DEBUG: Trying director name part: '{part}'")
                        part_mask = filtered['director'].str.contains(part, case=False, na=False)
                        part_results = filtered[part_mask]
                        if not part_results.empty:
                            director_results = part_results
                            print(f"DEBUG: Found movies using director name part '{part}'")
                            break
            
            # Additional fuzzy search - try removing common prefixes
            if director_results.empty:
                # Try searching for the last word (often the main name)
                words = director_name.split()
                if len(words) > 1:
                    last_word = words[-1]
                    if len(last_word) > 2:
                        print(f"DEBUG: Trying last word search: '{last_word}'")
                        last_word_mask = filtered['director'].str.contains(last_word, case=False, na=False)
                        director_results = filtered[last_word_mask]
            
            filtered = director_results
            print(f"DEBUG: Found {len(filtered)} movies with director search")
        
        # Runtime filtering (in minutes)
        if params.get('runtime') and params.get('runtime_operator'):
            runtime_value = params['runtime']
            operator = params['runtime_operator']
            print(f"DEBUG: Filtering by runtime {operator} {runtime_value} minutes")
            
            if operator == 'greater_than':
                runtime_mask = filtered['runtime'] > runtime_value
            elif operator == 'less_than':
                runtime_mask = filtered['runtime'] < runtime_value
            elif operator == 'equal_to':
                runtime_mask = filtered['runtime'] == runtime_value
            elif operator == 'between' and isinstance(runtime_value, list):
                min_runtime, max_runtime = runtime_value
                runtime_mask = (filtered['runtime'] >= min_runtime) & (filtered['runtime'] <= max_runtime)
            else:
                runtime_mask = filtered['runtime'] > runtime_value  # Default to greater than
            
            filtered = filtered[runtime_mask]
            print(f"DEBUG: After runtime filtering: {len(filtered)} movies")

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
            
        # Different handling for specific vs general searches
        is_specific_search = bool(params.get('description_keywords'))
        
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
        
        # Final debugging before returning results
        final_results = filtered.head(6)
        print(f"DEBUG: Final results summary:")
        for i, (_, movie) in enumerate(final_results.iterrows()):
            print(f"  {i+1}. {movie['name']} - Popularity: {movie['popular']}")
        
        return final_results
    
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
        
        movie_details = []
        for movie in movie_list:
            details = f"• {movie['name']} ({movie['released']}) - {movie['genre']}"
            movie_details.append(details)
        
        formatted_movies = "\n".join(movie_details)
        
        system_prompt = f"""You are a helpful movie recommendation assistant. Respond in English.

The user asked: "{original_query}"

Here are the recommended movies with their complete information:

{formatted_movies}

IMPORTANT: Use the EXACT movie information provided above. Do not say "genre information is unavailable" - all genre information is included.

Format your response as:
1. One brief intro sentence
2. Copy the exact movie list from above with bullet points (•)

Do not modify the genre information or add any disclaimers about missing data."""

        try:
            if self.model:
                # Debug: Show exactly what data is being sent to Gemini
                print(f"DEBUG: Movies being sent to Gemini for response generation:")
                for i, movie in enumerate(movie_list):
                    print(f"  {i+1}. {movie['name']} - Popularity: {movie['popular']}")
                
                prompt = system_prompt
                response = self.model.generate_content(prompt)
                
                # Debug: Show the actual Gemini response
                print(f"DEBUG: Gemini's actual response: {response.text}")
                
                return response.text
            else:
                return self.generate_fallback_response(filtered_movies, params)
            
        except Exception as e:
            return self.generate_fallback_response(filtered_movies, params)
    
    def generate_alternative_suggestions(self, params, original_query):
        """Generate alternative suggestions when no exact matches are found."""
        # Prepare the "Sorry, but I can suggest..." message
        sorry_message = "I don't have movies that exactly match your request, but here are some alternatives:"
        
        # Try to find reasonable alternatives
        alt_movies = None
        alt_params = params.copy()
        
        # Strategy 1: If looking for specific director + age group, try just the age group
        if params.get('director') and params.get('age_group'):
            print(f"DEBUG: No movies found for director '{params['director']}' and age '{params['age_group']}'")
            # Try just the age group
            alt_params = {'age_group': params['age_group']}
            alt_movies = self.filter_movies(alt_params)
            if not alt_movies.empty:
                print(f"DEBUG: Found {len(alt_movies)} movies for age group only")
        
        # Strategy 2: If looking for specific actor + age group, try just the age group  
        elif params.get('actor') and params.get('age_group'):
            alt_params = {'age_group': params['age_group']}
            alt_movies = self.filter_movies(alt_params)
        
        # Strategy 3: If looking for genre + age group, try just the age group
        elif params.get('genre') and params.get('age_group'):
            alt_params = {'age_group': params['age_group']}
            alt_movies = self.filter_movies(alt_params)
        
        # Strategy 4: If looking for specific year + age group, try just the age group
        elif params.get('year_range') and params.get('age_group'):
            alt_params = {'age_group': params['age_group']}
            alt_movies = self.filter_movies(alt_params)
        
        # Strategy 5: If no age group specified, try popular movies
        else:
            alt_params = {'popular': 'high'}
            alt_movies = self.filter_movies(alt_params)
        
        # Generate response with alternatives
        if alt_movies is not None and not alt_movies.empty:
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

I don't have exact matches for this request, but I found related alternatives. Create a response that:
1. First says clearly "I don't have movies that exactly match your request, but here are some alternatives:"
2. Then list 3-4 movies maximum in this simple format:
   • Movie Name (Year) - Genre
   • Movie Name (Year) - Genre
3. Keep it short and helpful in English
4. Focus on why these alternatives are relevant

Movies available: {json.dumps(movie_list[:4], ensure_ascii=False)}

Keep the response under 150 words."""
                        
                        response = self.model.generate_content(alt_prompt)
                        return response.text
                    except:
                        pass
                
                # Fallback if Gemini fails
                response = sorry_message + "\n\n"
                count = 0
                for _, movie in alt_movies.iterrows():
                    if count >= 4:
                        break
                    year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
                    genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
                    response += f"• {movie['name']} ({year}) - {genre}\n"
                    count += 1
                return response
        
        # If still no alternatives, suggest appropriate fallback
        if params.get('age_group'):
            # Try to get movies for the requested age group
            fallback_params = {'age_group': params['age_group']}
            fallback_movies = self.filter_movies(fallback_params)
            if not fallback_movies.empty:
                response = "I don't have movies that exactly match your request, but here are some alternatives:\n\n"
                count = 0
                for _, movie in fallback_movies.iterrows():
                    if count >= 4:
                        break
                    year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
                    genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
                    response += f"• {movie['name']} ({year}) - {genre}\n"
                    count += 1
                return response
        
        # Final fallback - popular movies
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
    
    def handle_general_movie_question(self, user_query, params):
        """Handle general questions about movies in the dataset."""
        try:
            if self.model:
                # Search for relevant movies based on any mentioned names
                relevant_movies = self.movies.copy()
                
                # Try to find specific movies mentioned in the query
                query_lower = user_query.lower()
                movie_matches = []
                
                # Extract potential movie names from query (look for quoted text or capitalized words)
                import re
                potential_names = []
                
                # Look for quoted movie names
                quoted_matches = re.findall(r'"([^"]*)"', user_query)
                potential_names.extend(quoted_matches)
                
                # Look for capitalized words that might be movie titles
                words = user_query.split()
                for i, word in enumerate(words):
                    if word[0].isupper() and len(word) > 2:
                        # Try single word
                        potential_names.append(word)
                        # Try combinations with next words if they're also capitalized
                        if i < len(words) - 1 and words[i+1][0].isupper():
                            potential_names.append(f"{word} {words[i+1]}")
                            if i < len(words) - 2 and words[i+2][0].isupper():
                                potential_names.append(f"{word} {words[i+1]} {words[i+2]}")
                
                # Add debug logging for movie search
                print(f"DEBUG: Searching for movies in query: {user_query}")
                print(f"DEBUG: Potential movie names found: {potential_names}")
                
                # Search for these potential movie names
                for potential_name in potential_names:
                    potential_lower = potential_name.lower()
                    print(f"DEBUG: Looking for '{potential_name}' in movie database")
                    for _, movie in relevant_movies.iterrows():
                        movie_name = str(movie['name']).lower()
                        if potential_lower in movie_name or movie_name in potential_lower:
                            movie_matches.append(movie)
                            print(f"DEBUG: Found match: {movie['name']}")
                            break
                
                # If no matches found with capitalized words, try a broader search
                if not movie_matches:
                    print("DEBUG: No matches found with capitalized words, trying broader search")
                    query_words = [word.lower() for word in user_query.split() if len(word) > 2]
                    for _, movie in relevant_movies.iterrows():
                        movie_name = str(movie['name']).lower()
                        # Check if any significant words from query appear in movie name
                        for word in query_words:
                            if word in movie_name and word not in ['movie', 'film', 'also', 'year']:
                                movie_matches.append(movie)
                                print(f"DEBUG: Broader search found: {movie['name']}")
                                break
                        if movie_matches:  # Stop after first match in broader search
                            break
                
                # Prepare context for Gemini
                context = f"User question: {user_query}\n\n"
                if movie_matches:
                    context += "Relevant movies from dataset:\n"
                    for movie in movie_matches[:3]:  # Limit to 3 most relevant
                        context += f"- {movie['name']} ({movie['released']}) - Genre: {movie['genre']}, Rating: {movie['popular']}\n"
                else:
                    context += "Dataset contains 5,371 movies with information about genres, release years, popularity ratings, cast, directors, and more.\n"
                
                prompt = f"""You are a movie expert assistant. Answer the user's question based on the movie dataset information provided. 
                
Respond in English only. Be helpful and informative, using only the data provided.

{context}

Answer the user's question directly and concisely."""
                
                response = self.model.generate_content(prompt)
                return response.text
            else:
                return "I can help with movie-related questions, but I need more specific information to provide accurate answers."
                
        except Exception as e:
            return "I encountered an issue processing your question. Please try rephrasing it."

    def get_recommendation(self, user_query):
        """Main method to get movie recommendations based on user query."""
        try:
            # Extract parameters from user query
            params = self.extract_query_parameters(user_query)
            
            # Check intent and handle accordingly
            intent = params.get('intent', 'recommend')
            
            if intent == 'off_topic':
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about films, actors, directors, or movie recommendations."
            
            elif intent == 'general_movie_question':
                return self.handle_general_movie_question(user_query, params)
            
            # For recommend, check_suitability, and filter intents
            # Filter movies based on parameters
            filtered_movies = self.filter_movies(params)
            
            # Generate natural language response
            response = self.generate_response(filtered_movies, params, user_query)
            
            return response
            
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a different query."

# Initialize the recommender
recommender = MovieRecommender("attached_assets/MergeAndCleaned_Movies.csv")

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
        
        # Check for watchlist-related queries
        query_lower = query.lower()
        
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
        
        # Get conversation context for continuity (but skip for precise filtering)
        context = get_conversation_context(user_id)
        
        # Include context in query if available, but skip for specific filtering queries
        enhanced_query = query
        is_precise_filter = any(word in query.lower() for word in ['popular rate', 'popularity rating', 'rating of', 'rate is'])
        if context and not is_precise_filter:
            enhanced_query = f"{context}\nCurrent question: {query}"
        
        # Regular movie recommendation
        recommendation = recommender.get_recommendation(enhanced_query)
        
        # Save conversation to memory
        save_conversation(user_id, query, recommendation)
        
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