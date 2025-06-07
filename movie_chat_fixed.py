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
        self.movies = self.load_movies(csv_file_path)
        print(f"Successfully loaded {len(self.movies)} movies")

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
                prompt = f"""Does this question ask about movies I just showed in a previous response?

"{query}"

If asking about previous results: ANALYSIS
If asking for new movies: SEARCH

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
                prompt = f"""Extract movie search parameters from this query only:

Query: "{user_query}"

Extract just: genre, year_range, actor, director, country, age_group, description_keywords, intent

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
                        
                # Special handling for inherited parameters in current query
                if context_params.get('age_group') and not any(indicator in query.lower() for indicator in ['adults', 'teens', 'young adults']):
                    print(f"DEBUG: Inherited age_group '{context_params['age_group']}' from context (current query had priority)")
                    params['age_group'] = context_params['age_group']

        # Extract parameters from the current query FIRST (higher priority)
        query_lower = query.lower()

        # Simple age detection for kids content
        age_indicators = ['year old', 'years old', 'month old', 'months old', 'baby', 'toddler', 'infant']
        for indicator in age_indicators:
            if indicator in query_lower:
                age_match = re.search(r'(\d+)\s*(year|month)', query_lower)
                if age_match:
                    age_num = int(age_match.group(1))
                    age_unit = age_match.group(2)
                    if (age_unit == 'month' and age_num <= 24) or (age_unit == 'year' and age_num <= 12):
                        params['age_group'] = 'Kids'
                break

        # Kids/Family detection
        kids_indicators = [
            'for kids', 'for children', 'suitable for kids', 'suit to kids',
            'that will suit', 'appropriate for kids', 'family friendly',
            'children can watch', 'kids can watch', 'child friendly'
        ]

        # Adults detection
        adults_indicators = [
            'for adults', 'suit for adults', 'suitable for adults', 'adult movies',
            'adults only', 'mature content', 'grown ups'
        ]

        for indicator in kids_indicators:
            if indicator in query_lower:
                params['age_group'] = 'Kids'
                break
                
        for indicator in adults_indicators:
            if indicator in query_lower:
                params['age_group'] = 'Adults'
                print(f"DEBUG: Detected adults request from current query: {indicator}")
                break

        # Genre detection using fuzzy matching
        genre_keywords = {
            'romance': ['romance', 'romantic', 'רומנטי', 'רומנטית', 'אהבה'],
            'action': ['action', 'actoin', 'akshen', 'אקשן', 'פעולה'],
            'comedy': ['comedy', 'comedies', 'comdy', 'funny', 'קומדיה'],
            'drama': ['drama', 'drame', 'דרמה'],
            'horror': ['horror', 'scary', 'horer', 'horor', 'אימה'],
            'thriller': ['thriller', 'thriler', 'suspense', 'מתח'],
            'sci-fi': ['sci-fi', 'science fiction', 'scifi', 'מדע בדיוני'],
            'fantasy': ['fantasy', 'fantesy', 'פנטזיה'],
            'documentary': ['documentary', 'documentry', 'docu', 'תיעודי'],
            'animation': ['animation', 'animated', 'cartoon', 'אנימציה']
        }

        # Split query into words for fuzzy matching
        query_words = query_lower.replace(',', ' ').replace('.', ' ').split()
        
        for genre, keywords in genre_keywords.items():
            for word in query_words:
                if self.fuzzy_match(word, keywords):
                    params['genre'] = genre
                    print(f"DEBUG: Detected genre '{genre}' from word '{word}'")
                    break
            if params['genre']:
                break
        
        # Year range detection (2014-2016, from 2014 to 2016, etc.)
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
                break
        
        # Single year detection (only if no range found)
        if not params.get('year_range'):
            year_patterns = [
                r'from (\d{4})', r'since (\d{4})', r'after (\d{4})',
                r'in (\d{4})', r'(\d{4}) movies', r'released in (\d{4})'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    year = int(match.group(1))
                    params['year_range'] = [year, year]
                    break
        
        # Popular/rating detection
        if any(word in query_lower for word in ['popular', 'top', 'best', 'highest rated']):
            params['popular'] = 'high'
        elif any(word in query_lower for word in ['bad', 'worst', 'lowest rated']):
            params['popular'] = 'low'

        # AFTER processing current query, inherit missing parameters from context
        # This ensures current query parameters have priority
        if conversation_context:
            if self.is_followup_query(query, conversation_context):
                context_params = self.extract_context_parameters(conversation_context)
                # Only inherit parameters that weren't found in current query
                for key, value in context_params.items():
                    if params.get(key) is None and value is not None:
                        params[key] = value
                        print(f"DEBUG: Inherited {key} '{value}' from context (current query had priority)")

        return params
    
    def fuzzy_match(self, word, target_words, threshold=0.8):
        """Check if word matches any target word with fuzzy matching."""
        import difflib

        word_lower = word.lower().strip()
        
        # Skip very short words to avoid false positives
        if len(word_lower) < 3:
            return False

        for target in target_words:
            target_lower = target.lower()
            
            # Skip if target is much longer than word (avoids "me" matching "comedy")
            if len(word_lower) < len(target_lower) / 2:
                continue

            # Exact match
            if word_lower == target_lower:
                return True

            # Check if word contains target or target contains word (but not for very short words)
            if len(word_lower) >= 4 and len(target_lower) >= 4:
                if word_lower in target_lower or target_lower in word_lower:
                    return True

            # Fuzzy matching using sequence similarity with higher threshold
            similarity = difflib.SequenceMatcher(None, word_lower, target_lower).ratio()
            if similarity >= threshold and len(word_lower) >= 4:
                return True

        return False

    def is_followup_query(self, query, context):
        """Use Gemini to determine if query is a follow-up or new topic."""
        try:
            if self.model:
                prompt = f"""Determine if this query is a follow-up to the previous conversation or a new topic.

Previous conversation:
{context}

Current query: {query}

Respond with exactly "FOLLOWUP" if it's a follow-up question, or "NEW" if it's a new topic."""

                response = self.model.generate_content(prompt)
                return "FOLLOWUP" in response.text.upper()
            else:
                # Enhanced heuristics for followup detection
                if not context:
                    return False
                    
                query_lower = query.lower().strip()
                
                # Strong followup indicators
                strong_indicators = [
                    'that suit', 'suit for', 'from that list', 'from those', 'of those',
                    'which one', 'what about', 'tell me about', 'more about',
                    'from the', 'of the', 'that are', 'those that'
                ]
                
                # Year/time indicators  
                import re
                year_patterns = [
                    r'from \d{4}', r'in \d{4}', r'after \d{4}', r'before \d{4}',
                    r'\d{4}-\d{4}', r'between \d{4}'
                ]
                
                # Check strong indicators
                for indicator in strong_indicators:
                    if indicator in query_lower:
                        return True
                
                # Check year patterns
                for pattern in year_patterns:
                    if re.search(pattern, query_lower):
                        return True
                
                # Short queries with weak indicators
                if len(query_lower.split()) <= 4:
                    weak_indicators = ['from', 'in', 'only', 'just', 'for', 'with']
                    return any(word in query_lower for word in weak_indicators)
                
                return False
        except:
            return False

    def extract_context_parameters(self, context):
        """Extract relevant parameters from conversation context."""
        params = {}
        context_lower = context.lower()

        # Extract genre from context
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
            'animation': ['animation', 'animated', 'cartoon'],
            'adventure': ['adventure'],
            'crime': ['crime'],
            'mystery': ['mystery'],
            'musical': ['musical', 'music'],
            'war': ['war'],
            'western': ['western'],
            'family': ['family']
        }

        # Look for user queries in context to extract parameters
        lines = context.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Extract from user queries specifically
            if 'user:' in line_lower:
                user_part = line_lower.split('user:')[-1].strip()
                
                # Check for genres in user queries
                for genre, keywords in genre_keywords.items():
                    for keyword in keywords:
                        if keyword in user_part:
                            params['genre'] = genre
                            print(f"DEBUG: Inherited genre '{genre}' from context user query")
                            break
                    if params.get('genre'):
                        break
                
                # Check for age group in user queries
                if any(word in user_part for word in ['kids', 'children', 'family']):
                    params['age_group'] = 'Kids'
                    print(f"DEBUG: Inherited age_group 'Kids' from context")
                elif any(word in user_part for word in ['adults', 'adult']):
                    params['age_group'] = 'Adults'
                    print(f"DEBUG: Inherited age_group 'Adults' from context")

        # Also check Assistant responses for age group context
        for line in lines:
            line_lower = line.lower()
            if 'assistant:' in line_lower:
                assistant_part = line_lower.split('assistant:')[-1].strip()
                
                # Check for age group in assistant responses
                if not params.get('age_group'):
                    if any(phrase in assistant_part for phrase in ['suitable for kids', 'for kids', 'children & family']):
                        params['age_group'] = 'Kids'
                        print(f"DEBUG: Inherited age_group 'Kids' from assistant response")
                    elif any(phrase in assistant_part for phrase in ['suitable for adults', 'for adults']):
                        params['age_group'] = 'Adults'
                        print(f"DEBUG: Inherited age_group 'Adults' from assistant response")

        # DON'T inherit genres from entire context to avoid conflicts with current query
        # Only inherit from explicit user queries

        # Fallback for age group from entire context
        if not params.get('age_group'):
            if any(word in context_lower for word in ['kids', 'children', 'family']):
                params['age_group'] = 'Kids'
                print(f"DEBUG: Inherited age_group 'Kids' from general context")
            elif any(word in context_lower for word in ['adults', 'adult']):
                params['age_group'] = 'Adults'
                print(f"DEBUG: Inherited age_group 'Adults' from general context")

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
                min_year, max_year = year_range
                year_mask = (
                    (filtered['released'] >= min_year) & 
                    (filtered['released'] <= max_year)
                )
                filtered = filtered[year_mask]
                print(f"DEBUG: After year filtering: {len(filtered)} movies")
        
        # Popularity filtering
        if params.get('popular'):
            popularity = params['popular']
            print(f"DEBUG: Filtering by popularity: {popularity}")
            
            if popularity == 'high':
                # Top 30% of movies by rating
                threshold = filtered['popular'].quantile(0.7)
                filtered = filtered[filtered['popular'] >= threshold]
            elif popularity == 'low':
                # Bottom 30% of movies by rating
                threshold = filtered['popular'].quantile(0.3)
                filtered = filtered[filtered['popular'] <= threshold]
            elif isinstance(popularity, (int, float)):
                # Specific rating
                filtered = filtered[filtered['popular'] >= popularity]
                
            print(f"DEBUG: After popularity filtering: {len(filtered)} movies")
        
        # Actor filtering
        if params.get('actor'):
            actor_name = params['actor'].lower()
            print(f"DEBUG: Filtering by actor: {actor_name}")
            
            # Search in relevant columns that might contain actor names
            if 'actors' in filtered.columns:
                actor_mask = filtered['actors'].str.lower().str.contains(actor_name, na=False)
                filtered = filtered[actor_mask]
            elif 'cast' in filtered.columns:
                actor_mask = filtered['cast'].str.lower().str.contains(actor_name, na=False)
                filtered = filtered[actor_mask]
            
            print(f"DEBUG: After actor filtering: {len(filtered)} movies")
        
        # Director filtering
        if params.get('director'):
            director_name = params['director'].lower()
            print(f"DEBUG: Filtering by director: {director_name}")
            
            if 'director' in filtered.columns:
                director_mask = filtered['director'].str.lower().str.contains(director_name, na=False)
                filtered = filtered[director_mask]
            
            print(f"DEBUG: After director filtering: {len(filtered)} movies")
        
        # Country filtering
        if params.get('country'):
            country_name = params['country'].lower()
            print(f"DEBUG: Filtering by country: {country_name}")
            
            if 'country' in filtered.columns:
                country_mask = filtered['country'].str.lower().str.contains(country_name, na=False)
                filtered = filtered[country_mask]
            
            print(f"DEBUG: After country filtering: {len(filtered)} movies")
        
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
        
        # Age group filtering using the actual age_group column
        if params.get('age_group'):
            requested_age = params['age_group']
            print(f"DEBUG: Filtering by age_group: {requested_age}")
            
            if 'age_group' in filtered.columns:
                age_filter = filtered['age_group'] == requested_age
                filtered = filtered[age_filter]
                print(f"DEBUG: After age_group filtering: {len(filtered)} movies for {requested_age}")
            else:
                print(f"DEBUG: age_group column not found, skipping age filtering")
        
        # Apply weighted scoring and sort (70% popularity, 30% year)
        if not filtered.empty and 'popular' in filtered.columns and 'released' in filtered.columns:
            # Normalize popularity (0-5 scale) and year (assume 1900-2025 range)
            filtered['popularity_normalized'] = filtered['popular'] / 5.0
            current_year = 2025
            filtered['year_normalized'] = (filtered['released'] - 1900) / (current_year - 1900)
            
            # Calculate combined score: 70% popularity + 30% year
            filtered['combined_score'] = (0.7 * filtered['popularity_normalized']) + (0.3 * filtered['year_normalized'])
            
            # Sort by combined score (highest first)
            filtered = filtered.sort_values('combined_score', ascending=False)
        elif 'popular' in filtered.columns:
            # Fallback to popularity only if year data unavailable
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
                
                prompt = f"""User asks: "{query}"

Movies: {movies_data}

Answer in 1-2 sentences max. Be direct and helpful."""

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
            params = self.extract_query_parameters(user_query, conversation_context)
            filtered_movies = self.filter_movies(params)
            return self.generate_analytical_response(filtered_movies, user_query, conversation_context)

        # Regular search query
        params = self.extract_query_parameters(user_query, conversation_context)
        filtered_movies = self.filter_movies(params)
        
        print(f"DEBUG: Final results summary:")
        for i, (_, movie) in enumerate(filtered_movies.head(6).iterrows(), 1):
            rating = movie['popular'] if pd.notna(movie['popular']) else 'N/A'
            print(f"  {i}. {movie['name']} - Popularity: {rating}")
        
        # Generate personalized response based on search criteria
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Try a different search!"
        
        # Create personalized introduction based on parameters
        intro = self.generate_personalized_intro(params)
        
        # Generate formatted movie list
        movies_list = []
        for _, movie in filtered_movies.head(6).iterrows():
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
            movies_list.append(f"• {movie['name']} ({year}) - {genre}")
        
        return intro + "\n" + "\n".join(movies_list)
    
    def generate_personalized_intro(self, params):
        """Generate personalized introduction based on search parameters."""
        intro_parts = []
        
        # Check what parameters were used
        if params.get('genre'):
            genre = params['genre']
            intro_parts.append(f"{genre} movies")
        
        if params.get('age_group'):
            age_group = params['age_group']
            age_mapping = {
                'Kids': 'suitable for kids',
                'Teens': 'suitable for teens', 
                'Adults': 'suitable for adults',
                'Young Adults': 'suitable for young adults'
            }
            age_text = age_mapping.get(age_group, age_group)
            intro_parts.append(f"that are {age_text}")
        
        if params.get('year_range'):
            year_range = params['year_range']
            if len(year_range) == 2 and year_range[0] == year_range[1]:
                intro_parts.append(f"from {year_range[0]}")
            else:
                intro_parts.append(f"from {year_range[0]}-{year_range[1]}")
        
        if params.get('country'):
            intro_parts.append(f"from {params['country']}")
            
        if params.get('popular'):
            popularity_map = {
                'high': 'popular',
                'medium': 'moderately popular',
                'low': 'lesser known'
            }
            pop_text = popularity_map.get(params['popular'], 'popular')
            intro_parts.append(pop_text)
        
        # Build the introduction
        if intro_parts:
            intro = "Here are " + " ".join(intro_parts) + ":"
        else:
            intro = "Here are some movie recommendations:"
            
        return intro

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
    movies_df = recommender.movies
    
    print("System initialized successfully!")

def setup_routes():
    """Setup Flask routes."""
    
    @app.route('/')
    def index():
        with open('index.html', 'r') as f:
            return f.read()

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