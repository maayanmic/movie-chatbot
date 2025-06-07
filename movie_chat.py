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
                params.update(context_params)
        
        # Extract additional parameters from the current query itself 
        # (beyond what was extracted from context)
        query_lower = query.lower()
        
        # Simple age detection for kids content
        age_indicators = ['year old', 'years old', 'month old', 'months old', 'baby', 'toddler', 'infant']
        for indicator in age_indicators:
            if indicator in query_lower:
                # Extract age if mentioned
                import re
                age_match = re.search(r'(\d+)\s*(year|month)', query_lower)
                if age_match:
                    age_num = int(age_match.group(1))
                    age_unit = age_match.group(2)
                    
                    # Convert to approximate age groups
                    if (age_unit == 'month' and age_num <= 24) or (age_unit == 'year' and age_num <= 2):
                        params['age_group'] = 'Kids'
                        print(f"DEBUG: Detected very young age ({age_num} {age_unit}), setting to Kids")
                    elif age_unit == 'year' and age_num <= 12:
                        params['age_group'] = 'Kids'
                        print(f"DEBUG: Detected child age ({age_num} {age_unit}), setting to Kids")
                break
        
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

        # Extract year or year range from current query if present
        import re
        
        # First try to find year ranges like "from 2019 to 2021", "2019-2021", "between 2019 and 2021"
        range_patterns = [
            r'from\s+(\d{4})\s+to\s+(\d{4})',
            r'between\s+(\d{4})\s+and\s+(\d{4})',
            r'(\d{4})\s*-\s*(\d{4})',
            r'(\d{4})\s+to\s+(\d{4})'
        ]
        
        year_range_found = False
        for pattern in range_patterns:
            range_match = re.search(pattern, query_lower)
            if range_match:
                start_year = int(range_match.group(1))
                end_year = int(range_match.group(2))
                # Ensure valid year range
                if 1900 <= start_year <= 2030 and 1900 <= end_year <= 2030:
                    params['year_range'] = [min(start_year, end_year), max(start_year, end_year)]
                    year_range_found = True
                    break
        
        # If no range found, look for single year
        if not year_range_found:
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
            
        if not self.model:
            # Simple fallback if no AI available
            return len(query.split()) <= 3
            
        try:
            prompt = f"""
Analyze if the user's current query is a follow-up to the previous conversation or a completely new movie request.

Previous conversation:
{context}

Current query: "{query}"

Return only "FOLLOWUP" or "NEW_TOPIC"

Guidelines:
- FOLLOWUP: refining/filtering previous results, asking for more options, temporal references, single words/short phrases
- NEW_TOPIC: complete new movie requests, different genres/topics, asking for recommendations from scratch

Examples:
- "only romantic" → FOLLOWUP
- "from 2019" → FOLLOWUP  
- "there is more?" → FOLLOWUP
- "but I want something older that came out in the 1990s" → FOLLOWUP
- "What action movies do you recommend?" → NEW_TOPIC
- "I want comedy movies" → NEW_TOPIC
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
            print(f"DEBUG: Gemini failed: {str(e)}, using enhanced fallback")
            # Enhanced fallback when Gemini is unavailable
            query_lower = query.lower().strip()
            
            # Clear new topic indicators - complete requests for movies
            new_topic_phrases = [
                'what movies', 'which movies', 'can you give me', 'give me', 'can you recommend', 
                'recommend', 'suggest', 'i want movies', 'i need movies', 'i want', 'i need',
                'find me', 'show me', 'looking for', 'help me find'
            ]
            
            # Check for genre + movies combinations that indicate new requests
            genre_requests = [
                'drama movies', 'action movies', 'comedy movies', 'horror movies',
                'romantic movies', 'sci-fi movies', 'fantasy movies', 'thriller movies'
            ]
            
            if any(phrase in query_lower for phrase in new_topic_phrases + genre_requests):
                return False
            
            # Strong follow-up indicators (regardless of length)
            follow_indicators = ['only', 'just', 'from', 'but', 'and', 'also', 'more', 'other', 'newer', 'older']
            if any(query_lower.startswith(word + ' ') for word in follow_indicators):
                return True
            
            # Year patterns indicate refinement
            import re
            if re.search(r'\b(19|20)\d{2}\b', query):
                return True
            
            # Default: short queries are follow-ups
            return len(query.split()) <= 4


    def extract_context_parameters(self, context):
        """Extract relevant parameters from conversation context."""
        params = {}
        
        # Find ALL user queries from context, not just the last one
        lines = context.split('\n')
        user_queries = []
        for line in lines:
            if line.startswith('User: '):
                user_queries.append(line[6:])  # Remove 'User: ' prefix
        
        if not user_queries:
            return params
            
        # Process all user queries to accumulate parameters
        # Start from the oldest query and build up context
        for query in user_queries:
            query_params = self.basic_parameter_extraction(query, "")
            
            # Add parameters that aren't already set (first occurrence wins for core attributes)
            for key, value in query_params.items():
                if value is not None and key != 'intent':
                    if key not in params or params[key] is None:
                        params[key] = value
                        print(f"DEBUG: Inherited {key}='{value}' from context query: {query}")
                
        return params

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

            # Check if this is a follow-up query or new topic
            is_followup = self.is_followup_query(user_query, conversation_context) if conversation_context else False
            
            # Extract parameters from query
            # Only pass context if it's a follow-up query
            context_to_use = conversation_context if is_followup else ""
            params = self.extract_query_parameters(user_query, context_to_use)

            # Handle off-topic intent
            if params.get('intent') == 'off_topic':
                return "I'm sorry, but I specialize only in the world of movies. Please ask me about movie recommendations, actors, directors, or anything related to films!"

            # Handle alternative suggestions for negative feedback
            if params.get('intent') == 'suggest_alternatives':
                return self.suggest_alternatives(conversation_context, user_query)

            # Filter movies based on parameters
            filtered_movies = self.filter_movies(params)

            # Generate response
            if not filtered_movies.empty:
                return self.generate_response(filtered_movies, params, user_query)
            else:
                return "I couldn't find any movies matching your specific criteria. Try broadening your search or asking for different genres, years, or actors."

        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a different query."
    
    def suggest_alternatives(self, conversation_context, user_query):
        """Suggest alternative genres when user doesn't like previous recommendations."""
        
        # Extract the previous genre from context
        previous_genre = None
        if conversation_context:
            context_params = self.extract_context_parameters(conversation_context)
            previous_genre = context_params.get('genre')
        
        # Define alternative genre mappings
        genre_alternatives = {
            'Romance': ['Comedy', 'Drama', 'Family'],
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
                question_patterns = ['which', 'what', 'how', 'are they', 'is it', 'tell me', 'pick', 'choose', 'recommend', 'suggest']
                # Check for analysis context words
                analysis_context = ['one', 'best', 'better', 'rating', 'suitable', 'good', 'about']
                
                has_question = any(pattern in query_lower for pattern in question_patterns)
                has_context = any(context in query_lower for context in analysis_context)
                
                return has_question and has_context
        except Exception as e:
            # Basic fallback
            query_lower = query.lower()
            return any(word in query_lower for word in ['which', 'pick', 'recommend', 'best', 'rating', 'suitable'])
    
    def generate_analytical_response(self, filtered_movies, query):
        """Generate analytical response using Gemini."""
        if filtered_movies.empty:
            return "I don't have any movies to analyze based on your previous search."
        
        try:
            if self.model:
                # Prepare movie data for analysis
                movies_data = []
                for _, movie in filtered_movies.head(10).iterrows():  # Limit to 10 for analysis
                    year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
                    genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
                    rating = movie['popular'] if pd.notna(movie['popular']) else 'Unknown'
                    movies_data.append({
                        'title': movie['name'],
                        'year': year,
                        'genre': genre,
                        'popularity_rating': rating
                    })
                
                # Create prompt for analysis
                prompt = f"""You are a movie recommendation chatbot. The user is asking: "{query}"

Here are the movies from their previous search:
{movies_data}

Give a SHORT, conversational response (2-3 sentences max). Examples:

User asks "which one you recommend?" → Pick ONE movie and briefly explain why
User asks "are they suitable for kids?" → Simple yes/no with quick reason
User asks about ratings → Brief comparison or explanation

Be friendly but CONCISE. Keep it short and helpful."""

                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return self.generate_basic_analytical_response(filtered_movies, query)
                
        except Exception as e:
            return self.generate_basic_analytical_response(filtered_movies, query)
    
    def generate_basic_analytical_response(self, filtered_movies, query):
        """Generate basic analytical response without AI - simple fallback."""
        if len(filtered_movies) == 0:
            return "I don't have any movies to analyze based on your search."
        
        # Very basic response that doesn't rely on specific keywords
        if len(filtered_movies) == 1:
            movie = filtered_movies.iloc[0]
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            rating = movie['popular'] if pd.notna(movie['popular']) else 'Unknown'
            genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
            return f"Looking at \"{movie['name']}\" ({year}) - it has a popularity rating of {rating}/5 in the {genre} category."
        
        # For multiple movies, provide general statistics
        total_movies = len(filtered_movies)
        avg_rating = filtered_movies['popular'].mean() if not filtered_movies['popular'].isnull().all() else 0
        return f"I found {total_movies} movies with an average popularity rating of {avg_rating:.1f}/5. Let me know what specific aspect you'd like to know more about."

    def generate_fallback_response(self, filtered_movies, params=None):
        """Generate a basic response without AI."""
        if filtered_movies.empty:
            return "I couldn't find any movies matching your criteria. Please try a different search."

        # Generate contextual intro based on parameters
        intro = "Here are some movie recommendations"
        if params:
            criteria = []
            if params.get('genre'):
                criteria.append(f"{params['genre'].lower()} movies")
            if params.get('age_group'):
                criteria.append(f"suitable for {params['age_group'].lower()}")
            if params.get('year_range'):
                criteria.append(f"from {params['year_range']}")
            if params.get('country'):
                criteria.append(f"from {params['country']}")
            
            if criteria:
                intro += f" for {' and '.join(criteria)}"
        
        intro += ":\n\n"
        
        response = intro
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
        # Use a fixed user ID for API testing, or session-based for web interface
        if request.headers.get('Content-Type') == 'application/json':
            # For API requests, use a fixed user ID to maintain conversation
            return 'api_user'
        else:
            # For web interface, use session-based user ID
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