import pandas as pd
import json
import os
import google.generativeai as genai
import re

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
            df = pd.read_csv(csv_file_path)
            required_columns = ['title', 'genre', 'year', 'rating', 'min_age']
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and validate data
            df = df.dropna(subset=['title'])
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['min_age'] = pd.to_numeric(df['min_age'], errors='coerce')
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def extract_query_parameters(self, user_query):
        """Use Gemini to extract parameters from natural language query."""
        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries.

Extract the following information from the user's query and return as JSON:
- age: target age (number) if mentioned
- genre: specific genre if mentioned (e.g., comedy, action, drama)
- max_age: maximum age restriction if asking about age-appropriate content
- year_range: [min_year, max_year] if mentioned
- rating_min: minimum rating if mentioned
- intent: the main intent (recommend, check_suitability, filter, general)

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
                
                return json.loads(response_text)
            else:
                return self.basic_parameter_extraction(user_query)
            
        except Exception as e:
            # Fallback to basic parameter extraction
            return self.basic_parameter_extraction(user_query)
    
    def basic_parameter_extraction(self, query):
        """Basic fallback parameter extraction without AI."""
        params = {
            'age': None,
            'genre': None,
            'max_age': None,
            'year_range': None,
            'rating_min': None,
            'intent': 'recommend'
        }
        
        # Extract age mentions
        age_pattern = r'\b(\d{1,2})\b'
        ages = [int(match) for match in re.findall(age_pattern, query)]
        if ages:
            params['age'] = ages[0]
            params['max_age'] = ages[0]
        
        # Basic genre detection
        genres = ['comedy', 'action', 'drama', 'horror', 'romance', 'animation', 'documentary', 'family', 'adventure', 'sci-fi', 'fantasy', 'musical']
        
        query_lower = query.lower()
        for genre in genres:
            if genre in query_lower:
                params['genre'] = genre
                break
        
        return params
    
    def filter_movies(self, params):
        """Filter movies based on extracted parameters."""
        filtered = self.movies.copy()
        
        # Age filtering
        if params.get('max_age'):
            filtered = filtered[filtered['min_age'] <= params['max_age']]
        
        # Genre filtering
        if params.get('genre'):
            genre_mask = filtered['genre'].str.contains(params['genre'], case=False, na=False)
            filtered = filtered[genre_mask]
        
        # Year range filtering
        if params.get('year_range'):
            min_year, max_year = params['year_range']
            filtered = filtered[
                (filtered['year'] >= min_year) & 
                (filtered['year'] <= max_year)
            ]
        
        # Rating filtering
        if params.get('rating_min'):
            filtered = filtered[filtered['rating'] >= params['rating_min']]
        
        return filtered.head(10)  # Limit to top 10 recommendations
    
    def generate_response(self, filtered_movies, params, original_query):
        """Generate a natural language response using Gemini."""
        if filtered_movies.empty:
            return self.generate_no_results_response(params)
        
        # Prepare movie data for the prompt
        movie_list = []
        for _, movie in filtered_movies.iterrows():
            movie_info = {
                'title': movie['title'],
                'genre': movie['genre'],
                'year': int(movie['year']) if pd.notna(movie['year']) else 'Unknown',
                'rating': float(movie['rating']) if pd.notna(movie['rating']) else 'N/A',
                'min_age': int(movie['min_age']) if pd.notna(movie['min_age']) else 'N/A'
            }
            movie_list.append(movie_info)
        
        system_prompt = f"""You are a helpful movie recommendation assistant. Respond in English.

The user asked: "{original_query}"

Provide a natural, conversational response that:
1. Acknowledges their request
2. Lists the recommended movies with key details
3. Explains why these movies are suitable
4. Uses an engaging, friendly tone

Format the response naturally, not as a list or formal structure."""

        try:
            if self.model:
                prompt = f"{system_prompt}\n\nHere are the movies I found: {json.dumps(movie_list, ensure_ascii=False)}"
                response = self.model.generate_content(prompt)
                return response.text
            else:
                return self.generate_fallback_response(filtered_movies, params)
            
        except Exception as e:
            return self.generate_fallback_response(filtered_movies, params)
    
    def generate_no_results_response(self, params):
        """Generate response when no movies match the criteria."""
        return "I'm sorry, I couldn't find any movies matching your criteria. Try adjusting your requirements or asking about a different genre."
    
    def generate_fallback_response(self, filtered_movies, params):
        """Generate a basic response without AI."""
        response = "Here are some recommendations for you:\n\n"
        for _, movie in filtered_movies.iterrows():
            response += f"🎬 **{movie['title']}** ({movie['year']})\n"
            response += f"   Genre: {movie['genre']}\n"
            response += f"   Rating: {movie['rating']}/10\n"
            response += f"   Suitable from age: {movie['min_age']}\n\n"
        
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
