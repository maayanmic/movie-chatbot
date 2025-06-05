import pandas as pd
import json
import os
from openai import OpenAI
import re

class MovieRecommender:
    def __init__(self, csv_file_path):
        """Initialize the movie recommender with CSV data and OpenAI client."""
        self.movies = self.load_movies(csv_file_path)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
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
        """Use OpenAI to extract parameters from natural language query."""
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries in Hebrew and English.

Extract the following information from the user's query and return as JSON:
- age: target age (number) if mentioned
- genre: specific genre if mentioned (e.g., comedy, action, drama)
- max_age: maximum age restriction if asking about age-appropriate content
- year_range: [min_year, max_year] if mentioned
- rating_min: minimum rating if mentioned
- language: detected language of the query (hebrew/english)
- intent: the main intent (recommend, check_suitability, filter, general)

Return JSON format only."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            # Fallback to basic parameter extraction
            return self.basic_parameter_extraction(user_query)
    
    def basic_parameter_extraction(self, query):
        """Basic fallback parameter extraction without OpenAI."""
        params = {
            'age': None,
            'genre': None,
            'max_age': None,
            'year_range': None,
            'rating_min': None,
            'language': 'hebrew' if any(ord(char) > 127 for char in query) else 'english',
            'intent': 'recommend'
        }
        
        # Extract age mentions
        age_pattern = r'\b(\d{1,2})\b'
        ages = [int(match) for match in re.findall(age_pattern, query)]
        if ages:
            params['age'] = ages[0]
            params['max_age'] = ages[0]
        
        # Basic genre detection
        genres = ['comedy', 'action', 'drama', 'horror', 'romance', 'animation', 'documentary']
        hebrew_genres = ['קומדיה', 'פעולה', 'דרמה', 'אימה', 'רומנטיקה', 'אנימציה']
        
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
        """Generate a natural language response using OpenAI."""
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
        
        language = params.get('language', 'english')
        
        system_prompt = f"""You are a helpful movie recommendation assistant. Respond in {"Hebrew" if language == 'hebrew' else "English"}.

The user asked: "{original_query}"

Provide a natural, conversational response that:
1. Acknowledges their request
2. Lists the recommended movies with key details
3. Explains why these movies are suitable
4. Uses an engaging, friendly tone

Format the response naturally, not as a list or formal structure."""

        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here are the movies I found: {json.dumps(movie_list, ensure_ascii=False)}"}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return self.generate_fallback_response(filtered_movies, params)
    
    def generate_no_results_response(self, params):
        """Generate response when no movies match the criteria."""
        language = params.get('language', 'english')
        
        if language == 'hebrew':
            return "מצטער, לא מצאתי סרטים שמתאימים לקריטריונים שלך. נסה לשנות את הדרישות או לשאול על ז'אנר אחר."
        else:
            return "I'm sorry, I couldn't find any movies matching your criteria. Try adjusting your requirements or asking about a different genre."
    
    def generate_fallback_response(self, filtered_movies, params):
        """Generate a basic response without OpenAI."""
        language = params.get('language', 'english')
        
        if language == 'hebrew':
            response = "הנה כמה המלצות עבורך:\n\n"
            for _, movie in filtered_movies.iterrows():
                response += f"🎬 **{movie['title']}** ({movie['year']})\n"
                response += f"   ז'אנר: {movie['genre']}\n"
                response += f"   דירוג: {movie['rating']}/10\n"
                response += f"   מתאים מגיל: {movie['min_age']}\n\n"
        else:
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
