from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
import google.generativeai as genai
import re

app = Flask(__name__, static_folder='.', template_folder='.')

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
            # Clean and validate data
            df = df.dropna(subset=['name'])
            df['released'] = pd.to_numeric(df['released'], errors='coerce')
            df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def extract_query_parameters(self, user_query):
        """Use Gemini to extract parameters from natural language query."""
        system_prompt = """You are a movie recommendation assistant that extracts search parameters from natural language queries.

Extract the following information from the user's query and return as JSON:
- age_group: target age group if mentioned (Kids, Teens, Young Adults, Adults)
- genre: specific genre if mentioned (e.g., Horror, Action, Drama, Comedy)
- rating: content rating if mentioned (G, PG, PG-13, R, TV-MA, etc.)
- year_range: [min_year, max_year] if mentioned
- country: specific country if mentioned
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
            'age_group': None,
            'genre': None,
            'rating': None,
            'year_range': None,
            'country': None,
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
        
        return params
    
    def filter_movies(self, params):
        """Filter movies based on extracted parameters."""
        filtered = self.movies.copy()
        
        # Age group filtering
        if params.get('age_group'):
            filtered = filtered[filtered['age_group'] == params['age_group']]
        
        # Genre filtering
        if params.get('genre'):
            genre_mask = filtered['genre'].str.contains(params['genre'], case=False, na=False)
            filtered = filtered[genre_mask]
        
        # Rating filtering
        if params.get('rating'):
            filtered = filtered[filtered['rating'] == params['rating']]
        
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
        
        return filtered.head(10)  # Limit to top 10 recommendations
    
    def generate_response(self, filtered_movies, params, original_query):
        """Generate a natural language response using Gemini."""
        if filtered_movies.empty:
            return self.generate_no_results_response(params)
        
        # Prepare movie data for the prompt
        movie_list = []
        for _, movie in filtered_movies.iterrows():
            movie_info = {
                'name': movie['name'],
                'genre': movie['genre'],
                'released': int(movie['released']) if pd.notna(movie['released']) else 'Unknown',
                'rating': movie['rating'] if pd.notna(movie['rating']) else 'N/A',
                'age_group': movie['age_group'] if pd.notna(movie['age_group']) else 'N/A',
                'runtime': int(movie['runtime']) if pd.notna(movie['runtime']) else 'N/A',
                'country': movie['country'] if pd.notna(movie['country']) else 'N/A',
                'description': movie['description'] if pd.notna(movie['description']) else 'No description available'
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
            response += f"🎬 **{movie['name']}** ({movie['released']})\n"
            response += f"   Genre: {movie['genre']}\n"
            response += f"   Rating: {movie['rating']}\n"
            response += f"   Age Group: {movie['age_group']}\n"
            if pd.notna(movie['description']):
                response += f"   Description: {movie['description'][:100]}...\n"
            response += "\n"
        
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
recommender = MovieRecommender("movies_clean.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        recommendation = recommender.get_recommendation(query)
        return jsonify({'recommendation': recommendation})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)