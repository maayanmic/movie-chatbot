import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


# Configuration class for better settings management
@dataclass
class Config:
    SECRET_KEY: str = 'movie_recommender_secret_key_2024'
    CSV_FILE_PATH: str = "attached_assets/MergeAndCleaned_Movies.csv"
    GEMINI_MODEL: str = 'gemini-1.5-flash'
    MAX_CONVERSATION_HISTORY: int = 20
    DEFAULT_MOVIES_LIMIT: int = 6
    RATE_LIMIT: str = "100 per hour"
    LOG_LEVEL: str = "INFO"


# Setup logging
def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('movie_chatbot.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# Custom exceptions
class MovieBotException(Exception):
    """Base exception for MovieBot"""
    pass


class DataLoadException(MovieBotException):
    """Raised when data loading fails"""
    pass


class APIException(MovieBotException):
    """Raised when API calls fail"""
    pass


# Abstract base class for AI services
class AIService(ABC):
    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        pass


# Gemini AI Service implementation
class GeminiService(AIService):
    def __init__(self, api_key: Optional[str], model: str):
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)
                self.logger.info("Gemini API initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini: {e}")
                raise APIException(f"Gemini initialization failed: {e}")
        else:
            self.logger.warning("No Gemini API key provided")
    
    def generate_content(self, prompt: str) -> str:
        if not self.model:
            raise APIException("Gemini model not available")
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            raise APIException(f"Content generation failed: {e}")
    
    def is_available(self) -> bool:
        return self.model is not None


# Data validation and cleaning utilities
class DataValidator:
    @staticmethod
    def validate_movie_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean movie data"""
        required_columns = ['name', 'genre', 'released', 'popular']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataLoadException(f"Missing required columns: {missing_columns}")
        
        # Clean data
        df = df.dropna(subset=['name'])
        df['released'] = pd.to_numeric(df['released'], errors='coerce')
        df['popular'] = pd.to_numeric(df['popular'], errors='coerce')
        
        # Remove invalid years and ratings
        df = df[(df['released'] >= 1900) & (df['released'] <= 2030)]
        df = df[(df['popular'] >= 1) & (df['popular'] <= 5)]
        
        return df


# Enhanced conversation memory management
class ConversationMemory:
    def __init__(self, max_history: int = 20):
        self.memory: Dict[str, List[Dict]] = {}
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
    
    def add_conversation(self, user_id: str, user_query: str, response: str):
        """Add conversation to memory with automatic cleanup"""
        if user_id not in self.memory:
            self.memory[user_id] = []
        
        self.memory[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'response': response
        })
        
        # Keep only recent conversations
        if len(self.memory[user_id]) > self.max_history:
            self.memory[user_id] = self.memory[user_id][-self.max_history:]
    
    def get_context(self, user_id: str, context_length: int = 3) -> str:
        """Get recent conversation context"""
        if user_id not in self.memory or not self.memory[user_id]:
            return ""
        
        recent_conversations = self.memory[user_id][-context_length:]
        context = "Recent conversation:\n"
        
        for conv in recent_conversations:
            context += f"User: {conv['user_query']}\n"
            context += f"Assistant: {conv['response'][:150]}...\n\n"
        
        return context
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for user"""
        if user_id in self.memory:
            self.memory[user_id] = []
            self.logger.info(f"Cleared conversation for user {user_id}")


# Enhanced parameter extraction with context inheritance
class ParameterExtractor:
    def __init__(self, ai_service: AIService, movies_df: pd.DataFrame):
        self.ai_service = ai_service
        self.movies_df = movies_df
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        
        # Extract actual values from dataset for intelligent matching
        self.actual_genres = set(movies_df['genre'].dropna().unique())
        self.actual_age_groups = set(movies_df['age_group'].dropna().unique()) if 'age_group' in movies_df.columns else set()
        
        self.logger.info(f"Found {len(self.actual_genres)} unique genres in dataset")
        self.logger.info(f"Found {len(self.actual_age_groups)} unique age groups in dataset")
    
    def extract_parameters(self, query: str, context: str = "") -> Dict[str, Any]:
        """Extract parameters from query with context inheritance"""
        cache_key = f"{query}_{hash(context)}"
        
        if cache_key in self._cache:
            self.logger.debug("Using cached parameters")
            return self._cache[cache_key]
        
        try:
            if self.ai_service.is_available():
                params = self._extract_with_ai(query, context)
            else:
                params = self._extract_basic(query, context)
            
            # Always try to inherit context parameters when context is available
            if context:
                context_params = self._extract_context_parameters(context)
                # Merge context parameters (context provides defaults for missing values)
                for key, value in context_params.items():
                    if value is not None and params.get(key) is None:
                        params[key] = value
                        self.logger.info(f"Inherited {key} from context: {value}")
            
            self._cache[cache_key] = params
            return params
            
        except Exception as e:
            self.logger.error(f"Parameter extraction failed: {e}")
            return self._extract_basic(query, context)
    
    def _extract_context_parameters(self, context: str) -> Dict[str, Any]:
        """Extract parameters from conversation context"""
        params = {
            'age_group': None,
            'genre': None,
            'year_range': None,
            'country': None,
            'popular': None,
            'actor': None,
            'director': None,
            'description_keywords': None
        }
        
        context_lower = context.lower()
        
        # Look for user queries in context to extract parameters
        lines = context.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Extract from user queries specifically
            if 'user:' in line_lower:
                user_part = line_lower.split('user:')[-1].strip()
                
                # Check for age group in user queries
                if any(word in user_part for word in ['kids', 'children', 'family']):
                    params['age_group'] = 'Kids'
                elif any(word in user_part for word in ['teens', 'teenager']):
                    params['age_group'] = 'Teens'
                elif any(word in user_part for word in ['young adults', 'young adult']):
                    params['age_group'] = 'Young Adults'
                elif any(word in user_part for word in ['adults', 'adult']):
                    params['age_group'] = 'Adults'

        # Also check Assistant responses for age group context
        for line in lines:
            line_lower = line.lower()
            if 'assistant:' in line_lower:
                assistant_part = line_lower.split('assistant:')[-1].strip()
                
                # Check for age group in assistant responses
                if not params.get('age_group'):
                    if any(phrase in assistant_part for phrase in ['suitable for kids', 'for kids', 'children & family']):
                        params['age_group'] = 'Kids'
                    elif any(phrase in assistant_part for phrase in ['suitable for teens', 'for teens']):
                        params['age_group'] = 'Teens'
                    elif any(phrase in assistant_part for phrase in ['suitable for adults', 'for adults']):
                        params['age_group'] = 'Adults'
                    elif any(phrase in assistant_part for phrase in ['young adults', 'young adult']):
                        params['age_group'] = 'Young Adults'

        return params
    
    def _extract_with_ai(self, query: str, context: str) -> Dict[str, Any]:
        """Extract parameters using AI"""
        system_prompt = self._get_system_prompt()
        
        context_info = f"\n\nPrevious conversation context:\n{context}\n" if context else ""
        prompt = f"{system_prompt}{context_info}\nUser query: {query}"
        
        response = self.ai_service.generate_content(prompt)
        
        # Clean JSON response
        if response.startswith('```json'):
            response = response[7:-3].strip()
        elif response.startswith('```'):
            response = response[3:-3].strip()
        
        try:
            return json.loads(response)
        except:
            self.logger.warning("AI response was not valid JSON, using fallback")
            return self._extract_basic(query, context)
    
    def _extract_basic(self, query: str, context: str) -> Dict[str, Any]:
        """Fallback parameter extraction without AI"""
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
        
        query_lower = query.lower()
        
        # Extract age group
        params['age_group'] = self._extract_age_group(query_lower)
        
        # Extract genre using fuzzy matching with actual dataset values
        params['genre'] = self._extract_genre_from_dataset(query_lower)
        
        # Extract year range
        params['year_range'] = self._extract_year_range(query)
        
        # Extract keywords
        params['description_keywords'] = self._extract_keywords(query)
        
        return params
    
    def _extract_age_group(self, query_lower: str) -> Optional[str]:
        """Extract age group from query"""
        if any(indicator in query_lower for indicator in ['for kids', 'children', 'family']):
            return 'Kids'
        elif any(indicator in query_lower for indicator in ['for teens', 'teenagers', 'teen']):
            return 'Teens'
        elif any(indicator in query_lower for indicator in ['young adults', 'young adult']):
            return 'Young Adults'
        elif any(indicator in query_lower for indicator in ['adults', 'adult', 'grown up']):
            return 'Adults'
        return None
    
    def _extract_genre_from_dataset(self, query_lower: str) -> Optional[str]:
        """Extract genre using fuzzy matching with actual dataset values"""
        # Create mapping from common terms to actual dataset genres
        genre_keywords = {}
        
        # Build dynamic mapping based on actual genres in dataset
        for actual_genre in self.actual_genres:
            genre_lower = actual_genre.lower()
            
            # Direct match
            genre_keywords[genre_lower] = actual_genre
            
            # Common variations
            if 'romantic' in genre_lower:
                genre_keywords['romance'] = actual_genre
                genre_keywords['romantic'] = actual_genre
                genre_keywords['love'] = actual_genre
            elif 'action' in genre_lower:
                genre_keywords['action'] = actual_genre
            elif 'comed' in genre_lower:
                genre_keywords['comedy'] = actual_genre
                genre_keywords['funny'] = actual_genre
            elif 'drama' in genre_lower:
                genre_keywords['drama'] = actual_genre
            elif 'horror' in genre_lower:
                genre_keywords['horror'] = actual_genre
                genre_keywords['scary'] = actual_genre
            elif 'thriller' in genre_lower:
                genre_keywords['thriller'] = actual_genre
                genre_keywords['suspense'] = actual_genre
            elif 'family' in genre_lower or 'children' in genre_lower:
                genre_keywords['family'] = actual_genre
                genre_keywords['kids'] = actual_genre
        
        # Check for matches
        for keyword, actual_genre in genre_keywords.items():
            if keyword in query_lower:
                self.logger.info(f"Detected genre '{actual_genre}' from word '{keyword}'")
                return self._normalize_genre_for_filtering(actual_genre)
        
        return None
    
    def _normalize_genre_for_filtering(self, genre: str) -> str:
        """Normalize genre for filtering - convert actual dataset genre to search term"""
        genre_lower = genre.lower()
        
        if 'romantic' in genre_lower:
            return 'romance'
        elif 'action' in genre_lower:
            return 'action'
        elif 'comed' in genre_lower:
            return 'comedy'
        elif 'drama' in genre_lower:
            return 'drama'
        elif 'horror' in genre_lower:
            return 'horror'
        elif 'thriller' in genre_lower:
            return 'thriller'
        elif 'family' in genre_lower or 'children' in genre_lower:
            return 'family'
        
        return genre_lower
    
    def _extract_year_range(self, query: str) -> Optional[List[int]]:
        """Extract year range from query"""
        import re
        year_match = re.search(r'\b(19|20)(\d{2})\b', query)
        if year_match:
            year = int(year_match.group(0))
            return [year, year]
        return None
    
    def _extract_keywords(self, query: str) -> Optional[List[str]]:
        """Extract description keywords"""
        query_lower = query.lower()
        if 'about' in query_lower:
            start_pos = query_lower.find('about') + 5
            remaining = query[start_pos:].strip()
            words = [word for word in remaining.split() if len(word) > 2]
            return words[:5] if words else None
        return None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for AI parameter extraction"""
        return """You are a movie recommendation assistant that extracts search parameters from natural language queries.

IMPORTANT: Handle Hebrew text, English text, mixed Hebrew-English, and typos. Be extremely flexible with genre recognition and spelling variations.

CONTEXT HANDLING: When analyzing the current query, consider the previous conversation context to understand:
- Follow-up questions (e.g., "only from 2019" after asking for kids movies)
- Refinements (e.g., "something newer" or "more recent")
- Continuations (e.g., "and also" or "what about")
- References to previous recommendations

Extract the following information from the user's query and return as JSON:
- age_group: target age group ONLY if explicitly mentioned (Kids, Teens, Young Adults, Adults) - if not mentioned, use null
- genre: specific genre ONLY if explicitly mentioned - if not mentioned, use null
- year_range: [min_year, max_year] ONLY if years are explicitly mentioned - if not mentioned, use null
- popular: ONLY if user explicitly asks for popular/top movies (high, medium, low) - if not mentioned, use null
- description_keywords: array of keywords describing plot/story elements - if no plot description, use null
- intent: the main intent (recommend, check_suitability, filter, general_movie_question, off_topic)

Return JSON format only."""


# Enhanced movie filtering with better performance and dataset-aware filtering
class MovieFilter:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies = movies_df
        self.logger = logging.getLogger(__name__)
        
        # Build dynamic genre mapping from actual dataset
        self.genre_mapping = self._build_genre_mapping()
        self.logger.info(f"Built genre mapping for {len(self.genre_mapping)} genre variations")
    
    def _build_genre_mapping(self) -> Dict[str, str]:
        """Build genre mapping from actual dataset values"""
        mapping = {}
        actual_genres = self.movies['genre'].dropna().unique()
        
        for actual_genre in actual_genres:
            genre_lower = actual_genre.lower()
            
            # Direct mapping
            mapping[genre_lower] = actual_genre
            
            # Build variations
            if 'romantic' in genre_lower:
                mapping['romance'] = actual_genre
                mapping['romantic'] = actual_genre
                mapping['love'] = actual_genre
            elif 'action' in genre_lower:
                mapping['action'] = actual_genre
            elif 'comed' in genre_lower:
                mapping['comedy'] = actual_genre
                mapping['funny'] = actual_genre
            elif 'drama' in genre_lower:
                mapping['drama'] = actual_genre
            elif 'horror' in genre_lower:
                mapping['horror'] = actual_genre
                mapping['scary'] = actual_genre
            elif 'thriller' in genre_lower:
                mapping['thriller'] = actual_genre
                mapping['suspense'] = actual_genre
            elif 'family' in genre_lower or 'children' in genre_lower:
                mapping['family'] = actual_genre
                mapping['kids'] = actual_genre
        
        return mapping
    
    def filter_movies(self, params: Dict[str, Any], limit: int = 6) -> pd.DataFrame:
        """Filter movies based on parameters with optimized performance"""
        filtered = self.movies.copy()
        self.logger.info(f"Starting with {len(filtered)} movies")
        self.logger.info(f"Parameters: {params}")
        
        # Apply filters in order of selectivity (most selective first)
        if params.get('description_keywords'):
            filtered = self._filter_by_keywords(filtered, params['description_keywords'])
        
        if params.get('genre'):
            filtered = self._filter_by_genre(filtered, params['genre'])
        
        if params.get('year_range'):
            filtered = self._filter_by_year_range(filtered, params['year_range'])
        
        if params.get('age_group'):
            filtered = self._filter_by_age_group(filtered, params['age_group'])
        
        # Sort and return top results
        return self._sort_and_limit(filtered, params, limit)
    
    def _filter_by_keywords(self, df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        """Filter by description keywords"""
        if 'description' not in df.columns:
            return df
        
        df = df.copy()
        df['keyword_score'] = 0
        
        for keyword in keywords:
            if len(keyword) > 2:
                mask = df['description'].str.contains(keyword, case=False, na=False)
                df.loc[mask, 'keyword_score'] += 1
        
        return df[df['keyword_score'] > 0]
    
    def _filter_by_genre(self, df: pd.DataFrame, genre: str) -> pd.DataFrame:
        """Filter by genre using dataset-aware mapping"""
        self.logger.info(f"User requested genre: {genre}")
        
        # Convert genre to lowercase for matching
        genre_lower = genre.lower()
        
        # For romance/romantic, look for "Romantic Movies" specifically
        if genre_lower in ['romance', 'romantic']:
            genre_search = 'Romantic Movies'
            self.logger.info(f"Looking for genre: {genre_search}")
            filtered = df[df['genre'].str.contains(genre_search, case=False, na=False)]
            self.logger.info(f"After genre filtering: {len(filtered)} movies")
            return filtered
        
        # For other genres, use flexible matching
        genre_patterns = {
            'action': 'Action',
            'comedy': 'Comed',
            'drama': 'Drama',
            'horror': 'Horror',
            'thriller': 'Thriller',
            'family': 'Family',
            'animation': 'Animation'
        }
        
        search_pattern = genre_patterns.get(genre_lower, genre)
        self.logger.info(f"Looking for genre pattern: {search_pattern}")
        filtered = df[df['genre'].str.contains(search_pattern, case=False, na=False)]
        self.logger.info(f"After genre filtering: {len(filtered)} movies")
        return filtered
    
    def _filter_by_year_range(self, df: pd.DataFrame, year_range: List[int]) -> pd.DataFrame:
        """Filter by year range"""
        if len(year_range) == 2:
            min_year, max_year = year_range
            filtered = df[(df['released'] >= min_year) & (df['released'] <= max_year)]
            self.logger.info(f"After year filtering ({min_year}-{max_year}): {len(filtered)} movies")
            return filtered
        return df
    
    def _filter_by_age_group(self, df: pd.DataFrame, age_group: str) -> pd.DataFrame:
        """Filter by age group"""
        if 'age_group' in df.columns:
            self.logger.info(f"Filtering by age_group: {age_group}")
            filtered = df[df['age_group'] == age_group]
            self.logger.info(f"After age_group filtering: {len(filtered)} movies for {age_group}")
            return filtered
        return df
    
    def _sort_and_limit(self, df: pd.DataFrame, params: Dict[str, Any], limit: int) -> pd.DataFrame:
        """Sort results and limit to top N"""
        if df.empty:
            return df
        
        # Sort by relevance score if available, otherwise by popularity
        if 'keyword_score' in df.columns:
            df = df.sort_values(['keyword_score', 'popular', 'released'], 
                              ascending=[False, False, False])
        else:
            df = df.sort_values(['popular', 'released'], ascending=[False, False])
        
        return df.head(limit)


# Enhanced response generator
class ResponseGenerator:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, movies: pd.DataFrame, params: Dict[str, Any], 
                         original_query: str) -> str:
        """Generate appropriate response based on context"""
        if movies.empty:
            return self._generate_no_results_response(params)
        
        if self._is_analytical_question(original_query):
            return self._generate_analytical_response(movies, original_query)
        else:
            return self._generate_recommendation_response(movies, params)
    
    def _is_analytical_question(self, query: str) -> bool:
        """Check if query requires analysis vs new search"""
        analytical_keywords = ['which', 'what', 'how', 'tell me about', 'analyze', 'best']
        return any(keyword in query.lower() for keyword in analytical_keywords)
    
    def _generate_recommendation_response(self, movies: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Generate movie recommendation response"""
        # Create personalized introduction
        intro = self._generate_personalized_intro(params)
        
        # Generate formatted movie list
        movies_list = []
        for _, movie in movies.head(6).iterrows():
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
            movies_list.append(f"• {movie['name']} ({year}) - {genre}")
        
        return intro + "\n" + "\n".join(movies_list)
    
    def _generate_personalized_intro(self, params: Dict[str, Any]) -> str:
        """Generate personalized introduction based on search parameters"""
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
                'low': 'lesser-known'
            }
            pop_text = popularity_map.get(params['popular'], params['popular'])
            intro_parts.append(f"that are {pop_text}")
        
        # Build the intro
        if intro_parts:
            intro = "Here are " + " ".join(intro_parts) + ":"
        else:
            intro = "Here are some movie recommendations:"
        
        return intro
    
    def _generate_analytical_response(self, movies: pd.DataFrame, query: str) -> str:
        """Generate analytical response using AI or fallback"""
        try:
            if self.ai_service.is_available():
                return self._generate_ai_analysis(movies, query)
            else:
                return self._generate_basic_analysis(movies, query)
        except Exception as e:
            self.logger.error(f"Analysis generation failed: {e}")
            return self._generate_basic_analysis(movies, query)
    
    def _generate_ai_analysis(self, movies: pd.DataFrame, query: str) -> str:
        """Generate AI-powered analysis"""
        movies_summary = self._create_movies_summary(movies)
        
        prompt = f"""Analyze these movies and answer the user's question: "{query}"

Movies data:
{movies_summary}

Provide a helpful analysis based on the data. Be specific and reference actual movie details."""
        
        return self.ai_service.generate_content(prompt)
    
    def _generate_basic_analysis(self, movies: pd.DataFrame, query: str) -> str:
        """Generate basic analysis without AI"""
        if movies.empty:
            return "I couldn't find any movies matching your criteria."
        
        # Basic statistics
        total_movies = len(movies)
        avg_rating = movies['popular'].mean() if 'popular' in movies.columns else None
        
        response = f"I found {total_movies} movies that match your criteria.\n\n"
        
        if avg_rating:
            response += f"Average rating: {avg_rating:.1f}/5.0\n"
        
        # List top movies
        response += "Top recommendations:\n"
        for i, (_, movie) in enumerate(movies.head(5).iterrows(), 1):
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            rating = movie['popular'] if pd.notna(movie['popular']) else 'N/A'
            response += f"{i}. {movie['name']} ({year}) - Rating: {rating}/5.0\n"
        
        return response
    
    def _generate_no_results_response(self, params: Dict[str, Any]) -> str:
        """Generate response when no movies found"""
        criteria = []
        if params.get('genre'):
            criteria.append(f"genre: {params['genre']}")
        if params.get('age_group'):
            criteria.append(f"age group: {params['age_group']}")
        if params.get('year_range'):
            criteria.append(f"year: {params['year_range']}")
        
        if criteria:
            criteria_text = ", ".join(criteria)
            return f"I couldn't find any movies matching your criteria ({criteria_text}). Try adjusting your search parameters."
        else:
            return "I couldn't find any movies matching your search. Please try a different query."
    
    def _create_movies_summary(self, movies: pd.DataFrame) -> str:
        """Create a summary of movies for AI analysis"""
        summary = ""
        for _, movie in movies.head(10).iterrows():
            year = int(movie['released']) if pd.notna(movie['released']) else 'Unknown'
            rating = movie['popular'] if pd.notna(movie['popular']) else 'N/A'
            genre = movie['genre'] if pd.notna(movie['genre']) else 'Unknown'
            summary += f"- {movie['name']} ({year}): {genre}, Rating: {rating}/5.0\n"
        return summary


# Main MovieBot class that orchestrates everything
class AdvancedMovieBot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL)
        
        # Initialize components
        self.ai_service = self._initialize_ai_service()
        self.movies_df = self._load_and_validate_data()
        self.conversation_memory = ConversationMemory(config.MAX_CONVERSATION_HISTORY)
        self.parameter_extractor = ParameterExtractor(self.ai_service, self.movies_df)
        self.movie_filter = MovieFilter(self.movies_df)
        self.response_generator = ResponseGenerator(self.ai_service)
        
        self.logger.info("Advanced MovieBot initialized successfully")
    
    def _initialize_ai_service(self) -> AIService:
        """Initialize AI service"""
        api_key = os.environ.get('GEMINI_API_KEY')
        return GeminiService(api_key, self.config.GEMINI_MODEL)
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate movie data"""
        try:
            # Try different encodings
            for encoding in ['latin-1', 'utf-8', 'cp1252']:
                try:
                    df = pd.read_csv(self.config.CSV_FILE_PATH, encoding=encoding)
                    self.logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DataLoadException("Failed to load CSV with any encoding")
            
            # Validate and clean data
            df = DataValidator.validate_movie_data(df)
            self.logger.info(f"Successfully loaded and validated {len(df)} movies")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load movie data: {e}")
            raise DataLoadException(f"Data loading failed: {e}")
    
    def get_recommendation(self, user_query: str, user_id: str) -> str:
        """Main method to get movie recommendations"""
        try:
            self.logger.info(f"Processing query from user {user_id}: {user_query}")
            
            # Get conversation context
            context = self.conversation_memory.get_context(user_id)
            
            # Extract parameters
            params = self.parameter_extractor.extract_parameters(user_query, context)
            
            # Filter movies
            filtered_movies = self.movie_filter.filter_movies(params, self.config.DEFAULT_MOVIES_LIMIT)
            
            # Generate response
            response = self.response_generator.generate_response(filtered_movies, params, user_query)
            
            # Save conversation
            self.conversation_memory.add_conversation(user_id, user_query, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing recommendation: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."


# Flask application setup
def create_app() -> Flask:
    """Create and configure Flask application"""
    config = Config()
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY
    
    # Initialize rate limiter
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[config.RATE_LIMIT]
    )
    
    # Initialize MovieBot
    movie_bot = AdvancedMovieBot(config)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/recommend', methods=['POST'])
    @limiter.limit("10 per minute")
    def recommend():
        try:
            data = request.get_json()
            user_query = data.get('query', '').strip()
            
            if not user_query:
                return jsonify({'error': 'Empty query'}), 400
            
            # Get or create user ID
            user_id = session.get('user_id')
            if not user_id:
                user_id = str(uuid.uuid4())
                session['user_id'] = user_id
            
            # Get recommendation
            response = movie_bot.get_recommendation(user_query, user_id)
            
            return jsonify({
                'response': response,
                'user_id': user_id
            })
            
        except Exception as e:
            movie_bot.logger.error(f"Error in recommend endpoint: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/clear_conversation', methods=['POST'])
    def clear_conversation():
        try:
            user_id = session.get('user_id')
            if user_id:
                movie_bot.conversation_memory.clear_conversation(user_id)
                return jsonify({'message': 'Conversation cleared'})
            else:
                return jsonify({'error': 'No active session'}), 400
        except Exception as e:
            movie_bot.logger.error(f"Error clearing conversation: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return app


# Main execution
if __name__ == '__main__':
    """Main entry point"""
    try:
        print("=" * 50)
        print("Advanced Movie Recommendation Chatbot")
        print("=" * 50)
        print("Initializing Advanced Movie Recommendation System...")
        
        app = create_app()
        
        print("System initialized successfully!")
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        exit(1)