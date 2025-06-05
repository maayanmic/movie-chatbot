import streamlit as st
import pandas as pd
import os
from movie_recommender import MovieRecommender

# Initialize the movie recommender
@st.cache_resource
def load_recommender():
    return MovieRecommender("sample_movies.csv")

def main():
    st.title("🎬 Movie Recommendation Chatbot")
    st.markdown("### Ask me for movie recommendations in Hebrew or English!")
    st.markdown("*Examples: 'יש סרטים מתאימים לילד בן 8?', 'What comedies are good for teenagers?'*")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load the recommender
    try:
        recommender = load_recommender()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about movies..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = recommender.get_recommendation(prompt)
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Sidebar with dataset info
        with st.sidebar:
            st.header("📊 Dataset Info")
            st.write(f"Total movies: {len(recommender.movies)}")
            
            # Show unique genres
            all_genres = set()
            for genres in recommender.movies['genre'].dropna():
                all_genres.update([g.strip() for g in str(genres).split(',')])
            st.write(f"Genres available: {len(all_genres)}")
            
            # Age range
            min_age = recommender.movies['min_age'].min()
            max_age = recommender.movies['min_age'].max()
            st.write(f"Age range: {min_age}-{max_age} years")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
                
    except Exception as e:
        st.error(f"Failed to load movie data: {str(e)}")
        st.info("Please make sure the CSV file exists and contains the required columns.")

if __name__ == "__main__":
    main()
