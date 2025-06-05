import sqlite3
import json
from datetime import datetime

class WatchlistDB:
    def __init__(self, db_path='watchlist.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create watchlist table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                movie_title TEXT NOT NULL,
                movie_year INTEGER,
                genre TEXT,
                added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'want_to_watch',
                rating INTEGER,
                notes TEXT,
                UNIQUE(user_id, movie_title, movie_year)
            )
        ''')
        
        # Create users table for simple user management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, user_id, username):
        """Add a new user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO users (user_id, username, created_date)
                VALUES (?, ?, ?)
            ''', (user_id, username, datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding user: {e}")
            return False
        finally:
            conn.close()
    
    def add_to_watchlist(self, user_id, movie_title, movie_year=None, genre=None):
        """Add a movie to user's watchlist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO watchlist 
                (user_id, movie_title, movie_year, genre, added_date, status)
                VALUES (?, ?, ?, ?, ?, 'want_to_watch')
            ''', (user_id, movie_title, movie_year, genre, datetime.now().isoformat()))
            
            if cursor.rowcount > 0:
                conn.commit()
                return True
            else:
                return False  # Movie already in watchlist
        except Exception as e:
            print(f"Error adding to watchlist: {e}")
            return False
        finally:
            conn.close()
    
    def get_watchlist(self, user_id, status=None):
        """Get user's watchlist, optionally filtered by status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status:
            cursor.execute('''
                SELECT movie_title, movie_year, genre, added_date, status, rating, notes
                FROM watchlist 
                WHERE user_id = ? AND status = ?
                ORDER BY added_date DESC
            ''', (user_id, status))
        else:
            cursor.execute('''
                SELECT movie_title, movie_year, genre, added_date, status, rating, notes
                FROM watchlist 
                WHERE user_id = ?
                ORDER BY added_date DESC
            ''', (user_id,))
        
        movies = cursor.fetchall()
        conn.close()
        
        return [
            {
                'title': movie[0],
                'year': movie[1],
                'genre': movie[2],
                'added_date': movie[3],
                'status': movie[4],
                'rating': movie[5],
                'notes': movie[6]
            }
            for movie in movies
        ]
    
    def update_movie_status(self, user_id, movie_title, movie_year, status, rating=None, notes=None):
        """Update movie status in watchlist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE watchlist 
                SET status = ?, rating = ?, notes = ?
                WHERE user_id = ? AND movie_title = ? AND movie_year = ?
            ''', (status, rating, notes, user_id, movie_title, movie_year))
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating movie status: {e}")
            return False
        finally:
            conn.close()
    
    def remove_from_watchlist(self, user_id, movie_title, movie_year):
        """Remove a movie from watchlist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM watchlist 
                WHERE user_id = ? AND movie_title = ? AND movie_year = ?
            ''', (user_id, movie_title, movie_year))
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing from watchlist: {e}")
            return False
        finally:
            conn.close()
    
    def get_watchlist_stats(self, user_id):
        """Get watchlist statistics for user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, COUNT(*) 
            FROM watchlist 
            WHERE user_id = ? 
            GROUP BY status
        ''', (user_id,))
        
        stats = dict(cursor.fetchall())
        conn.close()
        
        return stats