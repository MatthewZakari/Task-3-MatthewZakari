import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

class UserContext:
    """
    Manages user ratings and builds a preference profile based on high-rated movies.
    """
    def __init__(self, movies_df, ratings_file='user_ratings.csv'):
        self.movies_df = movies_df
        self.ratings_file = ratings_file
        self.ratings = self.load_ratings()

    def load_ratings(self):
        if os.path.exists(self.ratings_file):
            return pd.read_csv(self.ratings_file)
        return pd.DataFrame(columns=['title', 'rating'])

    def save_rating(self, movie_title, rating):
        # Update existing rating or add new one
        if movie_title in self.ratings['title'].values:
            self.ratings.loc[self.ratings['title'] == movie_title, 'rating'] = rating
        else:
            new_rating = pd.DataFrame({'title': [movie_title], 'rating': [rating]})
            self.ratings = pd.concat([self.ratings, new_rating], ignore_index=True)
        self.ratings.to_csv(self.ratings_file, index=False)

    def get_user_profile(self):
        """
        Maps user preferences by aggregating genres of highly-rated movies (rating >= 4).
        """
        high_rated = self.ratings[self.ratings['rating'] >= 4]
        if high_rated.empty:
            return ""
        
        # Merge with movies_df to get genres
        profile_movies = pd.merge(high_rated, self.movies_df, on='title')
        # Combine all genres into a single string
        return " ".join(profile_movies['genres'].tolist())

    def get_rated_movies(self):
        return self.ratings['title'].tolist()

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)
    return pd.read_csv(file_path)

def get_recommendations(user_input, user_context, df):
    """
    Matches user input + historical preferences with movies using cosine similarity.
    """
    # Combine explicit user input with historical profile
    history_profile = user_context.get_user_profile()
    combined_preferences = f"{user_input} {history_profile}".strip()
    
    if not combined_preferences:
        return []

    # Filter out already rated movies to suggest new ones
    rated_titles = user_context.get_rated_movies()
    available_movies = df[~df['title'].isin(rated_titles)].copy()
    
    if available_movies.empty:
        return []

    # Vectorize genres
    temp_df = pd.concat([available_movies, pd.DataFrame({'title': ['User'], 'genres': [combined_preferences]})], ignore_index=True)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(temp_df['genres'])
    
    # Calculate similarity
    similarity = cosine_similarity(count_matrix)
    user_idx = len(temp_df) - 1
    similarity_scores = list(enumerate(similarity[user_idx]))
    
    # Sort and get top matches
    sorted_movies = sorted(similarity_scores[:-1], key=lambda x: x[1], reverse=True)
    
    recommendations = []
    num_recs = min(5, len(sorted_movies))
    for i in range(num_recs):
        movie_idx = sorted_movies[i][0]
        recommendations.append(available_movies.iloc[movie_idx]['title'])
        
    return recommendations

def main():
    print("--- Enhanced Movie Recommendation System ---")
    movies_df = load_data('movies.csv')
    user_context = UserContext(movies_df)
    
    print("Database and User Profile loaded!")
    
    while True:
        print("\nCommands: 'rec' (get recommendations), 'rate' (rate a movie), 'profile' (view your mapping), 'exit'")
        choice = input("Select an option: ").strip().lower()

        if choice == 'exit':
            print("Goodbye!")
            break
        
        elif choice == 'rec':
            user_input = input("Enter genres you're in the mood for (or press Enter to use only your profile): ").strip()
            print("Finding matches based on your preferences and history...")
            recs = get_recommendations(user_input, user_context, movies_df)
            
            if not recs:
                print("No new recommendations found. Try rating more movies or entering different genres!")
            else:
                print("\nTop Recommendations for you:")
                for i, movie in enumerate(recs, 1):
                    print(f"{i}. {movie}")

        elif choice == 'rate':
            movie_name = input("Enter the movie title to rate: ").strip()
            # Basic partial match search
            matches = movies_df[movies_df['title'].str.contains(movie_name, case=False)]
            
            if matches.empty:
                print("Movie not found in database.")
            elif len(matches) > 1:
                print("Multiple matches found:")
                for i, t in enumerate(matches['title'], 1):
                    print(f"{i}. {t}")
                idx = int(input("Select the number: ")) - 1
                movie_name = matches.iloc[idx]['title']
            else:
                movie_name = matches.iloc[0]['title']

            try:
                rating = int(input(f"Rate '{movie_name}' (1-5 stars): "))
                if 1 <= rating <= 5:
                    user_context.save_rating(movie_name, rating)
                    print(f"Rating for '{movie_name}' saved!")
                else:
                    print("Please enter a rating between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == 'profile':
            profile = user_context.get_user_profile()
            print(f"\nYour Interest Map (Mapped Genres): {profile if profile else 'No high-rated movies yet.'}")
            if not user_context.ratings.empty:
                print("\nYour Current Ratings:")
                print(user_context.ratings)

        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()
