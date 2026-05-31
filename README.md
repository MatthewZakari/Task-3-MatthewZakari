# Project 3: AI Movie Recommendation System (Enhanced)

A sophisticated movie recommendation system that understands user preferences through direct input and historical ratings, matches patterns using cosine similarity, and maps interests to provide highly relevant suggestions.

## New Features
- **Preference Mapping**: The system builds a dynamic "Interest Map" based on movies you've rated 4 or 5 stars.
- **Rating System**: Rate movies from 1-5 stars. High ratings (4-5) improve recommendations, while any rated movie is filtered out of future suggestions to ensure you always see something new.
- **Hybrid Similarity Logic**: Combines your current mood (direct input) with your historical interest map to find the perfect match.
- **Persistent Storage**: Your ratings and profile are saved in `user_ratings.csv`, so the system "remembers" you across sessions.

## How it Works
1. **Direct Input**: You can tell the system what genres you want to see *right now*.
2. **Profile Influence**: The system automatically adds genres from your highly-rated movies to the search criteria.
3. **Similarity Engine**: Uses `CountVectorizer` and `Cosine Similarity` to compare your combined preferences against the database.
4. **Smart Filtering**: Already rated movies are removed from the recommendation list.

## Requirements
- Python 3.x
- pandas
- scikit-learn

## Commands
- `rec`: Get new recommendations.
- `rate`: Rate a movie in the database to refine your profile.
- `profile`: View your interest map and current ratings.
- `exit`: Quit the application.

## Setup & Usage
1.  Navigate to the project directory:
    ```bash
    cd DecodeLabs-Internship_project-3
    ```
2.  Install dependencies:
    ```bash
    pip install pandas scikit-learn
    ```
3.  Run the recommender:
    ```bash
    python3 recommender.py
    ```

## Example Usage
```text
Your Interest Map (Mapped Genres): Action Sci-Fi Adventure

Commands: 'rec' (get recommendations), 'rate' (rate a movie), 'profile' (view your mapping), 'exit'
Select an option: rec
Enter genres you're in the mood for (or press Enter to use only your profile): Drama

Finding matches based on your preferences and history...

Top Recommendations for you:
1. Interstellar
2. Blade Runner 2049
3. Arrival
4. Dune
5. The Dark Knight
```
