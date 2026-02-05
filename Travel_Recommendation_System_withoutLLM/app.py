from flask import Flask, render_template,request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
import numpy as np
# app
app = Flask(__name__)

# Load datasets and models
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']
model = pickle.load(open('code and dataset/model.pkl','rb'))
label_encoders = pickle.load(open('code and dataset/label_encoders.pkl','rb'))

destinations_df = pd.read_csv("code and dataset/Expanded_Destinations.csv")
userhistory_df = pd.read_csv("code and dataset/Final_Updated_Expanded_UserHistory.csv")
df = pd.read_csv("code and dataset/final_df.csv")


# Collaborative Filtering Function
# Create a user-item matrix based on user history
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')

# Fill missing values with 0 (indicating no rating/experience)
user_item_matrix.fillna(0, inplace=True)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)


# Function to recommend destinations based on user similarity
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    try:
        # Check if user_id is within bounds of the similarity matrix
        if user_id < 1 or user_id > len(user_similarity):
            return destinations_df.head(5)  # Return default if ID is invalid

        # 1. Find similar users
        similar_users = user_similarity[user_id - 1]
        similar_users_idx = np.argsort(similar_users)[::-1][1:11]

        # 2. Get ratings and IDs
        similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
        recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(20).index

        # 3. Filter and Deduplicate (Ensure subset matches your CSV column name)
        recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)].copy()

        # Change 'Name' to 'Name_x' if that is what your CSV uses
        name_col = 'Name' if 'Name' in recommendations.columns else 'Name_x'
        recommendations = recommendations.drop_duplicates(subset=[name_col])

        # 4. Sorting by current month
        current_month = datetime.now().strftime('%b')
        recommendations['in_season'] = recommendations['BestTimeToVisit'].apply(
            lambda x: 1 if current_month in str(x) else 0
        )

        recommendations = recommendations.sort_values(
            by=['in_season', 'Popularity'],
            ascending=[False, False]
        ).head(5)

        return recommendations

    except Exception as e:
        print(f"Error in recommendation: {e}")
        return destinations_df.head(5)  # Fallback to avoid 500 error


@app.route('/')
def index():
    return render_template('index.html')
# Route for Travel Recommendation Page
@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')



def recommend_destinations(user_input, model, label_encoders, features):
    try:
        encoded_input = {}
        for feature in features:
            val = user_input.get(feature)
            if feature in label_encoders:
                # FIX: Handle unseen labels by defaulting to the first class or a 'safe' value
                try:
                    encoded_input[feature] = label_encoders[feature].transform([val])[0]
                except ValueError:
                    # If label is new/unseen, use the first available encoding
                    encoded_input[feature] = 0
            else:
                # Convert numeric strings to actual numbers
                encoded_input[feature] = pd.to_numeric(val, errors='coerce') or 0

        input_df = pd.DataFrame([encoded_input])
        return model.predict(input_df)[0]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0.0


@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if request.method == "POST":
        try:
            user_id = int(request.form.get('user_id', 1))

            user_input = {
                'Name_x': request.form.get('name'),
                'Type': request.form.get('type'),
                'State': request.form.get('state'),
                'BestTimeToVisit': request.form.get('best_time'),
                'Preferences': request.form.get('preferences'),
                'Gender': request.form.get('gender'),
                'NumberOfAdults': request.form.get('adults'),
                'NumberOfChildren': request.form.get('children'),
            }

            # 1. Get Collaborative Recommendations
            recs = collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df)

            # 2. Safety check: Ensure recs is a DataFrame and not empty
            if recs is not None and not recs.empty:
                # Round only if the column exists to prevent crash
                if 'Popularity' in recs.columns:
                    recs['Popularity'] = pd.to_numeric(recs['Popularity'], errors='coerce').fillna(0).round(2)

            # 3. Get Predicted Popularity
            pred_pop = recommend_destinations(user_input, model, label_encoders, features)
            pred_pop = round(float(pred_pop), 2)

            return render_template('recommendation.html',
                                   recommended_destinations=recs,
                                   predicted_popularity=pred_pop)

        except Exception as e:
            # This will catch errors in the route itself and print them to your terminal
            print(f"Route Error: {e}")
            return f"An error occurred: {e}", 500

    return render_template('recommendation.html')

if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)
