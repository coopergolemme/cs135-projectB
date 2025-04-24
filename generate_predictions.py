import numpy as np, pandas as pd, joblib
from surprise import Dataset, Reader

# 1. Load the trained model and the *same* reader
model = joblib.load("models/svdpp_best2.joblib")
reader = Reader(rating_scale=(1, 5))

# 2. Leaderboard pairs
ratings_df = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")

print("ratings shape:", ratings_df.shape)
print("ratings head:\n", ratings_df.head())

ratings_df['user_id'] = ratings_df['user_id'].astype(str)
ratings_df['item_id'] = ratings_df['item_id'].astype(str)

ratings_df['svd_pred'] = ratings_df.apply(
    lambda row: model.predict(str(row['user_id']), str(row['item_id'])).est, axis=1
)

ratings_df.describe()

ratings_df["svd_pred"].to_csv("predicted_ratings_leaderboard.txt",
                       index=False, header=False)
print("Submission file written.")
