# file: src/train_movielens.py
import numpy as np, pandas as pd, joblib
from surprise import Dataset, Reader, SVDpp, accuracy
from surprise.model_selection import GridSearchCV, train_test_split


df = pd.read_csv("data_movie_lens_100k/ratings_all_development_set.csv")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)


param_grid = {
    "n_factors":  [40, 80, 120],
    "n_epochs":   [25, 50],
    "lr_all":     [0.002, 0.005],
    "reg_bu":     [0.02, 0.05],
    "reg_bi":     [0.02, 0.05],
    "reg_pu":     [0.06, 0.10],
    "reg_qi":     [0.06, 0.10],
}
gs = GridSearchCV(
        SVDpp,
        param_grid,
        measures=["mae"],
        cv=5,
        joblib_verbose=1,   # progress bar
        n_jobs=-1,
        refit=True          # keep the best model fitted on train+valid
)
gs.fit(data)

print(f"CV MAE: {gs.best_score['mae']:.4f}")
print("Best params:", gs.best_params["mae"])

best_algo = gs.best_estimator["mae"]


trainset = data.build_full_trainset()
best_algo.fit(trainset)

joblib.dump(best_algo, "models/svdpp_best.joblib")
