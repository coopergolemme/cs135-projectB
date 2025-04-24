import numpy as np, pandas as pd, joblib
from surprise import Dataset, Reader, SVDpp, SVD
from surprise.model_selection import cross_validate, GridSearchCV


reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)


train_set = Dataset.load_from_file(
    'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)


param_grid = {
    "n_factors": [50, 100, 200],
    "n_epochs" : [20, 50],
    "lr_all"   : [0.002, 0.005, 0.01],
    "reg_all"  : [0.02, 0.06, 0.1]
}
# 3 × 2 × 3 × 3 = 54 fits instead of 81, yet it explores every key dimension.

gs = GridSearchCV(
        SVD,
        param_grid,
        measures=["mae"],
        cv=3,
        joblib_verbose=1,   # progress bar
        n_jobs=-1,
        refit=True          # keep the best model fitted on train+valid
)
gs.fit(train_set)

print("Best MAE score:", gs.best_score["mae"])
print("Best params:", gs.best_params["mae"])


results_df = pd.DataFrame.from_dict(gs.cv_results)
results_df.to_csv("grid_search_results.csv", index=False)

best_algo = gs.best_estimator["mae"]

train_set = train_set.build_full_trainset()
best_algo.fit(train_set)

model_name = input("Enter model name (e.g., svdpp_best2): ")
joblib.dump(best_algo, f'models/{model_name}.joblib')
