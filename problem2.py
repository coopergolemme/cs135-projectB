import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from surprise import SVD, KNNBasic
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV



def load_process_user_info():
    filename = 'data_movie_lens_100k/user_info.csv'
    user_info = pd.read_csv(filename)
    # standard scaler the age column
    user_info['age'] = StandardScaler().fit_transform(user_info[['age']])

    return user_info

def load_process_item_info():
    # item_id,title,release_year,orig_item_id
    item_df = pd.read_csv('data_movie_lens_100k/movie_info.csv')
    item_df['release_year'] = item_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    item_df['release_year'].fillna(item_df['release_year'].median(), inplace=True)
    year_scaler = StandardScaler()
    item_df['release_year_scaled'] = year_scaler.fit_transform(item_df[['release_year']])
    item_features = item_df[['item_id', 'release_year_scaled']]
    return item_features

def load_all_data():
    # Load the user and item info
    user_info = load_process_user_info()
    item_info = load_process_item_info()

    # Load the ratings data
    ratings_df = pd.read_csv('data_movie_lens_100k/ratings_all_development_set.csv')
    merged_df = ratings_df.merge(user_info, on='user_id').merge(item_info, on='item_id')


    return merged_df




def train_svd(train_df):
    reader = Reader(rating_scale=(1,5))
    train_data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
    train_set = train_data.build_full_trainset()
    
    # Perform cross-validation using the SVD algorithm
    param_grid = {"n_factors": [100, 200],
                  "n_epochs": [10, 25],
                  "lr_all": [0.005, 0.01],
                  "reg_all": [0.01, 0.1]}
    
    # gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
    # gs.fit(train_data)

    # print("Best RMSE score: ", gs.best_score["rmse"])
    # print("Best parameters: ", gs.best_params["rmse"])
    # print("Best MAE score: ", gs.best_score["mae"])

    # # Fit the best model on the entire dataset using the best parameters
    # best_model = gs.best_estimator["rmse"]
    # best_model.fit(train_set)

    best_params = {'n_factors': 100, 'n_epochs': 25, 'lr_all': 0.01, 'reg_all': 0.1}
    best_model = SVD(**best_params);
    best_model.fit(train_set)
    print("SVD model trained with parameters: ", best_params)
    return best_model


def generate_SVD_predictions(row, model, train_set):
    try:
        u = train_set.to_inner_uid(row['user_id'])
        i = train_set.to_inner_iid(row['item_id'])
        pred = model.bu[u] + model.bi[i] + model.pu[u].dot(model.qi[i]) + train_set.global_mean
    except ValueError:
        print("User or item not found in training set. Using global mean.")
        pred = train_set.global_mean
    return np.clip(pred, 1, 5)


def train_XGBoost(SVD_model, train_df, train_set, feature_cols):
    # reader = Reader(rating_scale=(1,5))
    # train_data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
    # train_set = train_data.build_full_trainset()

    train_df['svd_pred'] = train_df.apply(lambda row: generate_SVD_predictions(row, SVD_model, train_set), axis=1)
    train_df['residual'] = train_df['rating'] - train_df['svd_pred']

    X_train = train_df[feature_cols]
    y_train = train_df['residual']
    print(X_train.head())
    print(y_train.head())

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    best_params = {'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}

    # Uncomment the below code to perform grid search

    # xgb = XGBRegressor(objective='reg:absoluteerror', random_state=42)
    # grid_search = GridSearchCV(
    #     estimator=xgb,
    #     param_grid=param_grid,
    #     scoring='neg_mean_absolute_error',
    #     cv=3,
    #     verbose=1,
    #     n_jobs=-1
    # )

    # grid_search.fit(X_train, y_train)

    # print(f"Best MAE: {-grid_search.best_score_}")
    # print(f"Best Parameters: {grid_search.best_params_}")

    # # Train the final model with the best parameters
    # best_xgb = grid_search.best_estimator_
    # best_xgb.fit(X_train, y_train)

    # commment out the above code and uncomment the below code to use the best parameters directly
    best_xgb = XGBRegressor(**best_params, objective='reg:absoluteerror', random_state=42)
    best_xgb.fit(X_train, y_train)

    print("XGBoost model trained with parameters: ", best_params)

    return best_xgb

def train_whole_data():
    train_df = load_all_data()
    excluded = ['user_id', 'item_id', 'rating', 'svd_pred', 'residual']
    feature_cols = [col for col in train_df.columns if col not in excluded]

    svd = train_svd(train_df)
    
    reader = Reader(rating_scale=(1,5))
    train_data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
    train_set = train_data.build_full_trainset()

    xgb = train_XGBoost(svd, train_df, train_set, feature_cols)

    return svd, xgb

def generate_final_predictions(model, user_features, item_features, final_svd, full_trainset, feature_cols):


    filename = 'data_movie_lens_100k/ratings_masked_leaderboard_set.csv'
    leaderboard_df = pd.read_csv(filename)
    leaderboard_merged = leaderboard_df.merge(user_features, on='user_id').merge(item_features, on='item_id')
    leaderboard_merged['svd_pred'] = leaderboard_merged.apply(lambda row: generate_SVD_predictions(row, final_svd, full_trainset), axis=1)
    X_leader = leaderboard_merged[feature_cols]
    leaderboard_residuals = model.predict(X_leader)
    leaderboard_preds = leaderboard_merged['svd_pred'] + leaderboard_residuals

    # Save predictions
    leaderboard_df['rating'] = leaderboard_preds
    leaderboard_df['rating'].to_csv('predicted_ratings_leaderboard.txt', index=False, header=False)

    


if __name__ == "__main__":
    train_df = load_all_data()
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    excluded = ['user_id', 'item_id', 'rating', 'svd_pred', 'residual', 'orig_item_id', 'orig_user_id']
    feature_cols = [col for col in train_df.columns if col not in excluded]

    svd = train_svd(train_df)
    

    reader = Reader(rating_scale=(1,5))
    train_data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
    train_set = train_data.build_full_trainset()

    xgb = train_XGBoost(svd, train_df, train_set, feature_cols)



    # # Generate predictions for the validation set
    valid_df['svd_pred'] = valid_df.apply(lambda row: generate_SVD_predictions(row, svd, train_set), axis=1)
    valid_df['xgb_pred'] = xgb.predict(valid_df[feature_cols])
    valid_df['final_pred'] = valid_df['svd_pred'] + valid_df['xgb_pred']
    valid_df['final_pred'] = np.clip(valid_df['final_pred'], 1, 5)
    mae = mean_absolute_error(valid_df['rating'], valid_df['final_pred'])
    print(f"Validation MAE: {mae}")
    # svd, xgb = train_whole_data();
    # generate_final_predictions(xgb,
    #                             load_process_user_info(),
    #                             load_process_item_info(),
    #                             svd,
    #                             Dataset.load_from_df(load_all_data()[['user_id', 'item_id', 'rating']], Reader(rating_scale=(1,5))).build_full_trainset(),
    #                             feature_cols)




    