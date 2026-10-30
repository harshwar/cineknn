import os
import re
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def unit_normalize(matrix):
    """Normalize row vectors to unit L2 length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HybridRecommender:
    """
    Dual-Branch Hybrid Recommender Engine:
    Combines Item-Item Collaborative Filtering KNN with Content-Based Genre Cosine KNN,
    using Bayesian Shrinkage Prior mean smoothing.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.ratings = None
        self.movies = None
        self.movie_meta = {}
        self.genre_list = []
        self.model = None
        self.optimal_alpha = 0.5
        self.load_and_train()

    def load_and_train(self):
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        movies_path = os.path.join(self.data_dir, "movies.csv")

        self.ratings = pd.read_csv(ratings_path).drop_duplicates(["userId", "movieId"])
        self.ratings["rating"] = self.ratings["rating"].fillna(self.ratings["rating"].mean())
        self.movies = pd.read_csv(movies_path)

        movie_counts = self.ratings.groupby("movieId").rating.count().to_dict()
        movie_means = self.ratings.groupby("movieId").rating.mean().to_dict()

        for _, row in self.movies.iterrows():
            mid = int(row["movieId"])
            title = str(row["title"])
            year_match = re.search(r"\((\d{4})\)", title)
            year = year_match.group(1) if year_match else str(row.get("year", ""))
            clean_title = re.sub(r"\s*\(\d{4}\)", "", title).strip()

            genres = [g.strip() for g in str(row["genres"]).split("|") if g.strip()] if pd.notna(row.get("genres")) else []

            self.movie_meta[mid] = {
                "movieId": mid,
                "title": clean_title,
                "full_title": title,
                "year": year,
                "genres": genres,
                "genre_str": "|".join(genres),
                "director": str(row.get("director", "")) if pd.notna(row.get("director")) else "",
                "actors": str(row.get("actors", "")) if pd.notna(row.get("actors")) else "",
                "overview": str(row.get("overview", "")) if pd.notna(row.get("overview")) else "",
                "poster_url": str(row.get("poster_url", "")) if pd.notna(row.get("poster_url")) else "",
                "imdb_rating": float(row.get("imdb_rating", 7.0)) if pd.notna(row.get("imdb_rating")) else 7.0,
                "count": movie_counts.get(mid, 0),
                "avg_rating": round(movie_means.get(mid, 4.0), 2)
            }

        # Multi-hot genre content matrix
        genres_df = self.movies.set_index("movieId").genres.str.get_dummies("|")
        self.genre_list = sorted(list(genres_df.columns))

        # Pivot interaction matrix (users x movies)
        ui = self.ratings.pivot(index="userId", columns="movieId", values="rating")
        col_movie_ids = list(ui.columns)
        g = self.ratings.groupby("movieId").rating
        gm = float(self.ratings.rating.mean())

        # Bayesian shrinkage mean: (sum(r) + 10 * gm) / (count + 10)
        counts = g.count().reindex(ui.columns).fillna(0).values
        sums = g.sum().reindex(ui.columns).fillna(0).values
        smoothed_means = (sums + 10.0 * gm) / (counts + 10.0)

        # Vector representations
        arr = ui.values
        cf_centered = np.nan_to_num(arr - smoothed_means).T
        cf_vectors = unit_normalize(cf_centered)
        cb_vectors = unit_normalize(genres_df.reindex(ui.columns).fillna(0).values.astype(float))

        self.model = {
            "arr": arr,
            "mean": smoothed_means,
            "gm": gm,
            "ids": col_movie_ids,
            "count": counts,
            "u": {u: i for i, u in enumerate(ui.index)},
            "i": {m: i for i, m in enumerate(col_movie_ids)},
            "cf": cf_vectors,
            "cb": cb_vectors
        }

        self.tune_alpha()

    def tune_alpha(self):
        M = self.model
        sample_df = self.ratings.sample(min(len(self.ratings), 1000), random_state=42).reset_index(drop=True)
        v_cf = np.full(len(sample_df), M["gm"])
        v_cb = np.full(len(sample_df), M["gm"])

        for u_id, group in sample_df.groupby("userId"):
            if u_id in M["u"]:
                user_row = M["arr"][M["u"][u_id]]
                c_indices = np.array([M["i"].get(m, -1) for m in group.movieId])
                valid_mask = c_indices >= 0
                if valid_mask.any():
                    valid_c = c_indices[valid_mask]
                    v_cf[group.index[valid_mask]] = self.predict_knn(user_row, valid_c, "cf")
                    v_cb[group.index[valid_mask]] = self.predict_knn(user_row, valid_c, "cb")

        y_true = sample_df.rating.values
        candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

        def rmse(pred):
            return np.sqrt(np.mean((y_true - np.clip(pred, 0.5, 5.0)) ** 2))

        self.optimal_alpha = min(candidates, key=lambda a: rmse(a * v_cf + (1.0 - a) * v_cb))

    def predict_knn(self, user_ratings_row, candidate_indices, kind, k=15):
        M = self.model
        user_ratings_row = np.atleast_1d(user_ratings_row)
        base = M["mean"][candidate_indices]
        rated_indices = np.where(~np.isnan(user_ratings_row))[0]

        if len(rated_indices) == 0:
            return base

        V = M[kind]
        S = V[candidate_indices] @ V[rated_indices].T  # Cosine similarity matrix

        kk = min(k, len(rated_indices))
        top_idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        s_top = np.take_along_axis(S, top_idx, axis=1)

        extra = (user_ratings_row[rated_indices] - M["mean"][rated_indices])[top_idx]
        denom = np.maximum(np.abs(s_top).sum(axis=1), 1e-9)
        return base + (s_top * extra).sum(axis=1) / denom

    def get_user_vector(self, user_id=None, custom_ratings_dict=None):
        M = self.model
        row = np.full(len(M["ids"]), np.nan)

        if user_id is not None and user_id in M["u"]:
            row = M["arr"][M["u"][user_id]].copy()

        if custom_ratings_dict:
            for mid, rating in custom_ratings_dict.items():
                if mid in M["i"]:
                    row[M["i"][mid]] = rating

        return row

    def get_recommendations_for_user(self, user_ratings_row, alpha=None, top_n=10, min_ratings=1):
        if alpha is None:
            alpha = self.optimal_alpha
        M = self.model

        rated_mask = ~np.isnan(user_ratings_row)
        candidates = np.where(~rated_mask & (M["count"] >= min_ratings))[0]
        if len(candidates) == 0:
            candidates = np.where(~rated_mask)[0]
        if len(candidates) == 0:
            return []

        p_cf = self.predict_knn(user_ratings_row, candidates, "cf")
        p_cb = self.predict_knn(user_ratings_row, candidates, "cb")
        hybrid_scores = np.clip(alpha * p_cf + (1 - alpha) * p_cb, 0.5, 5.0)

        top_order = np.argsort(-hybrid_scores)[:top_n]
        top_cand_indices = candidates[top_order]

        liked_indices = np.where(user_ratings_row >= 4.0)[0]
        results = []

        for rank, (cand_idx, score) in enumerate(zip(top_cand_indices, hybrid_scores[top_order]), start=1):
            mid = M["ids"][cand_idx]
            meta = self.movie_meta.get(mid, {})
            orig_pos = top_order[rank - 1]

            anchor_title = ""
            if len(liked_indices) > 0:
                sims = (alpha * (M["cf"][cand_idx] @ M["cf"][liked_indices].T) +
                        (1 - alpha) * (M["cb"][cand_idx] @ M["cb"][liked_indices].T))
                best_liked_idx = liked_indices[np.argmax(sims)]
                anchor_title = self.movie_meta.get(M["ids"][best_liked_idx], {}).get("title", "")

            results.append({
                "rank": rank,
                "movieId": mid,
                "title": meta.get("title", f"Movie {mid}"),
                "year": meta.get("year", ""),
                "genres": meta.get("genres", []),
                "genre_str": meta.get("genre_str", ""),
                "director": meta.get("director", ""),
                "actors": meta.get("actors", ""),
                "poster_url": meta.get("poster_url", ""),
                "imdb_rating": meta.get("imdb_rating", 7.0),
                "predicted_rating": round(float(score), 2),
                "cf_score": round(float(p_cf[orig_pos]), 2),
                "cb_score": round(float(p_cb[orig_pos]), 2),
                "avg_rating": meta.get("avg_rating", 4.0),
                "vote_count": int(M["count"][cand_idx]),
                "anchor": anchor_title
            })

        return results

    def predict_detailed(self, user_ratings_row, movie_id, alpha=None):
        if alpha is None:
            alpha = self.optimal_alpha
        M = self.model
        if movie_id not in M["i"]:
            return None

        cand_idx = M["i"][movie_id]
        meta = self.movie_meta.get(movie_id, {})
        user_ratings_row = np.atleast_1d(user_ratings_row)
        rated_indices = np.where(~np.isnan(user_ratings_row))[0]
        base_prior = float(M["mean"][cand_idx])

        if len(rated_indices) == 0:
            cf_score = base_prior
            cb_score = base_prior
            hybrid_score = base_prior
            neighbors = []
        else:
            p_cf_arr = self.predict_knn(user_ratings_row, [cand_idx], "cf")
            cf_score = float(np.clip(p_cf_arr[0], 0.5, 5.0))

            p_cb_arr = self.predict_knn(user_ratings_row, [cand_idx], "cb")
            cb_score = float(np.clip(p_cb_arr[0], 0.5, 5.0))

            hybrid_score = float(np.clip(alpha * cf_score + (1.0 - alpha) * cb_score, 0.5, 5.0))

            cf_sims = M["cf"][cand_idx] @ M["cf"][rated_indices].T
            cb_sims = M["cb"][cand_idx] @ M["cb"][rated_indices].T
            blended_sims = alpha * cf_sims + (1.0 - alpha) * cb_sims

            top_k_neighbors = min(4, len(rated_indices))
            top_neighbor_order = np.argsort(-blended_sims)[:top_k_neighbors]

            neighbors = []
            for n_pos in top_neighbor_order:
                n_idx = rated_indices[n_pos]
                n_mid = M["ids"][n_idx]
                n_meta = self.movie_meta.get(n_mid, {})
                sim_val = float(blended_sims[n_pos])
                if sim_val > 0.001:
                    neighbors.append({
                        "movieId": n_mid,
                        "title": n_meta.get("title", f"Movie {n_mid}"),
                        "user_rating": float(user_ratings_row[n_idx]),
                        "similarity": round(max(0.0, sim_val), 2),
                        "cf_sim": round(max(0.0, float(cf_sims[n_pos])), 2),
                        "cb_sim": round(max(0.0, float(cb_sims[n_pos])), 2)
                    })

        return {
            "movieId": movie_id,
            "title": meta.get("title", ""),
            "year": meta.get("year", ""),
            "genres": meta.get("genres", []),
            "genre_str": meta.get("genre_str", ""),
            "director": meta.get("director", ""),
            "actors": meta.get("actors", ""),
            "poster_url": meta.get("poster_url", ""),
            "imdb_rating": meta.get("imdb_rating", 7.0),
            "cf_score": round(cf_score, 2),
            "cb_score": round(cb_score, 2),
            "hybrid_score": round(hybrid_score, 2),
            "predicted_rating": round(hybrid_score, 2),
            "alpha": round(alpha, 2),
            "cf_weight_pct": int(round(alpha * 100)),
            "cb_weight_pct": int(round((1.0 - alpha) * 100)),
            "neighbors": neighbors,
            "avg_community_rating": round(base_prior, 2),
            "vote_count": int(M["count"][cand_idx])
        }

    def get_shelves(self, user_id=None, custom_ratings=None, favorite_genres=None, alpha=None, limit=8):
        if alpha is None:
            alpha = self.optimal_alpha
        M = self.model
        user_row = self.get_user_vector(user_id=user_id, custom_ratings_dict=custom_ratings)

        shelves = {}

        # 1. Top Picks
        top_hybrid = self.get_recommendations_for_user(user_row, alpha=alpha, top_n=limit)
        shelves["top_picks"] = {
            "title": "Top Recommended Movies For You",
            "subtitle": "Highest predicted ratings combining Collaborative and Content signals",
            "items": top_hybrid
        }

        # 2. Because You Liked
        liked_indices = np.where(user_row >= 4.0)[0]
        if len(liked_indices) > 0:
            highest_rated_idx = liked_indices[np.argmax(user_row[liked_indices])]
            anchor_mid = M["ids"][highest_rated_idx]
            anchor_title = self.movie_meta.get(anchor_mid, {}).get("title", "a favorite movie")

            unwatched = np.where(np.isnan(user_row))[0]
            if len(unwatched) > 0:
                sims = (alpha * (M["cf"][unwatched] @ M["cf"][highest_rated_idx]) +
                        (1 - alpha) * (M["cb"][unwatched] @ M["cb"][highest_rated_idx]))
                top_sim_order = np.argsort(-sims)[:limit]

                anchor_items = []
                for idx in top_sim_order:
                    cand_idx = unwatched[idx]
                    mid = M["ids"][cand_idx]
                    meta = self.movie_meta.get(mid, {})
                    anchor_items.append({
                        "movieId": mid,
                        "title": meta.get("title", ""),
                        "year": meta.get("year", ""),
                        "genres": meta.get("genres", []),
                        "poster_url": meta.get("poster_url", ""),
                        "imdb_rating": meta.get("imdb_rating", 7.0),
                        "avg_rating": meta.get("avg_rating", 4.0),
                        "similarity": round(float(sims[idx]), 2),
                        "anchor": anchor_title
                    })

                shelves["because_you_liked"] = {
                    "title": f"Because You Rated '{anchor_title}' Highly",
                    "subtitle": f"Titles with closest profile similarity to {anchor_title}",
                    "anchor": anchor_title,
                    "items": anchor_items
                }

        # 3. Top in Favorite Genres
        fav_genres = favorite_genres or []
        if fav_genres:
            unwatched = np.where(np.isnan(user_row))[0]
            matching_indices = [
                idx for idx in unwatched
                if any(g.lower() in [fg.lower() for fg in fav_genres] for g in self.movie_meta.get(M["ids"][idx], {}).get("genres", []))
            ]

            if matching_indices:
                cand_arr = np.array(matching_indices)
                p_cf = self.predict_knn(user_row, cand_arr, "cf")
                p_cb = self.predict_knn(user_row, cand_arr, "cb")
                scores = np.clip(alpha * p_cf + (1 - alpha) * p_cb, 0.5, 5.0)
                genre_order = np.argsort(-scores)[:limit]

                genre_items = []
                for idx in genre_order:
                    cand_idx = cand_arr[idx]
                    mid = M["ids"][cand_idx]
                    meta = self.movie_meta.get(mid, {})
                    genre_items.append({
                        "movieId": mid,
                        "title": meta.get("title", ""),
                        "year": meta.get("year", ""),
                        "genres": meta.get("genres", []),
                        "poster_url": meta.get("poster_url", ""),
                        "imdb_rating": meta.get("imdb_rating", 7.0),
                        "predicted_rating": round(float(scores[idx]), 2)
                    })

                shelves["favorite_genres"] = {
                    "title": f"Top in Your Favorite Genres ({', '.join(fav_genres[:3])})",
                    "subtitle": "Titles tailored to your chosen genres",
                    "items": genre_items
                }

        # 4. Top Rated Overall
        acclaimed_indices = np.where(M["count"] >= 10)[0]
        if len(acclaimed_indices) == 0:
            acclaimed_indices = np.arange(len(M["ids"]))
        acclaimed_order = np.argsort(-M["mean"][acclaimed_indices])[:limit]

        acclaimed_items = []
        for idx in acclaimed_order:
            cand_idx = acclaimed_indices[idx]
            mid = M["ids"][cand_idx]
            meta = self.movie_meta.get(mid, {})
            acclaimed_items.append({
                "movieId": mid,
                "title": meta.get("title", ""),
                "year": meta.get("year", ""),
                "genres": meta.get("genres", []),
                "poster_url": meta.get("poster_url", ""),
                "imdb_rating": meta.get("imdb_rating", 7.0),
                "avg_rating": round(float(M["mean"][cand_idx]), 2),
                "vote_count": int(M["count"][cand_idx])
            })

        shelves["acclaimed"] = {
            "title": "Top Rated Overall (Bayesian Mean)",
            "subtitle": "Highest rated classics across the entire catalog",
            "items": acclaimed_items
        }

        return shelves

    def search_movies(self, query: str, limit: int = 15):
        query = query.strip().lower()
        if not query:
            return []

        matches = []
        for mid, meta in self.movie_meta.items():
            if query in meta["title"].lower() or query in meta["genre_str"].lower():
                matches.append(meta)

        matches.sort(key=lambda x: x["count"], reverse=True)
        return matches[:limit]

    def get_calculation_breakdown(self, user_ratings_row, movie_id, alpha=None, k=15):
        """Full mathematical trace for predicting candidate movie rating."""
        if alpha is None:
            alpha = self.optimal_alpha
        M = self.model
        if movie_id not in M["i"]:
            return None

        cand_idx = M["i"][movie_id]
        meta = self.movie_meta.get(movie_id, {})
        user_ratings_row = np.atleast_1d(user_ratings_row)
        rated_indices = np.where(~np.isnan(user_ratings_row))[0]

        rating_count = int(M["count"][cand_idx])
        matching_ratings = self.ratings[self.ratings["movieId"] == movie_id]["rating"]
        rating_sum = float(matching_ratings.sum()) if len(matching_ratings) > 0 else 0.0
        raw_mean = float(matching_ratings.mean()) if len(matching_ratings) > 0 else M["gm"]
        prior_c = 10.0
        smoothed_baseline = float(M["mean"][cand_idx])

        genres_active = meta.get("genres", [])
        genre_norm = float(np.sqrt(len(genres_active))) if len(genres_active) > 0 else 1.0

        collab_dim = M["arr"].shape[0]

        dataset_metrics = {
            "total_movies": len(self.movies),
            "total_users": collab_dim,
            "total_ratings": len(self.ratings),
            "global_mean": round(M["gm"], 4),
            "optimal_alpha": self.optimal_alpha
        }

        bayesian_info = {
            "prior_c": prior_c,
            "global_mean": round(M["gm"], 4),
            "rating_count": rating_count,
            "rating_sum": round(rating_sum, 2),
            "raw_mean": round(raw_mean, 4),
            "smoothed_baseline": round(smoothed_baseline, 4)
        }

        if len(rated_indices) == 0:
            return {
                "target_movie": meta,
                "dataset_metrics": dataset_metrics,
                "bayesian_info": bayesian_info,
                "cf_score": round(smoothed_baseline, 2),
                "cb_score": round(smoothed_baseline, 2),
                "hybrid_score": round(smoothed_baseline, 2),
                "alpha": round(alpha, 2),
                "neighbors": []
            }

        cf_sims = M["cf"][cand_idx] @ M["cf"][rated_indices].T
        cb_sims = M["cb"][cand_idx] @ M["cb"][rated_indices].T
        hybrid_sims = alpha * cf_sims + (1.0 - alpha) * cb_sims

        kk = min(k, len(rated_indices))
        cf_top_idx = np.argpartition(-cf_sims, kk - 1)[:kk]
        cf_s_top = cf_sims[cf_top_idx]
        cf_extra = user_ratings_row[rated_indices[cf_top_idx]] - M["mean"][rated_indices[cf_top_idx]]
        cf_dev = (cf_s_top * cf_extra).sum() / max(np.abs(cf_s_top).sum(), 1e-9)
        cf_pred = float(np.clip(smoothed_baseline + cf_dev, 0.5, 5.0))

        cb_top_idx = np.argpartition(-cb_sims, kk - 1)[:kk]
        cb_s_top = cb_sims[cb_top_idx]
        cb_extra = user_ratings_row[rated_indices[cb_top_idx]] - M["mean"][rated_indices[cb_top_idx]]
        cb_dev = (cb_s_top * cb_extra).sum() / max(np.abs(cb_s_top).sum(), 1e-9)
        cb_pred = float(np.clip(smoothed_baseline + cb_dev, 0.5, 5.0))

        hyb_raw = alpha * cf_pred + (1.0 - alpha) * cb_pred
        hyb_final = float(np.clip(hyb_raw, 0.5, 5.0))

        # Top neighbors table
        sort_by_hyb = np.argsort(-hybrid_sims)[:10]
        neighbors_table = []
        for pos in sort_by_hyb:
            r_idx = rated_indices[pos]
            n_mid = M["ids"][r_idx]
            n_meta = self.movie_meta.get(n_mid, {})
            u_rating = float(user_ratings_row[r_idx])
            n_mean = float(M["mean"][r_idx])

            neighbors_table.append({
                "movieId": n_mid,
                "title": n_meta.get("title", f"Movie {n_mid}"),
                "genres": n_meta.get("genres", []),
                "user_rating": round(u_rating, 1),
                "item_baseline": round(n_mean, 2),
                "deviation": round(u_rating - n_mean, 2),
                "cf_sim": round(float(cf_sims[pos]), 3),
                "cb_sim": round(float(cb_sims[pos]), 3),
                "hybrid_sim": round(float(hybrid_sims[pos]), 3)
            })

        return {
            "target_movie": meta,
            "dataset_metrics": dataset_metrics,
            "bayesian_info": bayesian_info,
            "cf_score": round(cf_pred, 2),
            "cb_score": round(cb_pred, 2),
            "hybrid_score": round(hyb_final, 2),
            "alpha": round(alpha, 2),
            "cf_weight_pct": int(round(alpha * 100)),
            "cb_weight_pct": int(round((1.0 - alpha) * 100)),
            "neighbors": neighbors_table
        }


# Singleton pattern
_RECOMMENDER_INSTANCE = None


def get_recommender():
    global _RECOMMENDER_INSTANCE
    if _RECOMMENDER_INSTANCE is None:
        _RECOMMENDER_INSTANCE = HybridRecommender()
    return _RECOMMENDER_INSTANCE


if __name__ == "__main__":
    r = get_recommender()
    print(f"Recommender initialized! Catalog: {len(r.movie_meta)} movies, Optimal alpha: {r.optimal_alpha}")
    sample_shelves = r.get_shelves(user_id=101, limit=3)
    for k, v in sample_shelves.items():
        print(f"Shelf '{v['title']}': {len(v['items'])} items")
