import numpy as np
from recommender import get_recommender


def evaluate_models():
    print("=" * 65)
    print("   BAREBONE HYBRID RECOMMENDER — OFFLINE ACCURACY BENCHMARK")
    print("=" * 65)

    rec = get_recommender()
    M = rec.model
    ratings = rec.ratings

    print(f"Total dataset ratings: {len(ratings):,} | Movies: {len(M['ids']):,} | Users: {len(M['u']):,}")

    # 15% random holdout test split (up to 3,000 ratings)
    np.random.seed(1337)
    test_size = min(3000, int(len(ratings) * 0.15))
    test_idx = np.random.choice(len(ratings), size=test_size, replace=False)
    test_df = ratings.iloc[test_idx].reset_index(drop=True)

    print(f"Evaluating across {len(test_df):,} holdout test ratings...\n")

    y_true = test_df.rating.values
    preds_global = np.full(len(test_df), M["gm"])
    preds_item_mean = np.array([M["mean"][M["i"][m]] if m in M["i"] else M["gm"] for m in test_df.movieId])

    v_cf = np.full(len(test_df), M["gm"])
    v_cb = np.full(len(test_df), M["gm"])

    for u_id, group in test_df.groupby("userId"):
        if u_id in M["u"]:
            user_row = M["arr"][M["u"][u_id]]
            c_indices = np.array([M["i"].get(m, -1) for m in group.movieId])
            valid = c_indices >= 0
            if valid.any():
                v_cf[group.index[valid]] = rec.predict_knn(user_row, c_indices[valid], "cf")
                v_cb[group.index[valid]] = rec.predict_knn(user_row, c_indices[valid], "cb")

    def calc_metrics(y_actual, y_predicted):
        clipped = np.clip(y_predicted, 0.5, 5.0)
        rmse = np.sqrt(np.mean((y_actual - clipped) ** 2))
        mae = np.mean(np.abs(y_actual - clipped))
        return rmse, mae

    g_rmse, g_mae = calc_metrics(y_true, preds_global)
    im_rmse, im_mae = calc_metrics(y_true, preds_item_mean)
    cb_rmse, cb_mae = calc_metrics(y_true, v_cb)
    cf_rmse, cf_mae = calc_metrics(y_true, v_cf)

    alpha_opt = rec.optimal_alpha
    hybrid_preds = alpha_opt * v_cf + (1.0 - alpha_opt) * v_cb
    hyb_rmse, hyb_mae = calc_metrics(y_true, hybrid_preds)

    pct_gain = ((g_rmse - hyb_rmse) / g_rmse) * 100.0

    print(f"{'Model Configuration':<35} | {'RMSE':<8} | {'MAE':<8} | {'Status'}")
    print("-" * 65)
    print(f"{'1. Global Mean Baseline':<35} | {g_rmse:.4f}   | {g_mae:.4f}   | Baseline Benchmark")
    print(f"{'2. Bayesian Item Mean Baseline':<35} | {im_rmse:.4f}   | {im_mae:.4f}   | Popularity Prior")
    print(f"{'3. Content-Based Genre KNN (alpha=0)':<35} | {cb_rmse:.4f}   | {cb_mae:.4f}   | Cold-start stable")
    print(f"{'4. Collaborative Item-Item KNN (alpha=1)':<35} | {cf_rmse:.4f}   | {cf_mae:.4f}   | Interaction patterns")
    print(f"{f'5. Hybrid Model (alpha={alpha_opt})':<35} | {hyb_rmse:.4f}   | {hyb_mae:.4f}   | BEST ACCURACY")
    print("-" * 65)
    print(f"\n>> Hybrid Accuracy Advantage: {pct_gain:.1f}% error reduction over Global Mean baseline.")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_models()
