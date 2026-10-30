import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import database as db
from recommender import get_recommender

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "barebone_hybrid_secret_key_981273")

# Initialize SQLite database
db.init_db()

# Warm up ML Recommender singleton
rec = get_recommender()


def get_current_user_context():
    """
    Returns (user_id, is_demo, username, favorite_genres, ratings_dict, watchlist_ids_set)
    """
    if "user_id" in session:
        uid = session["user_id"]
        user_data = db.get_user_by_id(uid)
        username = user_data["username"] if user_data else session.get("username", "User")
        fav_genres = user_data["favorite_genres"] if user_data else []
        ratings_dict = db.get_user_ratings_dict(uid)
        watchlist_ids = set(db.get_user_watchlist_ids(uid))
        return uid, False, username, fav_genres, ratings_dict, watchlist_ids
    elif "demo_user_id" in session:
        demo_id = session["demo_user_id"]
        username = f"Demo User #{demo_id}"
        user_row = rec.get_user_vector(user_id=demo_id)
        ratings_dict = {}
        for idx in range(len(user_row)):
            if not np.isnan(user_row[idx]):
                mid = rec.model["ids"][idx]
                ratings_dict[mid] = float(user_row[idx])
        demo_watchlist = set(session.get("demo_watchlist", [1, 2, 4]))
        return demo_id, True, username, ["Drama", "Action", "Romance"], ratings_dict, demo_watchlist

    return None, False, None, [], {}, set()


@app.route("/")
def index():
    uid, is_demo, username, fav_genres, ratings_dict, watchlist_ids = get_current_user_context()

    # Default to demo user 101 if not logged in so the app immediately shows rich recommendations
    if not uid and not is_demo:
        session["demo_user_id"] = 101
        session["username"] = "Demo User #101"
        session["demo_watchlist"] = [1, 2, 4]
        uid, is_demo, username, fav_genres, ratings_dict, watchlist_ids = get_current_user_context()

    current_alpha = request.args.get("alpha", default=rec.optimal_alpha, type=float)
    if not (0.0 <= current_alpha <= 1.0):
        current_alpha = rec.optimal_alpha

    shelves = rec.get_shelves(
        user_id=uid if is_demo else None,
        custom_ratings=None if is_demo else ratings_dict,
        favorite_genres=fav_genres,
        alpha=current_alpha,
        limit=8
    )

    # Watchlist items with predicted ratings
    user_row = rec.get_user_vector(user_id=uid if is_demo else None, custom_ratings_dict=None if is_demo else ratings_dict)
    watchlist_items = []
    for mid in watchlist_ids:
        meta = rec.movie_meta.get(mid)
        if meta:
            pred = rec.predict_detailed(user_row, mid, alpha=current_alpha)
            watchlist_items.append({
                "movieId": mid,
                "title": meta["title"],
                "year": meta["year"],
                "genres": meta["genres"],
                "poster_url": meta.get("poster_url", ""),
                "imdb_rating": meta.get("imdb_rating", 7.0),
                "predicted_rating": pred["hybrid_score"] if pred else 4.0
            })

    # User ratings items
    rated_items = []
    for mid, score in list(ratings_dict.items())[:20]:
        meta = rec.movie_meta.get(mid, {})
        rated_items.append({
            "movieId": mid,
            "title": meta.get("title", f"Movie #{mid}"),
            "year": meta.get("year", ""),
            "genres": meta.get("genres", []),
            "poster_url": meta.get("poster_url", ""),
            "rating": score
        })

    demo_profiles = [
        {"id": 101, "name": "User #101 (Classic Drama & Biography)", "count": 49},
        {"id": 205, "name": "User #205 (Action & Thriller)", "count": 38},
        {"id": 350, "name": "User #350 (Romance & Comedy)", "count": 42}
    ]

    return render_template(
        "index.html",
        username=username,
        is_demo=is_demo,
        uid=uid,
        ratings_dict=ratings_dict,
        watchlist_ids=list(watchlist_ids),
        watchlist_items=watchlist_items,
        rated_items=rated_items,
        shelves=shelves,
        current_alpha=current_alpha,
        optimal_alpha=rec.optimal_alpha,
        demo_profiles=demo_profiles,
        all_genres=rec.genre_list
    )


@app.route("/demo/<int:user_id>")
def demo_login(user_id):
    if user_id in rec.model["u"]:
        session.clear()
        session["demo_user_id"] = user_id
        session["username"] = f"Demo User #{user_id}"
        session["demo_watchlist"] = [1, 2, 4]
        flash(f"Switched to Demo User #{user_id}.", "info")
    else:
        flash("Demo user not found in dataset.", "error")
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = db.verify_user(username, password)
        if user:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("index"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    fav_genres = request.form.getlist("genres")
    fav_str = ",".join(fav_genres)

    uid, err = db.create_user(username, password, favorite_genres=fav_str)
    if err:
        flash(err, "error")
        return redirect(url_for("login"))

    session.clear()
    session["user_id"] = uid
    session["username"] = username
    flash("Account registered! You can now rate movies to get personalized recommendations.", "success")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    flash("Signed out.", "info")
    return redirect(url_for("index"))


@app.route("/rate", methods=["POST"])
def rate_movie():
    uid, is_demo, _, _, ratings_dict, _ = get_current_user_context()
    mid = request.form.get("movieId", type=int)
    rating = request.form.get("rating", type=float)

    if mid is not None and rating is not None:
        if is_demo:
            flash("Demo profile ratings are fixed. Create a free account to save your own ratings!", "info")
        else:
            db.add_or_update_rating(uid, mid, rating)
            flash(f"Rating {rating}★ saved!", "success")

    redirect_url = request.referrer or url_for("index")
    return redirect(redirect_url)


@app.route("/ratings/delete", methods=["POST"])
def delete_rating():
    uid, is_demo, _, _, _, _ = get_current_user_context()
    mid = request.form.get("movieId", type=int)
    if mid and not is_demo and uid:
        db.delete_rating(uid, mid)
        flash("Rating removed.", "info")
    return redirect(request.referrer or url_for("index"))


@app.route("/watchlist/toggle", methods=["POST"])
def toggle_watchlist():
    uid, is_demo, _, _, _, watchlist_ids = get_current_user_context()
    mid = request.form.get("movieId", type=int)

    if mid is not None:
        if is_demo:
            wl = session.get("demo_watchlist", [1, 2, 4])
            if mid in wl:
                wl.remove(mid)
                flash("Removed from watchlist.", "info")
            else:
                wl.append(mid)
                flash("Added to watchlist.", "success")
            session["demo_watchlist"] = wl
            session.modified = True
        else:
            if db.is_in_watchlist(uid, mid):
                db.remove_from_watchlist(uid, mid)
                flash("Removed from watchlist.", "info")
            else:
                db.add_to_watchlist(uid, mid)
                flash("Added to watchlist.", "success")

    return redirect(request.referrer or url_for("index"))


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = rec.search_movies(query, limit=20)
    uid, is_demo, username, _, ratings_dict, watchlist_ids = get_current_user_context()

    user_row = rec.get_user_vector(user_id=uid if is_demo else None, custom_ratings_dict=None if is_demo else ratings_dict)
    movies_with_preds = []
    for m in results:
        mid = m["movieId"]
        pred = None
        if mid not in ratings_dict:
            pred_obj = rec.predict_detailed(user_row, mid)
            pred = pred_obj["hybrid_score"] if pred_obj else None

        movies_with_preds.append({
            "movieId": mid,
            "title": m["title"],
            "year": m["year"],
            "genres": m["genres"],
            "director": m.get("director", ""),
            "actors": m.get("actors", ""),
            "poster_url": m.get("poster_url", ""),
            "imdb_rating": m.get("imdb_rating", 7.0),
            "is_rated": mid in ratings_dict,
            "user_rating": ratings_dict.get(mid),
            "in_watchlist": mid in watchlist_ids,
            "predicted_rating": pred
        })

    return render_template(
        "search.html",
        query=query,
        results=movies_with_preds,
        ratings_dict=ratings_dict,
        watchlist_ids=list(watchlist_ids)
    )


@app.route("/working")
@app.route("/algorithm")
def algorithm():
    uid, is_demo, username, _, ratings_dict, _ = get_current_user_context()

    movie_id = request.args.get("movie_id", default=1, type=int)
    if movie_id not in rec.model["i"]:
        movie_id = 1

    profile_param = request.args.get("profile", "current")
    if profile_param == "current" and (uid or is_demo):
        user_row = rec.get_user_vector(user_id=uid if is_demo else None, custom_ratings_dict=None if is_demo else ratings_dict)
        profile_label = f"Current User ({username})"
    elif profile_param.isdigit():
        demo_id = int(profile_param)
        user_row = rec.get_user_vector(user_id=demo_id)
        profile_label = f"Demo User #{demo_id}"
    else:
        user_row = rec.get_user_vector(user_id=101)
        profile_label = "Demo User #101"
        profile_param = "101"

    alpha = request.args.get("alpha", default=rec.optimal_alpha, type=float)
    if not (0.0 <= alpha <= 1.0):
        alpha = rec.optimal_alpha

    breakdown = rec.get_calculation_breakdown(user_row, movie_id, alpha=alpha)

    # Curated popular quick picks for inspection
    quick_picks = [
        {"movieId": 1, "title": "3 Idiots (2009)"},
        {"movieId": 2, "title": "Taare Zameen Par (2007)"},
        {"movieId": 4, "title": "Dangal (2016)"},
        {"movieId": 10, "title": "Gangs of Wasseypur (2012)"},
        {"movieId": 20, "title": "Dilwale Dulhania Le Jayenge (1995)"},
        {"movieId": 45, "title": "Andhadhun (2018)"}
    ]

    all_movies = [
        {"movieId": mid, "title": f"{m['title']} ({m['year']})"}
        for mid, m in sorted(rec.movie_meta.items(), key=lambda x: x[1]["title"])
    ]

    return render_template(
        "algorithm.html",
        breakdown=breakdown,
        movie_id=movie_id,
        current_alpha=alpha,
        optimal_alpha=rec.optimal_alpha,
        profile=profile_param,
        profile_label=profile_label,
        quick_picks=quick_picks,
        all_movies=all_movies
    )


# --- Minimal REST API Endpoints ---

@app.route("/api/predict/<int:movie_id>")
def api_predict(movie_id):
    uid, is_demo, _, _, ratings_dict, _ = get_current_user_context()
    alpha = request.args.get("alpha", default=rec.optimal_alpha, type=float)
    user_row = rec.get_user_vector(user_id=uid if is_demo else None, custom_ratings_dict=None if is_demo else ratings_dict)
    detailed = rec.predict_detailed(user_row, movie_id, alpha=alpha)
    if not detailed:
        return jsonify({"status": "error", "message": "Movie not found"}), 404
    return jsonify({"status": "ok", "prediction": detailed})


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    results = rec.search_movies(query, limit=10)
    return jsonify({"status": "ok", "count": len(results), "results": results})


@app.route("/api/rate", methods=["POST"])
def api_rate():
    uid, is_demo, _, _, _, _ = get_current_user_context()
    if not uid or is_demo:
        return jsonify({"status": "error", "message": "Sign in to a personal account to save ratings."}), 403

    data = request.get_json() or {}
    mid = data.get("movieId")
    rating = data.get("rating")
    if mid is None or rating is None:
        return jsonify({"status": "error", "message": "Missing movieId or rating"}), 400

    db.add_or_update_rating(uid, int(mid), float(rating))
    return jsonify({"status": "ok", "movieId": int(mid), "rating": float(rating)})


@app.route("/api/watchlist/toggle", methods=["POST"])
def api_watchlist_toggle():
    uid, is_demo, _, _, _, _ = get_current_user_context()
    data = request.get_json() or {}
    mid = data.get("movieId")
    if mid is None:
        return jsonify({"status": "error", "message": "Missing movieId"}), 400

    mid = int(mid)
    if is_demo:
        wl = session.get("demo_watchlist", [1, 2, 4])
        if mid in wl:
            wl.remove(mid)
            in_wl = False
        else:
            wl.append(mid)
            in_wl = True
        session["demo_watchlist"] = wl
        session.modified = True
    else:
        if db.is_in_watchlist(uid, mid):
            db.remove_from_watchlist(uid, mid)
            in_wl = False
        else:
            db.add_to_watchlist(uid, mid)
            in_wl = True

    return jsonify({"status": "ok", "movieId": mid, "in_watchlist": in_wl})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Barebone Hybrid Recommender on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
