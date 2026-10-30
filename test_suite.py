import os
import unittest
import database as db
from recommender import get_recommender
from app import app


class TestBareboneRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(os.path.dirname(__file__), "test_users.db")
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        db.init_db(cls.test_db)
        db.DB_PATH = cls.test_db

        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret-key"
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        try:
            if os.path.exists(cls.test_db):
                os.remove(cls.test_db)
        except Exception:
            pass

    def test_01_database_operations(self):
        # 1. Create user
        uid, err = db.create_user("alice", "Password123!", "Drama,Comedy")
        self.assertIsNone(err)
        self.assertIsNotNone(uid)

        # 2. Duplicate username error
        uid_dup, err_dup = db.create_user("alice", "Pass2")
        self.assertIsNotNone(err_dup)

        # 3. Verify user
        user = db.verify_user("alice", "Password123!")
        self.assertIsNotNone(user)
        self.assertEqual(user["id"], uid)
        self.assertIn("Drama", user["favorite_genres"])

        # 4. Wrong password
        user_bad = db.verify_user("alice", "WrongPass")
        self.assertIsNone(user_bad)

        # 5. Add / update ratings
        db.add_or_update_rating(uid, 1, 5.0)  # 3 Idiots
        db.add_or_update_rating(uid, 4, 4.5)  # Dangal
        ratings = db.get_user_ratings_dict(uid)
        self.assertEqual(len(ratings), 2)
        self.assertEqual(ratings[1], 5.0)

        # 6. Watchlist CRUD
        self.assertFalse(db.is_in_watchlist(uid, 10))
        added = db.add_to_watchlist(uid, 10)
        self.assertTrue(added)
        self.assertTrue(db.is_in_watchlist(uid, 10))
        wl_ids = db.get_user_watchlist_ids(uid)
        self.assertIn(10, wl_ids)
        removed = db.remove_from_watchlist(uid, 10)
        self.assertTrue(removed)
        self.assertFalse(db.is_in_watchlist(uid, 10))

    def test_02_recommender_engine(self):
        rec = get_recommender()
        self.assertIsNotNone(rec.model)
        self.assertEqual(len(rec.movies), 600)

        # Search catalog
        search_res = rec.search_movies("Idiots")
        self.assertTrue(len(search_res) > 0)
        self.assertIn("Idiots", search_res[0]["title"])

        # Shelves generation
        custom_ratings = {1: 5.0, 4: 4.5}
        shelves = rec.get_shelves(custom_ratings=custom_ratings, favorite_genres=["Drama", "Comedy"], limit=5)
        self.assertIn("top_picks", shelves)
        self.assertIn("because_you_liked", shelves)
        self.assertIn("favorite_genres", shelves)
        self.assertIn("acclaimed", shelves)
        self.assertTrue(len(shelves["top_picks"]["items"]) > 0)

        # Detailed prediction breakdown
        user_row = rec.get_user_vector(custom_ratings_dict=custom_ratings)
        detailed = rec.predict_detailed(user_row, movie_id=2)  # Taare Zameen Par
        self.assertIsNotNone(detailed)
        self.assertIn("cf_score", detailed)
        self.assertIn("cb_score", detailed)
        self.assertIn("hybrid_score", detailed)
        self.assertIn("neighbors", detailed)

    def test_03_web_routes(self):
        # Home page
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"CineKNN", res.data)
        self.assertIn(b"Top Recommended Movies For You", res.data)

        # Demo profile switch
        res = self.client.get("/demo/101", follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Demo #101", res.data)

        # Search page
        res = self.client.get("/search?q=dangal")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Dangal", res.data)

        # Working / Calculation breakdown page
        res = self.client.get("/working?movie_id=1&alpha=0.5")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Working", res.data)
        self.assertIn(b"Bayesian Shrinkage Prior", res.data)

        # API Predict
        res = self.client.get("/api/predict/1")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("hybrid_score", data["prediction"])

        # API Search
        res = self.client.get("/api/search?q=sholay")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["count"] > 0)

        # API Watchlist toggle
        res = self.client.post("/api/watchlist/toggle", json={"movieId": 2})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
