import sqlite3
import hashlib
import secrets
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def get_connection(db_path=None):
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                favorite_genres TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL CHECK(rating >= 0.5 AND rating <= 5.0),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, movie_id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_ratings_user ON user_ratings(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_ratings_movie ON user_ratings(movie_id);")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, movie_id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_user ON user_watchlist(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_movie ON user_watchlist(movie_id);")
        conn.commit()


def hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return hashed, salt


def create_user(username: str, password: str, favorite_genres: str = ""):
    username = username.strip()
    if not username or not password:
        return None, "Username and password cannot be empty."

    hashed, salt = hash_password(password)
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt, favorite_genres) VALUES (?, ?, ?, ?)",
                (username, hashed, salt, favorite_genres)
            )
            conn.commit()
            return cursor.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "Username already exists. Please choose a different one."
    except Exception as e:
        return None, str(e)


def verify_user(username: str, password: str):
    username = username.strip()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return None
        hashed, _ = hash_password(password, row["salt"])
        if hashed == row["password_hash"]:
            genres = [g.strip() for g in row["favorite_genres"].split(",") if g.strip()]
            return {
                "id": row["id"],
                "username": row["username"],
                "favorite_genres": genres
            }
    return None


def get_user_by_id(user_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            genres = [g.strip() for g in row["favorite_genres"].split(",") if g.strip()]
            return {
                "id": row["id"],
                "username": row["username"],
                "favorite_genres": genres
            }
    return None


def update_user_genres(user_id: int, genres: list):
    genres_str = ",".join(genres)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET favorite_genres = ? WHERE id = ?", (genres_str, user_id))
        conn.commit()


def add_or_update_rating(user_id: int, movie_id: int, rating: float):
    rating = max(0.5, min(5.0, round(float(rating) * 2) / 2))
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_ratings (user_id, movie_id, rating)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, movie_id) DO UPDATE SET rating = excluded.rating, created_at = CURRENT_TIMESTAMP
        """, (user_id, movie_id, rating))
        conn.commit()
        return True


def get_user_ratings(user_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT movie_id, rating, created_at 
            FROM user_ratings 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_user_ratings_dict(user_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id, rating FROM user_ratings WHERE user_id = ?", (user_id,))
        return {row["movie_id"]: row["rating"] for row in cursor.fetchall()}


def delete_rating(user_id: int, movie_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_ratings WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        conn.commit()
        return cursor.rowcount > 0


def add_to_watchlist(user_id: int, movie_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO user_watchlist (user_id, movie_id)
                VALUES (?, ?)
                ON CONFLICT(user_id, movie_id) DO NOTHING
            """, (user_id, movie_id))
            conn.commit()
            return True
        except Exception:
            return False


def remove_from_watchlist(user_id: int, movie_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_watchlist WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        conn.commit()
        return cursor.rowcount > 0


def get_user_watchlist_ids(user_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id FROM user_watchlist WHERE user_id = ? ORDER BY added_at DESC", (user_id,))
        return [row["movie_id"] for row in cursor.fetchall()]


def is_in_watchlist(user_id: int, movie_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM user_watchlist WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        return cursor.fetchone() is not None


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
