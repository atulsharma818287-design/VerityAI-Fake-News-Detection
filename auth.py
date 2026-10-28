from contextlib import closing
import sqlite3
from pathlib import Path

import bcrypt

DB_PATH = Path(__file__).resolve().parent / "users.db"
MAX_PASSWORD_BYTES = 72


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def create_users_table() -> None:
    with closing(get_connection()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()

    if len(username) < 3:
        return False, "Username must have at least 3 characters."

    if len(password) < 8:
        return False, "Password must have at least 8 characters."

    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False, "Password must be no more than 72 UTF-8 bytes."

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    )

    try:
        with closing(get_connection()) as conn, conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def verify_user(username: str, password: str) -> bool:
    username = username.strip().lower()

    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return False

    stored_hash = row[0]
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")

    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash)
    except ValueError:
        return False


create_users_table()
