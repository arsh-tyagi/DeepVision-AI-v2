"""
database.py
===========
SQLite database layer for DeepVision-AI.
Handles user authentication (signup/login) and image analysis history.
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "deepvision.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_transaction():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    with db_transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                image_data BLOB,
                ai_probability REAL NOT NULL,
                real_probability REAL NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                exif_summary TEXT,
                analysis_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_created_at ON history(created_at DESC)")

def hash_password(password: str) -> str:
    salt = "deepvision_ai_salt_2026"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    if not username or not email or not password:
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    with db_transaction() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username.strip(), email.strip().lower(), hash_password(password))
            )
            return True, "Account created successfully! Please log in."
        except sqlite3.IntegrityError as e:
            if "username" in str(e).lower(): return False, "Username already taken."
            elif "email" in str(e).lower(): return False, "Email already registered."
            return False, "Registration failed."

def login_user(username: str, password: str) -> tuple[bool, str, Optional[Dict]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        user = cursor.fetchone()
    if not user: return False, "Invalid username or password.", None
    if not verify_password(password, user["password_hash"]): return False, "Invalid username or password.", None
    return True, "Login successful!", dict(user)

def delete_user(user_id: int) -> tuple[bool, str]:
    with db_transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cursor.rowcount == 0: return False, "User not found."
    return True, "Account deleted successfully."

def save_analysis(user_id: int, image_name: str, image_data: bytes, ai_prob: float, real_prob: float, verdict: str, confidence: float, exif_summary: str, analysis_text: str) -> int:
    with db_transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO history (user_id, image_name, image_data, ai_probability, real_probability, verdict, confidence, exif_summary, analysis_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, image_name, image_data, ai_prob, real_prob, verdict, confidence, exif_summary, analysis_text))
        return cursor.lastrowid

def get_user_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, image_name, image_data, ai_probability, real_probability, verdict, confidence, exif_summary, analysis_text, created_at
            FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
