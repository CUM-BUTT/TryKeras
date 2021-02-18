import sqlite3

con = sqlite3.connect("base.db")
cur = con.cursor()
with con:
    cur.execute("""
        CREATE TABLE user (
            user_name TEXT NOT NULL PRIMARY KEY,
            password TEXT NOT NULL,
            is_active BOOL DEFAULT TRUE,
        );
    """)

    cur.execute("""
            CREATE TABLE prediction (
                ticker TEXT NOT NULL PRIMARY KEY,
                date DATETIME NOT NULL PRIMARY KEY,
            );
        """)