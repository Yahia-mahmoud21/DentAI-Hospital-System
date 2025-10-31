import db, os, sqlite3
print('DB_PATH:', db.DB_PATH)
print('exists:', os.path.exists(db.DB_PATH))
conn = sqlite3.connect(db.DB_PATH)
print('tables:', conn.execute("select name from sqlite_master where type='table';").fetchall())
conn.close()
