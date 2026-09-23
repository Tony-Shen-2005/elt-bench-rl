"""Reference EL for tiny_shop: load every source table listed in config.yaml."""
import sqlite3

import duckdb
import yaml

cfg = yaml.safe_load(open("config.yaml"))
dest = cfg["duckdb"]["config"]
schema = dest["schema"]
con = duckdb.connect(dest["path"])
con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
src = cfg["sources"]
for f in src["flat_files"]:
    con.execute(f'CREATE OR REPLACE TABLE "{schema}"."{f["table"]}" AS SELECT * FROM read_csv_auto(\'{f["path"]}\')')
for f in src["object_store"]:
    con.execute(f'CREATE OR REPLACE TABLE "{schema}"."{f["table"]}" AS SELECT * FROM read_json_auto(\'{f["path"]}\')')
for db in src["sqlite"]:
    lite = sqlite3.connect(db["path"])
    for t in db["tables"]:
        cur = lite.execute(f"SELECT * FROM {t}")
        df = __import__("pandas").DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        con.register("df", df)
        con.execute(f'CREATE OR REPLACE TABLE "{schema}"."{t}" AS SELECT * FROM df')
        con.unregister("df")
con.close()
print("loaded")
