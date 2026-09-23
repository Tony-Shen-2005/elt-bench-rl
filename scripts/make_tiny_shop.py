"""Generate the credential-free ``tiny_shop`` task used by the local integration test.

Mirrors an ELT-Bench task at small scale: three source kinds (a CSV flat
file, a JSONL object-store export, a SQLite database standing in for
Postgres), two data models, and edge cases the grader must handle (a
customer with no orders, NULL prices, ties broken by a stated rule).

    uv run python scripts/make_tiny_shop.py
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sqlite3
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny_shop"

REFERENCE_SQL = {
    "customer_ltv": """
        SELECT c.customer_id,
               c.first_name || ' ' || c.last_name AS customer_name,
               COUNT(o.order_id) AS n_orders,
               COALESCE(SUM(o.quantity * p.unit_price), 0) AS total_spent,
               CAST(MIN(o.order_date) AS DATE) AS first_order_date
        FROM tiny_shop.customers c
        LEFT JOIN tiny_shop.orders o ON o.customer_id = c.customer_id
        LEFT JOIN tiny_shop.products p ON p.product_id = o.product_id
        GROUP BY ALL
    """,
    "product_sales": """
        WITH s AS (
            SELECT p.product_id, p.product_name,
                   COALESCE(SUM(o.quantity), 0) AS units_sold
            FROM tiny_shop.products p
            LEFT JOIN tiny_shop.orders o ON o.product_id = p.product_id
            GROUP BY ALL
        )
        SELECT *, ROW_NUMBER() OVER (ORDER BY units_sold DESC, product_id ASC) AS sales_rank
        FROM s
    """,
}

DATA_MODEL = """models:
  - name: customer_ltv
    description: One row per customer (including customers without orders) with lifetime order statistics.
    columns:
      - name: customer_id
        description: Customer identifier.
      - name: customer_name
        description: First name and last name separated by a single space.
      - name: n_orders
        description: Number of orders placed by the customer; 0 if none.
      - name: total_spent
        description: Sum over the customer's orders of quantity times the product's unit price. Orders of products with unknown price contribute nothing. 0 if no orders.
      - name: first_order_date
        description: Date (YYYY-MM-DD) of the customer's earliest order; NULL if no orders.
  - name: product_sales
    description: One row per product with units sold and a sales rank.
    columns:
      - name: product_id
        description: Product identifier.
      - name: product_name
        description: Product name.
      - name: units_sold
        description: Total quantity sold across all orders; 0 if never ordered.
      - name: sales_rank
        description: Rank by units_sold descending, ties broken by product_id ascending; starts at 1.
"""

CONFIG = """task: tiny_shop
sources:
  flat_files:
    - {format: csv, path: sources/customers.csv, table: customers}
  object_store:
    - {format: jsonl, path: sources/orders.jsonl, table: orders}
  sqlite:
    - {path: sources/shop.sqlite, tables: [products]}
"""


def main() -> None:
    rng = random.Random(0)
    shutil.rmtree(ROOT, ignore_errors=True)
    (ROOT / "sources").mkdir(parents=True)
    (ROOT / "grader" / "sql").mkdir(parents=True)
    (ROOT / "grader" / "gt").mkdir(parents=True)

    first = ["Ada", "Alan", "Grace", "Edsger", "Barbara", "Donald", "Frances", "John"]
    last = ["Lovelace", "Turing", "Hopper", "Dijkstra", "Liskov", "Knuth", "Allen", "Backus"]
    customers = [(i, first[i - 1], last[i - 1]) for i in range(1, 9)]
    with open(ROOT / "sources" / "customers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "first_name", "last_name"])
        w.writerows(customers)

    products = [(1, "widget", 2.50), (2, "gadget", 10.00), (3, "gizmo", None),
                (4, "doohickey", 7.25), (5, "sprocket", 1.10)]
    con = sqlite3.connect(ROOT / "sources" / "shop.sqlite")
    con.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT, unit_price REAL)")
    con.executemany("INSERT INTO products VALUES (?, ?, ?)", products)
    con.commit()
    con.close()

    orders = []
    oid = 100
    for cid in range(1, 8):  # customer 8 never orders
        for _ in range(rng.randint(1, 4)):
            oid += 1
            pid = rng.choice([1, 2, 3, 4])  # product 5 never sold
            orders.append({"order_id": oid, "customer_id": cid, "product_id": pid,
                           "quantity": rng.randint(1, 5),
                           "order_date": f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T12:00:00"})
    with open(ROOT / "sources" / "orders.jsonl", "w") as f:
        for o in orders:
            f.write(json.dumps(o) + "\n")

    (ROOT / "config.yaml").write_text(CONFIG)
    (ROOT / "data_model.yaml").write_text(DATA_MODEL)
    expected = {"customers": len(customers), "orders": len(orders), "products": len(products)}
    (ROOT / "grader" / "expected_rows.json").write_text(json.dumps(expected, indent=2))
    (ROOT / "grader" / "sort_key.json").write_text(
        json.dumps({"customer_ltv": ["customer_id"], "product_sales": ["product_id"]}, indent=2))

    # Ground truth: load sources with the reference loader, run reference SQL.
    db = duckdb.connect()
    db.execute("CREATE SCHEMA tiny_shop")
    db.execute(f"CREATE TABLE tiny_shop.customers AS SELECT * FROM read_csv_auto('{ROOT}/sources/customers.csv')")
    db.execute(f"CREATE TABLE tiny_shop.orders AS SELECT * FROM read_json_auto('{ROOT}/sources/orders.jsonl')")
    db.execute("CREATE TABLE tiny_shop.products (product_id INTEGER, product_name VARCHAR, unit_price DOUBLE)")
    db.executemany("INSERT INTO tiny_shop.products VALUES (?, ?, ?)", products)
    for name, sql in REFERENCE_SQL.items():
        (ROOT / "grader" / "sql" / f"{name}.sql").write_text(f"select * from tiny_shop.{name}\n")
        db.execute(f"CREATE TABLE tiny_shop.{name} AS {sql}")
        db.execute(f"SELECT * FROM tiny_shop.{name}").df().to_csv(ROOT / "grader" / "gt" / f"{name}.csv", index=False)
    print(f"wrote {ROOT}: {expected}")


if __name__ == "__main__":
    main()
