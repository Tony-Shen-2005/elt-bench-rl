"""Reference transformations for tiny_shop."""
import duckdb
import yaml

MODELS = {
    "customer_ltv": """
        SELECT c.customer_id, c.first_name || ' ' || c.last_name AS customer_name,
               COUNT(o.order_id) AS n_orders,
               COALESCE(SUM(o.quantity * p.unit_price), 0) AS total_spent,
               CAST(MIN(o.order_date) AS DATE) AS first_order_date
        FROM customers c
        LEFT JOIN orders o ON o.customer_id = c.customer_id
        LEFT JOIN products p ON p.product_id = o.product_id
        GROUP BY ALL""",
    "product_sales": """
        SELECT product_id, product_name, units_sold,
               ROW_NUMBER() OVER (ORDER BY units_sold DESC, product_id) AS sales_rank
        FROM (SELECT p.product_id, p.product_name, COALESCE(SUM(o.quantity), 0) AS units_sold
              FROM products p LEFT JOIN orders o ON o.product_id = p.product_id GROUP BY ALL)""",
}
cfg = yaml.safe_load(open("config.yaml"))["duckdb"]["config"]
con = duckdb.connect(cfg["path"])
con.execute(f'SET schema = \'{cfg["schema"]}\'')
for name, sql in MODELS.items():
    con.execute(f"CREATE OR REPLACE TABLE {name} AS {sql}")
con.close()
print("transformed")
