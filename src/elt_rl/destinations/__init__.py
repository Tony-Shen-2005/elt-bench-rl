from elt_rl.destinations.base import Destination, Namespace


def get_destination(name: str, **kwargs) -> Destination:
    """Registry. To add a warehouse, subclass Destination and add a branch here."""
    if name == "duckdb":
        from elt_rl.destinations.duckdb import DuckDBDestination

        return DuckDBDestination(**kwargs)
    if name == "snowflake":
        from elt_rl.destinations.snowflake import SnowflakeDestination

        return SnowflakeDestination(**kwargs) if kwargs else SnowflakeDestination.from_env()
    raise ValueError(f"Unknown destination {name!r} (available: duckdb, snowflake)")


__all__ = ["Destination", "Namespace", "get_destination"]
