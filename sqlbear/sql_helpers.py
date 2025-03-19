from datetime import timedelta
from sqlalchemy import text
from dateutil import tz
import pytz
import re


def supports_timezone(engine):
    """Checks if the database supports timezone-aware timestamps."""
    dialect = engine.dialect.name

    if dialect == "postgresql":
        return True  # PostgreSQL fully supports TIMESTAMP WITH TIME ZONE

    if dialect in ["mssql", "mssql+pymssql", "mssql+pyodbc"]:
        return True  # SQL Server supports DATETIMEOFFSET

    if dialect in ["mysql", "mariadb"]:
        return False  # MySQL TIMESTAMP is NOT truly timezone-aware

    if dialect == "sqlite":
        return False  # SQLite doesn't store time zones

    return False  # Default assumption



def normalize_timezone(server_timezone):
    """
    Converts a database-reported timezone into a format usable with pandas.
    
    Parameters:
        server_timezone (str): The timezone string reported by the database.
    
    Returns:
        str: A valid timezone string usable in pandas `tz_convert()`.
    """
    if not server_timezone or server_timezone in ["SYSTEM", "LOCAL"]:
        return "UTC"  # Default to UTC if unknown
    # If it's a named timezone (e.g., "America/New_York"), return as is
    if server_timezone in pytz.all_timezones:
        return server_timezone
    # Handle MySQL/Sysdate/SQL Server offset formats like "+00:00" or "-05:00"
    offset_match = re.match(r"([+-]\d{2}):(\d{2})", server_timezone)
    if offset_match:
        hours, minutes = map(int, offset_match.groups())
        return tz.tzoffset(None, hours * 3600 + minutes * 60)  # Convert to fixed offset
    print(f"Couldn't parse {server_timezone}, normalize_timezone falling back to UTC")
    return "UTC"  # Fallback if unrecognized


def convert_timedelta_to_offset(td):
    """
    Converts a MySQL TIMEDIFF result (which is a timedelta object) into a standard "+HH:MM" or "-HH:MM" format.
    
    Example:
        timedelta(hours=-5) -> "-05:00"
        timedelta(hours=6, minutes=30) -> "+06:30"
    """
    if isinstance(td, timedelta):
        total_seconds = td.total_seconds()
        hours, remainder = divmod(abs(total_seconds), 3600)
        minutes = remainder // 60
        sign = "+" if total_seconds >= 0 else "-"
        return f"{sign}{int(hours):02}:{int(minutes):02}"
    
    print(f"Warning: Unexpected TIMEDIFF result {td}, defaulting to UTC")
    return "UTC"



def get_server_timezone(engine):
    """Detects the database server's timezone setting."""
    dialect = engine.dialect.name
    with engine.connect() as conn:
        if dialect in ["mysql", "mariadb"]:
            result = conn.execute(text("SELECT @@global.time_zone;")).scalar()
            if result in ("SYSTEM", "LOCAL") or result is None:
                # Retrieve offset as a timedelta object
                timedelta_result = conn.execute(text("SELECT TIMEDIFF(NOW(), UTC_TIMESTAMP);")).scalar()
                return convert_timedelta_to_offset(timedelta_result)
            return str(result)
        elif dialect == "postgresql":
            return conn.execute(text("SHOW TIMEZONE;")).scalar()
        elif dialect in ["mssql", "mssql+pymssql", "mssql+pyodbc"]:
            return conn.execute(text("SELECT SYSDATETIMEOFFSET();")).scalar()
        elif dialect == "sqlite":
            return "UTC"
    return "Unknown"


def check_timezone_support(engine):
    """
    Checks if the database supports timezone-aware timestamps.
    If not, detects the server's current timezone.
    
    Returns:
        dict: {'supports_timezone': bool, 'server_timezone': str}
    """
    supports_tz = supports_timezone(engine)
    server_tz = None if supports_tz else get_server_timezone(engine)

    return {
        "supports_timezone": supports_tz,
        "server_timezone": normalize_timezone(server_tz)
    }



