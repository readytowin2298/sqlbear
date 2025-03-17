from sqlalchemy import text, Text, TEXT, VARCHAR, NVARCHAR, inspect
from sqlalchemy.exc import ProgrammingError
from collections.abc import Iterable
from dateutil import tz
import pandas as pd
from .bson_connector import ObjectId
import pytz
import re


def infer_sql_text_types(df, con):
    """
        Infers the best SQL text field types for a Pandas DataFrame and scans for illegal characters.
        Args:
            df (pd.DataFrame): The input DataFrame.
            connection (sqlalchemy.engine.base.Connection): An active SQLAlchemy connection.
        Returns:
            tuple: (dict of suggested SQL column types, dict of database charset info)
        Raises:
            ValueError: If illegal characters are found in any column.
    """
    # Query database for charset and collation
    with con.connect() as connection:
        dialect = connection.engine.dialect.name.lower()
        illegal_chars = None
        charset_info = {}
        if dialect == "mysql":
            query = text("SELECT @@character_set_database AS charset, @@collation_database AS collation")
            result = connection.execute(query).fetchone()
            charset_info = {"charset": result.charset, "collation": result.collation}
            # Known illegal characters for MySQL charsets
            if result.charset.lower() == "utf8mb4":
                illegal_chars = set()  # No known illegal characters
            elif result.charset.lower() == "latin1":
                illegal_chars = set("\x00")  # NULL character is restricted
        elif dialect == "postgresql":
            query = text("SHOW SERVER_ENCODING")
            result = connection.execute(query).fetchone()
            charset_info = {"charset": result[0], "collation": "C.UTF-8"}
            # No strict illegal character rules for UTF-8 in PostgreSQL
            illegal_chars = set()
        elif dialect == "mssql":
            query = text("SELECT SERVERPROPERTY('Collation') AS collation")
            result = connection.execute(query).fetchone()
            charset_info = {"charset": "UTF-16LE", "collation": result.collation}
            # SQL Server disallows some control characters
            illegal_chars = set("\x00\x1A")  # NULL and SUBSTITUTE characters
        else:
            raise ValueError(f"Unsupported database dialect: {dialect}")
        # Infer text column types
        suggested_types = {}
        error_list = []
        for col in df.select_dtypes(include=[object, "string"]):
            max_length = df[col].dropna().astype(str).str.len().max()
            # Suggest field type based on max string length
            if max_length is None or max_length == 0:
                suggested_types[col] = TEXT()  # Default to TEXT if unknown
            elif max_length <= 255:
                suggested_types[col] = VARCHAR(max_length)
            elif max_length <= 4000 and dialect == "mssql":
                suggested_types[col] = NVARCHAR(max_length)
            else:
                suggested_types[col] = TEXT(max_length)
            # Scan for illegal characters if any are defined
            if illegal_chars:
                offending_rows = df[col].dropna().apply(lambda x: any(c in illegal_chars for c in str(x)))
                if offending_rows.any():
                    indexes = df.index[offending_rows].tolist()
                    error_list.append((col, indexes, "\n", df.loc[indexes, col], "\n"))
        # Raise error if illegal characters are found
        if error_list:
            raise ValueError(f"Illegal characters found in columns: {error_list}")
        return suggested_types, charset_info




def check_table_schema(engine, table_name, required_schema):
    """
    Checks if an existing table meets or exceeds the column type requirements.

    Parameters:
        engine (sqlalchemy.engine.base.Engine): The SQLAlchemy database engine.
        table_name (str): The name of the table to check.
        required_schema (dict): A dictionary like {'column_name': 'VARCHAR(255)', 'description': 'TEXT', 'email': 'NVARCHAR(100)'}

    Returns:
        dict: A dictionary with columns that need to be altered or created.
    """
    inspector = inspect(engine)
    existing_columns = {}
    # Check if the table exists
    if table_name not in inspector.get_table_names():
        return False
    # Fetch existing column info
    columns = inspector.get_columns(table_name)
    for col in columns:
        existing_columns[col["name"]] = str(col["type"]).upper()  # Normalize type for comparison
    # Analyze column types
    columns_to_update = {}
    for col, required_type in required_schema.items():
        required_type = str(required_type).upper()  # Normalize input
        if col not in existing_columns:
            columns_to_update[col] = required_type  # Column is missing, needs creation
        else:
            existing_type = existing_columns[col]
            # Handle VARCHAR(N) and NVARCHAR(N) types
            required_match = re.match(r"(NVARCHAR|VARCHAR)\((\d+)\)", required_type)
            existing_match = re.match(r"(NVARCHAR|VARCHAR)\((\d+)\)", existing_type)
            if required_match and existing_match:
                required_length = int(required_match.group(2))
                existing_length = int(existing_match.group(2))
                # If existing column is too small, suggest an update
                if existing_length < required_length:
                    columns_to_update[col] = required_type
            # Handle TEXT and NVARCHAR(MAX) types
            elif "TEXT" in required_type or "NVARCHAR(MAX)" in required_type:
                if not (existing_type.startswith("TEXT") or existing_type.startswith("NVARCHAR(MAX)")):
                    columns_to_update[col] = required_type  # Needs upgrade to TEXT/NVARCHAR(MAX)
    return columns_to_update



def delete_from_table(con, table, col, data):
    """Delete rows from a table based on column values.
    
    Args:
        con: SQLAlchemy engine or connection.
        table (str): Table name.
        col (str or list): Column name(s) to filter by.
        data (list or pd.DataFrame): Values for deletion.

    Raises:
        ValueError: If arguments are missing or invalid.
    """
    if not table or not con or not col or data is None:
        raise ValueError("Must include all arguments!")
        # 
    if isinstance(col, str):  # Single-column case
        if isinstance(data, str) or isinstance(data, pd.DataFrame) or not isinstance(data, Iterable):
            print(f"String: {isinstance(data, str)}",)
            print(f"DataFrame: {isinstance(data, pd.DataFrame)}")
            print(f"Iterable: {isinstance(data, Iterable)}")
            print(data)
            raise ValueError("For a single column, data must be a list-like iterable.")
            # 
        query = f"DELETE FROM {table} WHERE {col} IN ({', '.join([':param'+str(i) for i in range(len(data))])})"
        params = {f'param{i}': value for i, value in enumerate(data)}
        # 
    elif isinstance(col, Iterable) and all(isinstance(c, str) for c in col):  # Multi-column case
        if not isinstance(data, pd.DataFrame):
            raise ValueError("For multiple columns, data must be a DataFrame.")
            # 
        col_list = ', '.join(col)
        placeholders = ', '.join([f"({', '.join([':param'+str(i)+'_'+str(j) for j in range(len(col))])})" for i in range(len(data))])
        query = f"DELETE FROM {table} WHERE ({col_list}) IN ({placeholders})"
        # 
        params = {f'param{i}_{j}': data.iloc[i, j] for i in range(len(data)) for j in range(len(col))}
        # 
    else:
        raise ValueError("Invalid column format. Must be a string or list of strings.")
        # 
    # Execute the query with a transaction
    with con.connect() as connection:
        with connection.begin() as transaction:
            connection.execute(text(query), params)



def add_indexes(con, table, cols, data):
    """
    Adds indexes to the specified columns in a table if they are not already indexed.
    
    Parameters:
        con (sqlalchemy.engine.base.Engine): The SQLAlchemy connection engine.
        table (str): The name of the table to modify.
        cols (list): A list of column names or iterables of column names for multi-column indexes.
        data (pd.DataFrame): The dataframe containing the table's data to determine string lengths.
    """
    with con.connect() as conn:
        # Query existing indexes on the table
        df = pd.read_sql_query(f"SHOW INDEX FROM {table}", con)
        existing_indexes = set([tuple(group["Column_name"]) for _, group in df.groupby("Key_name")])
        # Convert column definitions to tuples for easy comparison
        cols_to_index = [tuple([col]) if isinstance(col, str) else tuple(col) for col in cols]
        # 
        for col_tuple in cols_to_index:
            if col_tuple in existing_indexes:
                continue  # Skip if index already exists
                # 
            # Build index SQL query
            index_name = "idx_" + "_".join(col_tuple)
            column_defs = [f"`{col}`" for col in col_tuple]
            #  
            index_sql = f"CREATE INDEX `{index_name}` ON `{table}` ({', '.join(column_defs)})"
            conn.execute(text(index_sql))
            print(f"Created index: {index_name}")



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
    return "UTC"  # Fallback if unrecognized


def get_server_timezone(engine):
    """Detects the database server's timezone setting."""
    dialect = engine.dialect.name
    with engine.connect() as conn:
        if dialect in ["mysql", "mariadb"]:
            result = conn.execute(text("SELECT @@global.time_zone;")).scalar()
            if result in ("SYSTEM", "LOCAL"):
                result = conn.execute(text("SELECT TIMEDIFF(NOW(), UTC_TIMESTAMP);")).scalar()
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





def put_table(con, table, col, data, index_cols=[]):
    # Identify column indices where all values are NA
    na_columns = [idx for idx in range(data.shape[1]) if data.iloc[:, idx].isna().mean() == 1]
    # 
    # Keep only the columns that are *not* in na_columns
    data = data.iloc[:, [idx for idx in range(data.shape[1]) if idx not in na_columns]].copy()
    tz_support = check_timezone_support(con)
    required_types, _ = infer_sql_text_types(data, con)
    columns_to_update = check_table_schema(con, table, required_types)
    if tz_support['server_timezone']: 
        timestamps = []
        for this_col in data.select_dtypes(include=["datetime"]).columns.tolist():
            data[this_col] = pd.to_datetime(data[this_col]).apply(lambda x: x.tz_localize("UTC") if x is not None and x.tzinfo is None else x)
            data[this_col] = data[this_col].dt.tz_convert(tz_support['server_timezone'])
            data[this_col] = data[this_col].dt.tz_localize(None)
    if columns_to_update != False and len(columns_to_update.keys()) == 0:
        delete_from_table(con, table, col, data[col].apply(lambda x: str(x) if isinstance(x, ObjectId) else x))
        data.to_sql(
            name=table,
            index=False,
            if_exists='append',
            con=con
        )
        add_indexes(con, table, [col, *index_cols], data)
    elif columns_to_update != False and len(columns_to_update.keys()) > 0:
        old_table = pd.read_sql_table(table, con)
        new_table = pd.concat([old_table, data], ignore_index=True).sort_index(axis=1)
        required_types, _ = infer_sql_text_types(new_table, con)
        new_table.to_sql(
            name=table,
            index=False,
            if_exists='replace',
            con=con,
            dtype=required_types
        )
        add_indexes(con, table, [col, *index_cols], data)
    else:
        data.sort_index(axis=1).to_sql(
            name=table,
            index=False,
            if_exists='fail',
            con=con,
            dtype=required_types
        )
        add_indexes(con, table, [col, *index_cols], data)
