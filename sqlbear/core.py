import re
import sys
import subprocess
import pandas as pd
import importlib.util
from typing import Union
from .bson_connector import ObjectId
from collections.abc import Iterable
from sqlalchemy.engine.url import make_url
from sqlalchemy import create_engine, inspect, text, TEXT, VARCHAR, NVARCHAR
from sqlbear.sql_helpers import check_timezone_support

class SQLBear:
    """A Wrapper to integrate the pandas library with a sqlalchemy connection engine. 
        Provide a connection string to initialize. Custom methods to help interact with 
        the sql server are provided, but the sqlalchemy connection engine is available at SQLBear.prototype.engine"""
    def __init__(self, connection_string: str):
        """Provide a connection string compliant with SQLAlchemy Connection URL standards, https://docs.sqlalchemy.org/en/20/core/engines.html"""
        self.ensure_connector_installed(connection_string)
        self.engine = create_engine(connection_string)
    
    def ensure_connector_installed(self, conn_str: str) -> None:
        """Ensures that the required database driver for the connection string is installed."""
        # Parse connection string to get the driver
        url = make_url(conn_str)
        driver = url.drivername.split("+")[0]  # Extract the DB type (e.g., 'postgresql', 'mysql', 'sqlite')
        # Mapping of database types to common connector packages
        driver_to_package = {
            "postgresql": "psycopg2",
            "postgresql+asyncpg": "asyncpg",
            "mysql": "mysql-connector-python",
            "mysql+pymysql": "pymysql",
            "sqlite": None,  # SQLite is built-in, no need to install
            "mssql": "pyodbc",
        }
        if conn_str[:13] == 'mysql+pymysql':
            package = 'pymysql'
        elif conn_str[:18] == "postgresql+asyncpg":
            package = 'asyncpg'
        else:
            package = driver_to_package.get(driver)
        if package is None:
            return
        # Check if the package is installed
        if importlib.util.find_spec(package) is None:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            return
        
    def infer_sql_text_types(self, df: pd.DataFrame) -> tuple[dict, dict]:
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
        with self.engine.connect() as connection:
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
    
    def check_table_schema(self, table_name: str, required_schema: dict) -> Union[dict, bool]:
        """
        Checks if an existing table meets or exceeds the column type requirements.

        Parameters:
            engine (sqlalchemy.engine.base.Engine): The SQLAlchemy database engine.
            table_name (str): The name of the table to check.
            required_schema (dict): A dictionary like {'column_name': 'VARCHAR(255)', 'description': 'TEXT', 'email': 'NVARCHAR(100)'}

        Returns:
            dict: A dictionary with columns that need to be altered or created.
        """
        inspector = inspect(self.engine)
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

    def delete_from_table(self, table: str, col: Union[str, Iterable], data: pd.DataFrame) -> None:
        """Delete rows from a table based on column values.
        
        Args:
            con: SQLAlchemy engine or connection.
            table (str): Table name.
            col (str or list): Column name(s) to filter by.
            data (list or pd.DataFrame): Values for deletion.

        Raises:
            ValueError: If arguments are missing or invalid.
        """
        if not table or not self.engine or not col or data is None:
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
        with self.engine.connect() as connection:
            with connection.begin() as transaction:
                connection.execute(text(query), params)
    
    def add_indexes(self, table: str, cols: list) -> None:
        """
        Adds indexes to the specified columns in a table if they are not already indexed.
        
        Parameters:
            con (sqlalchemy.engine.base.Engine): The SQLAlchemy connection engine.
            table (str): The name of the table to modify.
            cols (list): A list of column names or iterables of column names for multi-column indexes.
            data (pd.DataFrame): The dataframe containing the table's data to determine string lengths.
        """
        # Query existing indexes on the table
        df = pd.read_sql_query(f"SHOW INDEX FROM {table}", self.engine)
        existing_indexes = set([tuple(group["Column_name"]) for _, group in df.groupby("Key_name")])
        # Convert column definitions to tuples for easy comparison
        cols_to_index = [tuple([col]) if isinstance(col, str) else tuple(col) for col in cols]
        # 
        with self.engine.connect() as conn:
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



    def put_table(self, table: str, col: Union[str, Iterable], data: pd.DataFrame, index_cols: list=[]) -> None:
        # Identify column indices where all values are NA
        na_columns = [idx for idx in range(data.shape[1]) if data.iloc[:, idx].isna().mean() == 1]
        # 
        # Keep only the columns that are *not* in na_columns
        data = data.iloc[:, [idx for idx in range(data.shape[1]) if idx not in na_columns]].copy()
        tz_support = check_timezone_support(self.engine)
        required_types, _ = self.infer_sql_text_types(data)
        columns_to_update = self.check_table_schema(table, required_types)
        # If server is tz unaware this property will be the local timezone of the server and all 
        # timestamps will be converted to that timezone before writing to the database
        if tz_support['server_timezone']: 
            timestamps = []
            for this_col in data.select_dtypes(include=["datetime"]).columns.tolist():
                data[this_col] = pd.to_datetime(data[this_col]).apply(lambda x: x.tz_localize("UTC") if x is not None and x.tzinfo is None else x)
                data[this_col] = data[this_col].dt.tz_convert(tz_support['server_timezone'])
                data[this_col] = data[this_col].dt.tz_localize(None)
        if columns_to_update != False and len(columns_to_update.keys()) == 0:
            self.delete_from_table(table, col, data[col].apply(lambda x: str(x) if isinstance(x, ObjectId) else x))
            data.to_sql(
                name=table,
                index=False,
                if_exists='append',
                con=self.engine
            )
            self.add_indexes(table, [col, *index_cols])
        elif columns_to_update != False and len(columns_to_update.keys()) > 0:
            old_table = pd.read_sql_table(table, self.engine)
            new_table = pd.concat([old_table, data], ignore_index=True).sort_index(axis=1)
            required_types, _ = self.infer_sql_text_types(new_table)
            new_table.to_sql(
                name=table,
                index=False,
                if_exists='replace',
                con=self.engine,
                dtype=required_types
            )
            self.add_indexes(table, [col, *index_cols])
        else:
            data.sort_index(axis=1).to_sql(
                name=table,
                index=False,
                if_exists='fail',
                con=self.engine,
                dtype=required_types
            )
            self.add_indexes(table, [col, *index_cols])
    