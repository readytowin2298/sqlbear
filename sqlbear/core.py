import sys
import subprocess
import importlib.util
from sqlalchemy.engine.url import make_url
from sqlalchemy import create_engine
from sqlbear.sql_helpers import put_table, check_timezone_support, check_table_schema, get_server_timezone, \
    normalize_timezone, supports_timezone, add_indexes, delete_from_table, infer_sql_text_types


class SQLBear:
    """A Wrapper to integrate the pandas library with a sqlalchemy connection engine. 
        Provide a connection string to initialize. Custom methods to help interact with 
        the sql server are provided, but the sqlalchemy connection engine is available at SQLBear.prototype.engine"""
    def __init__(self, connection_string: str):
        """Provide a connection string compliant with SQLAlchemy Connection URL standards, https://docs.sqlalchemy.org/en/20/core/engines.html"""
        self.ensure_connector_installed(connection_string)
        self.engine = create_engine(connection_string)
    
    def ensure_connector_installed(self, conn_str: str):
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
        else:
            package = driver_to_package.get(driver)
        if package is None:
            # print(f"No external package needed for {driver}.")
            return
        # Check if the package is installed
        if importlib.util.find_spec(package) is None:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            # print(f"Package '{package}' is already installed.")
            return
    
    def put_table(self, table, col, data, index_cols=[]):
        put_table(self.engine, table, col, data, index_cols)
    