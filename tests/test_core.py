import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlbear.core import SQLBear


@pytest.fixture
def sqlbear():
    """Fixture to create an instance of SQLBear with a mocked engine."""
    with patch("sqlbear.core.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        return SQLBear("sqlite:///:memory:")  # ✅ Instance, not class


def test_init(sqlbear):
    """Test that SQLBear initializes correctly."""
    assert sqlbear.engine is not None

@patch("sqlbear.core.importlib.util.find_spec", return_value=True)
@patch("sqlbear.core.subprocess.check_call")
def test_ensure_connector_installed(mock_check_call, mock_find_spec, sqlbear):
    """Test connector installation logic."""
    sqlbear.ensure_connector_installed("postgresql://user:pass@localhost/db")
    mock_check_call.assert_not_called()  # No installation if already found

    mock_find_spec.return_value = None  # Simulate missing package
    sqlbear.ensure_connector_installed("mysql+pymysql://user:pass@localhost/db")
    mock_check_call.assert_called_once()

# @patch("sqlbear.core.SQLBear.engine.connect")
def test_infer_sql_text_types(sqlbear):
    """Test inference of SQL text types."""
    with patch.object(sqlbear.engine, "connect", autospec=True) as mock_connect:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "desc": ["A" * 300, "B" * 500]})
        mock_connect.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = ["utf8mb4"]

        types, charset_info = sqlbear.infer_sql_text_types(df)
        
        assert charset_info["charset"] == "utf8mb4"
        assert str(types["name"]).startswith("VARCHAR")
        assert str(types["desc"]).startswith("TEXT")

# @patch("sqlbear.core.SQLBear.engine.connect")
def test_check_table_schema(mock_connect, sqlbear):
    """Test table schema checking."""
    mock_inspector = MagicMock()
    mock_inspector.get_table_names.return_value = ["test_table"]
    mock_inspector.get_columns.return_value = [{"name": "id", "type": "VARCHAR(50)"}]
    
    with patch("sqlbear.core.inspect", return_value=mock_inspector):
        result = sqlbear.check_table_schema("test_table", {"id": "VARCHAR(100)"})
    
    assert result == {"id": "VARCHAR(100)"}  # Needs an update

# @patch("sqlbear.core.SQLBear.engine.connect")
def test_delete_from_table(sqlbear):
    """Test deletion logic from table."""
    with patch.object(sqlbear.engine, "connect", autospec=True) as mock_connect:
        sqlbear.delete_from_table("users", "id", ["1", "2", "3"])
        mock_connect.assert_called_once()

@patch("sqlbear.core.SQLBear.engine.connect")
def test_add_indexes(mock_connect, sqlbear):
    """Test adding indexes."""
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchall.return_value = []
    
    sqlbear.add_indexes("users", ["email"])
    mock_conn.execute.assert_called()

@patch("sqlbear.core.SQLBear.engine.connect")
def test_put_table(mock_connect, sqlbear):
    """Test `put_table` functionality."""
    df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    mock_connect.return_value.__enter__.return_value.execute.return_value = []
    
    sqlbear.put_table("users", "id", df, index_cols=["name"])
    mock_connect.assert_called()
