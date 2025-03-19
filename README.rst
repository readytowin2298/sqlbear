SQLBear
========

SQLBear is a lightweight wrapper around SQLAlchemy and Pandas to simplify SQL interactions.

Installation
------------
.. code-block:: bash

    pip install sqlbear

Usage
-----
.. code-block:: python

    from sqlbear import SQLBear
    from datetime import datetime
    # Connects to sqlite database
    connection_str = "sqlite:///example.db"
    sqlbear = SQLBear(connection_str)
    # Query account and user tables
    df = sqlbear.read_sql_query("SELECT * FROM users JOIN account ON users.account = account.id WHERE lastUpdated > DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)")
    # Perform some transformation
    df['added_to_report'] = datetime.now()
    # Write to new or existing sql table, handling new columns, mismatched sql types, timezone issues if present, and duplicates across the user_id column
    nob_bear.put_table(
        'account_users',
        'user_id',
        df,
        ['id','user_id']
    )

Features
--------
- Efficient SQL querying with progress bars
- Automatic handling of time zones and schema updates
- Simple table insertion and schema validation
