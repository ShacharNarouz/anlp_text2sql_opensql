PREDICT_TABLES_PROMPT_DESCRIPTION = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: DESCRIPTION
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables , with their properties:
{table_info}
### Question: {question}
### Note that: {note}

Find the required tables based on the QUESTION.
Tables:
"""


PREDICT_COLUMNS_PROMPT_N = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}

Given the tables:
{used_tables} .
From the given tables, find the required columns based on the QUESTION.
Columns:
"""

PREDICT_COLUMNS_PROMPT_VDT = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...)
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}

Given the tables:
{used_tables} .
From the given tables, find the required columns based on the QUESTION.
Columns:
"""

PREDICT_COLUMNS_PROMPT_ALL = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...), PRIMARY_KEY
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...)
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}

Given the tables:
{used_tables} .
From the given tables, find the required columns based on the QUESTION.
Columns:
"""


PREDICT_SKELETON_PROMPT_ALL = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...)
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}

Given the tables and columns used in the SQL query:
{used_tables_and_columns} .

Based on the given the tables and columns, write the skeleton of the SQL query corresponding to the question.
Skeleton:
"""


PREDICT_SQL_PROMPT_N = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}
Please generate the SQL script STEP BY STEP.
Given the tables and columns used in the SQL query:
{used_tables_and_columns} .
### Complete sqlite SQL query based on the given tables and columns.
Reply only with the SQL query.

SELECT
"""

PREDICT_SQL_PROMPT_VDT = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...)
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}
Please generate the SQL script STEP BY STEP.
Given the tables and columns used in the SQL query:
{used_tables_and_columns} .
### Complete sqlite SQL query based on the given tables and columns.
Reply only with the SQL query.

SELECT
"""

PREDICT_SQL_PROMPT_ALL = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...), PRIMARY_KEY
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...)
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}
Please generate the SQL script STEP BY STEP.
Given the tables and columns used in the SQL query:
{used_tables_and_columns} .
### Complete sqlite SQL query based on the given tables and columns.
Reply only with the SQL query.

SELECT
"""


PREDICT_SQL_NO_COT_PROMPT_ALL = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...), PRIMARY_KEY
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...)
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}
Please generate the SQL script STEP BY STEP.

### Complete sqlite SQL query based on the given tables and columns.
Reply only with the SQL query.

SELECT
"""


PREDICT_SQL_NO_COT_PROMPT_VDT = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME: TYPE, (DESCRIPTION), (VALUE1, VALUE2, ...)
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}
Please generate the SQL script STEP BY STEP.

### Complete sqlite SQL query based on the given tables and columns.
Reply only with the SQL query.

SELECT
"""


PREDICT_SQL_NO_COT_PROMPT_N = """
### SQLite SQL tables are requested to be represented in the following format.
TABLE_NAME (
COLUMN_NAME
)
FOREIGN KEYS:
TABLE_NAME1.COLUMN_NAME1=TABLE_NAME2.COLUMN_NAME2
### Here are SQLite SQL tables that will be used, with their properties:
{table_info}
### Question: {question}
### Note that: {note}
Please generate the SQL script STEP BY STEP.

### Complete sqlite SQL query based on the given tables and columns.
Reply only with the SQL query.

SELECT
"""


PREDICT_SQLCODER_PROMPT = """
### Task
Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{table_metadata_string_DDL_statements}

### External Knowledge
Here is some external knowledge that might be relevant to the question and the query:
{external_knowledge}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{user_question}[/QUESTION]
[SQL]
"""

