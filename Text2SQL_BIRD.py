
# ---------- imports --------------------------------------------------------------------------------------------------

import json
import pickle
import sqlite3
import Prompts
import anthropic
import os
# import dotenv

from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline


# ---------- paths --------------------------------------------------------------------------------------------------

path_scheme_A ="./data/schema_A.json"
path_scheme_D ="./data/schema_D.json"
path_table ="./data/table_info.json"
path_table_dev ="./data/dev_tables.json"
path_dev_json ="dev.json"

databases_to_check = ['debit_card_specializing', 'financial', 'formula_1', 'student_club', 'superhero', 'thrombosis_prediction', 'toxicology']
sqlcoder_pipe = pipeline("text-generation", model="defog/sqlcoder-7b-2", device=0)


# ---------- Models --------------------------------------------------------------------------------------------------


def call_claude(prompt: str) -> str:
    # Initialize the client
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="API KEY",
    )

    # Call the API
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
            }
        ]
    )

    print(message.content)
    return message.completion

def call_a21(prompt: str) -> str:
    client = AI21Client(api_key="API KEY")
    messages = [
        ChatMessage(
            role="user",
            content=prompt
        )
    ]

    response = client.chat.completions.create(
        model="jamba-instruct-preview",
        messages=messages,
        top_p=0.1 # Setting to 1 encourages different responses each call.
    )

    return response.choices[0].message.content, response.usage

def call_sqlcoder(prompt: str) -> str:
    # Use a pipeline as a high-level helper
    sqlcoder_pipe.model.to("cuda:0")

    messages = [
        {"role": "user", "content": prompt},
    ]

    response = sqlcoder_pipe(messages, max_new_tokens=200, num_beams=4, do_sample=False)
    return response[0]['generated_text'][1]['content'], None


# ---------- DB --------------------------------------------------------------------------------------------------


def table_to_columns(table_name: str, tables_info: dict) -> List[str]:
    curent_table_info = tables_info[table_name]
    columns = []
    for key in curent_table_info.keys():
        if key == 'tables':
            continue
        columns.extend(curent_table_info[key])

    raise columns

def response_to_tables(dataset_name:str, response: str, tables_info: dict) -> List[str]:
    dataset_tables = tables_info[dataset_name]['tables']
    extracted_tables = []
    for table_name in dataset_tables:
        if table_name in response:
            extracted_tables.append(table_name)
    return extracted_tables

def response_to_columns(dataset_name:str, response: str, tables_info: dict, relevant_tables: list[str]) -> List[str]:
    dataset_info = tables_info[dataset_name]
    extracted_columns = []
    for table_name in relevant_tables:
        table_columns = dataset_info[table_name]
        for column_name in table_columns:
            if column_name in response:
                extracted_columns.append(f"{table_name}.{column_name}")
    return extracted_columns


# ---------- Question Schemes --------------------------------------------------------------------------------------------------


class BirdQuestion(BaseModel):
    question_id: int
    db_id: str
    question: str
    evidence: str
    SQL: str
    difficulty: str


def schema_D(question: BirdQuestion) -> str:
    schema_info = json.load(open(schema_D_info_file_path, 'r'))

    schema_str = ""
    schema_foriegn_keys = schema_info[question.db_id]['foreign_keys']
    for table_name, columns in schema_info[question.db_id].items():
        if table_name == 'foreign_keys':
            continue
        schema_str += table_name + " (\n"
        for column_name, column_info in columns.items():
            schema_str += column_name + ": " + column_info + "\n"

        schema_str.rstrip("\n")
        schema_str += ")\n"

    schema_str += "FOREIGN KEYS:\n"
    for foreign_key in schema_foriegn_keys:
        schema_str += foreign_key + "\n"

    schema_str.rstrip("\n")

    return schema_str

def schema_A(question: BirdQuestion, predicted_tables: List[str]) -> str:
    schema_info = json.load(open(schema_A_info_file_path, 'r'))

    schema_str = ""
    schema_foriegn_keys = schema_info[question.db_id]['foreign_keys']
    for table_name, columns in schema_info[question.db_id].items():
        if table_name == 'foreign_keys' or table_name not in predicted_tables:
            continue
        schema_str += table_name + " (\n"
        for column_name, column_info in columns.items():
            schema_str += column_name + ": " + column_info + "\n"

        schema_str.rstrip("\n")
        schema_str += ")\n"

    schema_str += "FOREIGN KEYS:\n"
    for foreign_key in schema_foriegn_keys:
        key1, key2 = foreign_key.split('=')
        table_of_key1 = key1.split('.')[0]
        table_of_key2 = key2.split('.')[0]
        if table_of_key1 in predicted_tables and table_of_key2 in predicted_tables:
            schema_str += foreign_key + "\n"

    schema_str.rstrip("\n")

    return schema_str

def schema_N(question: BirdQuestion, predicted_tables: List[str]) -> str:
    schema_info = json.load(open(schema_D_info_file_path, 'r'))

    schema_str = ""
    schema_foriegn_keys = schema_info[question.db_id]['foreign_keys']
    for table_name, columns in schema_info[question.db_id].items():
        if table_name == 'foreign_keys' or table_name not in predicted_tables:
            continue
        schema_str += table_name + " (\n"
        for column_name, column_info in columns.items():
            schema_str += column_name + "\n"

        schema_str.rstrip("\n")
        schema_str += ")\n"

    schema_str += "FOREIGN KEYS:\n"
    for foreign_key in schema_foriegn_keys:
        key1, key2 = foreign_key.split('=')
        table_of_key1 = key1.split('.')[0]
        table_of_key2 = key2.split('.')[0]
        if table_of_key1 in predicted_tables and table_of_key2 in predicted_tables:
            schema_str += foreign_key + "\n"

    schema_str.rstrip("\n")

    return schema_str

def schema_VDT(question: BirdQuestion, predicted_tables: List[str]) -> str:
    schema_info = json.load(open(schema_A_info_file_path, 'r'))

    schema_str = ""
    schema_foriegn_keys = schema_info[question.db_id]['foreign_keys']
    for table_name, columns in schema_info[question.db_id].items():
        if table_name == 'foreign_keys' or table_name not in predicted_tables:
            continue
        schema_str += table_name + " (\n"
        for column_name, column_info in columns.items():
            schema_str += column_name + ": " + column_info.replace(', PRIMARY KEY', '') + "\n"

        schema_str.rstrip("\n")
        schema_str += ")\n"

    schema_str += "FOREIGN KEYS:\n"
    for foreign_key in schema_foriegn_keys:
        key1, key2 = foreign_key.split('=')
        table_of_key1 = key1.split('.')[0]
        table_of_key2 = key2.split('.')[0]
        if table_of_key1 in predicted_tables and table_of_key2 in predicted_tables:
            schema_str += foreign_key + "\n"

    schema_str.rstrip("\n")

    return schema_str


def schema_A_all(question: BirdQuestion) -> str:
    schema_info = json.load(open(schema_A_info_file_path, 'r'))

    schema_str = ""
    schema_foriegn_keys = schema_info[question.db_id]['foreign_keys']
    for table_name, columns in schema_info[question.db_id].items():
        if table_name == 'foreign_keys':
            continue
        schema_str += table_name + " (\n"
        for column_name, column_info in columns.items():
            schema_str += column_name + ": " + column_info + "\n"

        schema_str.rstrip("\n")
        schema_str += ")\n"

    schema_str += "FOREIGN KEYS:\n"
    for foreign_key in schema_foriegn_keys:
        schema_str += foreign_key + "\n"

    schema_str.rstrip("\n")

    return schema_str

def schema_N_all(question: BirdQuestion) -> str:
    schema_info = json.load(open(schema_D_info_file_path, 'r'))

    schema_str = ""
    schema_foriegn_keys = schema_info[question.db_id]['foreign_keys']
    for table_name, columns in schema_info[question.db_id].items():
        if table_name == 'foreign_keys':
            continue
        schema_str += table_name + " (\n"
        for column_name, column_info in columns.items():
            schema_str += column_name + "\n"

        schema_str.rstrip("\n")
        schema_str += ")\n"

    schema_str += "FOREIGN KEYS:\n"
    for foreign_key in schema_foriegn_keys:
        schema_str += foreign_key + "\n"

    schema_str.rstrip("\n")

    return schema_str

def schema_VDT_all(question: BirdQuestion) -> str:
    schema_info = json.load(open(schema_A_info_file_path, 'r'))

    schema_str = ""
    schema_foriegn_keys = schema_info[question.db_id]['foreign_keys']
    for table_name, columns in schema_info[question.db_id].items():
        if table_name == 'foreign_keys':
            continue
        schema_str += table_name + " (\n"
        for column_name, column_info in columns.items():
            schema_str += column_name + ": " + column_info.replace(', PRIMARY KEY', '') + "\n"
            
        schema_str.rstrip("\n")
        schema_str += ")\n"

    schema_str += "FOREIGN KEYS:\n"
    for foreign_key in schema_foriegn_keys:
        schema_str += foreign_key + "\n"

    schema_str.rstrip("\n")

    return schema_str



# ---------- Predictions --------------------------------------------------------------------------------------------------


def predicted_query(question: BirdQuestion, method: str = "ALL", model: str = "A21") -> str:
    schema_d_info = schema_D(question)
    tables_info = json.load(open(table_info_file_path, 'r'))

    # print("======================= STEP 1 =======================")
    # Step 1 - Extract tables
    predict_tables_prompt = Prompts.PREDICT_TABLES_PROMPT_DESCRIPTION.format(table_info=schema_d_info,
                                                                     question=question.question,
                                                                     note=question.evidence)
    
    if model == "A21":
        response, usage = call_a21(predict_tables_prompt)
    elif model == "SQLCODER":
        response, usage = call_sqlcoder(predict_tables_prompt)
    else:
        raise ValueError("Invalid model")
	
    extracted_tables = response_to_tables(question.db_id, response, tables_info)

    schema_a_info = schema_A(question, extracted_tables)
    # print("======================= STEP 2 =======================")
    # Step 2 - Extract columns
    if method == "ALL":
        predict_columns_prompt = Prompts.PREDICT_COLUMNS_PROMPT_ALL.format(table_info=schema_A(question, extracted_tables),
                                                                question=question.question,
                                                                note=question.evidence,
                                                                used_tables= ', '.join(extracted_tables))
    elif method == "VDT":
        predict_columns_prompt = Prompts.PREDICT_COLUMNS_PROMPT_VDT.format(table_info=schema_VDT(question, extracted_tables),
                                                                question=question.question,
                                                                note=question.evidence,
                                                                used_tables= ', '.join(extracted_tables))
    elif method == "N":
        predict_columns_prompt = Prompts.PREDICT_COLUMNS_PROMPT_N.format(table_info=schema_N(question, extracted_tables),
                                                                question=question.question,
                                                                note=question.evidence,
                                                                used_tables= ', '.join(extracted_tables))
    else:
        raise ValueError("Invalid method")


    if model == "A21":
        response, usage = call_a21(predict_columns_prompt)
    elif model == "SQLCODER":
        response, usage = call_sqlcoder(predict_columns_prompt)
    else:
        raise ValueError("Invalid model")
        
    extracted_columns = response_to_columns(question.db_id, response, tables_info, extracted_tables)

    # print("======================= STEP 3 =======================")
    # Step 3 - Generate SQL query
    if method == "ALL":
        predict_sql_query_prompt = Prompts.PREDICT_SQL_PROMPT_ALL.format(table_info=schema_A(question, extracted_tables),
                                                            question=question.question,
                                                            note=question.evidence,
                                                            used_tables_and_columns= ', '.join(extracted_columns))
    elif method == "VDT":
        predict_sql_query_prompt = Prompts.PREDICT_SQL_PROMPT_VDT.format(table_info=schema_VDT(question, extracted_tables),
                                                            question=question.question,
                                                            note=question.evidence,
                                                            used_tables_and_columns= ', '.join(extracted_columns))
    elif method == "N":
        predict_sql_query_prompt = Prompts.PREDICT_SQL_PROMPT_N.format(table_info=schema_N(question, extracted_tables),
                                                            question=question.question,
                                                            note=question.evidence,
                                                            used_tables_and_columns= ', '.join(extracted_columns))

    if model == "A21":
        response, usage = call_a21(predict_sql_query_prompt)
    elif model == "SQLCODER":
        response, usage = call_sqlcoder(predict_sql_query_prompt)
    else:
        raise ValueError("Invalid model")

    return response

def predicted_query_no_COT(question: BirdQuestion, method: str = "ALL", model: str = "A21") -> str:
    # ========================== CHANGE THIS ==========================
    schema_d_info = schema_D(question)
    tables_info = json.load(open(table_info_file_path, 'r'))

    # print("======================= STEP 3 =======================")
    # Step 3 - Generate SQL query
    if method == "ALL":
        predict_sql_query_prompt = Prompts.PREDICT_SQL_NO_COT_PROMPT_ALL.format(table_info=schema_A_all(question),
                                                            question=question.question,
                                                            note=question.evidence)
    elif method == "VDT":
        predict_sql_query_prompt = Prompts.PREDICT_SQL_NO_COT_PROMPT_VDT.format(table_info=schema_VDT_all(question),
                                                            question=question.question,
                                                            note=question.evidence)
    elif method == "N":
        predict_sql_query_prompt = Prompts.PREDICT_SQL_NO_COT_PROMPT_N.format(table_info=schema_N_all(question),
                                                            question=question.question,
                                                            note=question.evidence)

    if model == "A21":
        response, usage = call_a21(predict_sql_query_prompt)
    elif model == "SQLCODER":
        response, usage = call_sqlcoder(predict_sql_query_prompt)
    else:
        raise ValueError("Invalid model")

    return response

def predicted_query_sqlcoder(question: BirdQuestion):
    predict_sql_query_prompt = Prompts.PREDICT_SQLCODER_PROMPT.format(table_metadata_string_DDL_statements=schema_A_all(question),
                                                                      user_question=question.question,
                                                                      external_knowledge=question.evidence)

    response, usage = call_sqlcoder(predict_sql_query_prompt)
    return response

def compare_queries(question: BirdQuestion, predicted_query: str) -> bool:
    # Open a connection to the SQLite database file
    sqlite_database = f"dev_datasets/{question.db_id}.sqlite"
    # print(sqlite_database)
    con = sqlite3.connect(sqlite_database)
    cur = con.cursor()

    # Run predicted SQL query
    try:
        cur.execute(predicted_query)
    except Exception as e:
        print(e)
        return False, e
    predicted_query_results = cur.fetchall()
    # print("Predicted SQL query:")
    # for row in predicted_query_results:
    #     print(row)

    # Run golden SQL query
    cur.execute(question.SQL)
    golden_query_results = cur.fetchall()
    # print("Golden SQL query:")
    # for row in golden_query_results:
    #     print(row)

    con.close()

    # Compare the results:
    if len(predicted_query_results) != len(golden_query_results):
        return False, None

    for i in range(len(predicted_query_results)):
        if predicted_query_results[i] != golden_query_results[i]:
            return False, None

    return True, None


# ---------- Process --------------------------------------------------------------------------------------------------


def process_dev_data(method:str, cot:bool, model:str, dev_file_path:str = dev_file_path, num:int = 300) ->list[dict]:
    results = []
    dev = json.load(open(dev_file_path, 'r'))
    relevate_dev_dataset = [question for question in dev if question['db_id'] in databases_to_check]
    if cot:
        file_name = f'{model}_results_cot_{method.lower()}_predict.pkl'
    else:
        file_name = f'{model}_results_no_cot_{method.lower()}_predict.pkl'
        
    for i, question in enumerate(relevate_dev_dataset[:num]):

        question = BirdQuestion(**question)
        if cot:
            predicted_sql = predicted_query(question, method, model)
        else:
            predicted_sql = predicted_query_no_COT(question, method, model)
            
        compare_response = compare_queries(question, predicted_sql)
        results.append({'question': question,
                        'sucess_status': compare_response[0],
                        'reason': compare_response[1]})

        if i % 10 == 0:
            print(i)
            # Pickle the object
            with open(file_name, 'wb') as file:
                pickle.dump(results, file)

    return results
