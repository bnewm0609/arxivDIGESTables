import json
import os
from typing import Any, Dict, List, Optional, Sequence

import openai
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential


def load_json_file(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def validate_table(table, paper_list, column_num):
    """Validate whether the predicted table is in the correct format and return the error type if not."""
    if isinstance(table, str):
        if table.strip()[-1] != "}":
            # print(table)
            return False, "length_error"
        else:
            # print(table)
            return False, "json_error"
    elif (
        isinstance(table, dict)
        and len(paper_list) == len(list(table.values())[0])
        and len(list(table.keys())) >= column_num
    ):
        return True, ""
    elif (
        isinstance(table, dict)
        and len(paper_list) != len(list(table.values())[0])
        and len(list(table.keys())) < column_num
    ):
        return False, "column_num_error"
    else:
        # print(table)
        return False, "paper_num_error"

def check_dict_values_are_lists(input_dict):
    """
    Check if all values in the dictionary are of list type.

    Args:
    input_dict (dict): The dictionary to check.

    Returns:
    bool: True if all values are lists, False otherwise.
    """
    return all(isinstance(value, list) for value in input_dict.values())

def validate_list_scheme(scheme):
    """Validate whether the predicted scheme is in the correct list format and return the error type if not.

    """
    if isinstance(scheme, list):    
        return True, ""
    elif isinstance(scheme, str): 
        # if there is [JSON] once in the table
        if scheme.strip()[-1] != "}": # length issue - generate once more
            return False, "scheme_length_error"
        else:
            # print(scheme)
            return False, "scheme_json_error"
    else:
        # print(scheme)
        return False, "scheme_unknown_error"
    
def validate_scheme(scheme):
    """Validate whether the predicted scheme is in the correct format and return the error type if not.

    """
    if isinstance(scheme, dict) and check_dict_values_are_lists(scheme):    
        return True, ""
    elif isinstance(scheme, str):
        # if there is [JSON] once in the table
        if scheme.strip()[-1] != "}":  # length issue - generate once more
            return False, "scheme_length_error"
        else:
            # print(scheme)
            return False, "scheme_json_error"
    else:
        # print(scheme)
        return False, "scheme_unknown_error"


def str_to_json(text, parse_str):
    """Make generated string to json format using parse_str marker."""
    try:
        if parse_str == "```json":
            if parse_str in text:
                json_str = text.split(parse_str)[1].strip()
                json_str = json_str.split("```")[0].strip()
            else:
                json_str = text.split("```")[1].strip()
        elif parse_str == "```list":
            if parse_str in text:
                json_str = text.split(parse_str)[1].strip()
                json_str = json_str.split("```")[0].strip()
            # elif "```" not in text:
            #     json_str = json_str
            else:
                json_str = text.split("```")[1].strip()
        else:
            json_str = text.split(parse_str)[1].strip()
        return json.loads(json_str)
    except:
        # potential parse_str: ["[\JSON]"]
        potential_parse_str = ["[/JSON]", "{/JSON}", "[/json]"]
        for p_str in potential_parse_str:
            try:
                json_str = text.split(parse_str)[1].split(p_str)[0].strip()
                # json_str = text.split(p_str)[1].strip()
                return json.loads(json_str)
            except:
                continue
        print(text)
        print('\t\tFailed to parse with the given parse_str')
        return text
    
def str_to_list(text, parse_str):
    """Make generated string to json format using parse_str marker.
    """
    try:
        if parse_str == "```list":
            if parse_str in text:
                list_str = text.split(parse_str)[1].strip()
                list_str = list_str.split("```")[0].strip()
            else:
                list_str = text.split("```")[1].strip()
        elif parse_str == "```list":
            if parse_str in text:
                list_str = text.split(parse_str)[1].strip()
                list_str = list_str.split("```")[0].strip()
            # elif "```" not in text:
            #     json_str = json_str
            else:
                list_str = text.split("```")[1].strip()
        else:
            list_str = text.split(parse_str)[1].strip()
        return json.loads(list_str)
    except:
        # potential parse_str: ["[\JSON]"]
        potential_parse_str = ["[/JSON]", "{/JSON}", "[/json]"]
        for p_str in potential_parse_str:
            try:
                json_str = text.split(parse_str)[1].split(p_str)[0].strip()
                # json_str = text.split(p_str)[1].strip()
                return json.loads(json_str)
            except:
                continue
        print(text)
        print('\t\tFailed to parse with the given parse_str')
        return text


def make_paper_list_input(paper_text: str, index: int, paper: Dict, source: str, paper_loop: str) -> str:
    """Make paper list to input format so that it can be used in the prompt."""
    abstract = paper["abstract"].strip() if "abstract" in paper and paper["abstract"] else None
    introduction = paper["introduction"].strip() if "introduction" in paper and paper["introduction"] else None
    full_text = paper["full_text"].strip() if "full_text" in paper and paper["full_text"] else None
    title = paper["title"]

    index_str = f"{str(index + 1)} " if paper_loop == "multiple" else ""
    intro_text = f"Paper {index_str}text: {introduction}\n"
    full_text = f"Paper {index_str}text: {full_text}\n"

    paper_text = (
        f"Paper {index_str}title: {title}\n"
        f"Paper {index_str}abstract: {abstract}\n"
        f"{intro_text if (source == 'intro' and introduction is not None) else ''}"
        f"{full_text if (source == 'full' and full_text is not None) else ''}"
        "\n"
    )

    return paper_text


def divide_column_num(column_num, paper_num, max_length):
    """Divide the column number based on the paper number and the maximum length of the prompt."""
    division = (column_num * paper_num) // max_length + 1
    base = column_num // division
    remainder = column_num % division
    column_list = [base + 1 if i < remainder else base for i in range(division)]
    return column_list


def baseline_create_json_format_template(template, column_num, paper_list, paper_text, gold_caption=None):
    """Create JSON format template that needs to be filled in for the baseline prompt."""
    json_format = {}
    paper_num = len(paper_list)
    for i in range(column_num):
        json_format[f"<dimension {i+1} that can compare papers>"] = {}
        for p_idx in range(len(paper_list)):
            json_format[f"<dimension {i+1} that can compare papers>"][f"paper_{p_idx+1}"] = [
                f"<relevant value to the dimension {i+1} grounded on Paper {p_idx+1}>"
            ]
    json_format = json.dumps(json_format, indent=2)

    if type(gold_caption) == str:
        tmp_prompt = template.format(
            col_num=column_num,
            paper_num=paper_num,
            input_info=paper_text,
            caption=gold_caption,
            json_format=json_format,
        )
    else:
        tmp_prompt = template.format(
            col_num=column_num, paper_num=paper_num, input_info=paper_text, json_format=json_format
        )
    return tmp_prompt


def ours_create_json_format_template(partial_template, template, paper_text, paper_num, similarity, attributes):
    """Create JSON format template that needs to be filled in for our prompting method."""
    col_names = "\n".join([f"Column name {index+1}: {att}" for index, att in enumerate(attributes)])
    input_info = partial_template.format(paper=paper_text, similarity=similarity, columns=col_names)
    # Create JSON format template to add to the prompt
    json_format = {}
    for att in attributes:
        json_format[att] = {}
        for i in range(paper_num):
            json_format[att][f"paper_{i+1}"] = [f"<value for this column grounded on Paper {i+1}>"]
    json_format = json.dumps(json_format, indent=2)
    combined_prompt = template.format(input_info=input_info, json_format=json_format)
    return combined_prompt


def merge_tables(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple tables that have the correct format and same number of rows into one table."""
    is_error = any(["text" in table for table in tables])
    if not is_error:
        merged_table = {
            "id": tables[0]["id"],
            "tabid": tables[0]["tabid"],
            "caption": tables[0]["caption"],
            "schema": [],
            "table": {},
            "gold_col": 0,
            "predicted_col_num": 0,
            "error_counts": {},
        }
    # check if there is a text key in any of the tables
    else:
        merged_table = {
            "id": tables[0]["id"],
            "tabid": tables[0]["tabid"],
            "text": tables[0]["text"],
            "error_counts": {},
        }

    for table in tables:
        if not is_error:
            merged_table["schema"].extend(table["schema"])
            merged_table["table"].update(table["table"])
            merged_table["gold_col"] += table["gold_col"]
            merged_table["predicted_col_num"] += table["predicted_col_num"]
            merged_table["type"] = table["type"]

        for key, value in table["error_counts"].items():
            if key in merged_table["error_counts"]:
                merged_table["error_counts"][key] += value
            else:
                merged_table["error_counts"][key] = value

    return merged_table


def mark_length_error(error_data):
    """Mark the error type as length error if the length of the table is over the maximum length."""
    if error_data["length_error"] == 5:
        error_data["over_max_length_error"] = True
        error_data["have_length_error"] = True
    elif error_data["length_error"] > 0 and error_data["length_error"] < 5:
        error_data["over_max_length_error"] = False
        error_data["have_length_error"] = True
    else:
        error_data["over_max_length_error"] = False
        error_data["have_length_error"] = False
    return error_data

def expand_hierarchy(result):
    """Expand the dictionary to list
    """
    expanded = []
    for key, value in result.items():
        expanded.append(key)
        expanded.extend(value)
    return expanded
   

def generate(tmp_prompt, model_type, generation_type, data_type, template=None):
    return generate_handler(tmp_prompt, model_type, generation_type, data_type, template)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_handler(tmp_prompt, model_type, generation_type, data_type, template=None):
    """Generate outputs using the given model_type and prompt."""
    explanation = ""
    if model_type == "gpt4" or model_type == "gpt3.5":
        api_key = os.environ["OPENAI_KEY"]
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(api_key)}
        model = "gpt-4-1106-preview" if model_type == "gpt4" else "gpt-3.5-turbo-1106"
    elif model_type == "mixtral":
        api_key = os.environ["TOGETHER_API_KEY"]
        url = "https://api.together.xyz"
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(api_key)}
    elif model_type == "llama":
        api_key = os.environ["TOGETHER_API_KEY"]
        url = "https://api.together.xyz/v1/chat/completions"
        model = "meta-llama/Llama-2-70b-chat-hf"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(api_key)}

    try:
        if template["system_instruction"] == None:
            prompt = [{"role": "user", "content": tmp_prompt}]
        else:
            prompt = [
                {"role": "assistant", "content": template["system_instruction"]},
                {"role": "user", "content": tmp_prompt},
            ]
        if generation_type == "verification":
            temperature = 0.3
            max_tokens = 30
        elif generation_type == "specificity":
            temperature = 0.3
            max_tokens = 1000
        else:
            temperature = 1
            max_tokens = 3000

        if model_type == "gpt4" or model_type == "gpt3.5":
            data = {"messages": prompt, "model": model, "max_tokens": max_tokens, "temperature": temperature}
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raises a HTTPError if the response contains an HTTP error status code
            output = response.json()
            if "choices" in output:
                for choice in output["choices"]:
                    message = choice["message"]
                    if message["role"] == "assistant":
                        explanation = message["content"]
        elif model_type == "mixtral":
            client = openai.OpenAI(api_key=api_key, base_url=url)
            chat_completion = client.chat.completions.create(
                model=model, messages=prompt, max_tokens=max_tokens, temperature=temperature
            )
            explanation = chat_completion.choices[0].message.content
        elif model_type == "llama":
            data = {
                "prompt": f"[INST] {prompt} [/INST]",
                "model": model,
                "max_tokens": 650,
                "temperature": temperature,
                "stop": ["[/INST]", "</s>"],
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            output = response.json()
            if 'choices' in output:
                for choice in output['choices']:
                    message = choice['message']
                    if message['role'] == 'assistant':
                        explanation = message['content'] 
            print("generation is completed")  
        print(explanation.strip())
        if data_type == "list":
            return str_to_list(explanation.strip(), template["parse_str"])  

        else:
            return str_to_list(explanation, template["parse_str"])  
            
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(json.dumps(data, indent=2))
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err
