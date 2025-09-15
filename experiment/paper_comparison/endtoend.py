import json
from typing import Any, Optional, Sequence, Dict, List
import openai
import requests
import itertools
import random
import re

from omegaconf import DictConfig

from paper_comparison.custom_types.table import Table
from paper_comparison.generation import load_json_file, generate, divide_column_num, make_paper_list_input, merge_tables, validate_table, baseline_create_json_format_template, mark_length_error, ours_create_json_format_template, validate_scheme, expand_hierarchy, validate_list_scheme

class BaseEndToEnd:
    def __init__(self, args: DictConfig):        
        pass
    
    def __call__(self, args, data) -> List[Table]:
        return []

class BaselineEndToEnd:
    def __init__(self, args: DictConfig):
        self.template = load_json_file(args.endtoend.prompt_path)
        
    def __call__(self, args, sample, tab_id, index, column_num, gold_caption) -> Table:
        """ Make a list of tables based on the paper data with baseline prompting method.
        
         Args:
            args (DictConfig): Arguments
            sample (Dict[str, Any]): a sample of a set of paper (sample["x"] is a list of papers)
            tab_id (str): Table ID 
            index (int): Index
            column_num (int): Number of gold columns
            gold_caption (Optional[str]): Gold caption (default: None)
        Returns:
            List[Table]: List of tables
            
        """
        table_list = []
        for i in range(args.endtoend.num_commonality):
            if "single_call" in args.endtoend.baseline_type:
                if column_num * len(sample["x"]) < args.endtoend.max_length:    # when the number of columns and papers is enough to be generated in one table
                    baseline_table = self.baseline_tab_gen(sample["x"], args.endtoend.model_type, index, tab_id, column_num, gold_caption, source="abstract")
                    baseline_table["error_counts"] = mark_length_error(baseline_table["error_counts"])  # classify which error type this table is included
                else:
                    print("The number of columns and papers is large")
                    
                    # Divide the total number of columns to be created into multiple turns to generate the columns over several turns.
                    column_list = divide_column_num(column_num, len(sample["x"]), args.endtoend.max_length)
                    
                    # Create and merge tables over multiple turns
                    baseline_table_set = []
                    for partial_column_num in column_list:
                        partial_table = self.baseline_tab_gen(sample["x"], args.endtoend.model_type, index, tab_id, partial_column_num, gold_caption, source="abstract")
                        baseline_table_set.append(partial_table)
                    baseline_table = merge_tables(baseline_table_set)
                    baseline_table["error_counts"] = mark_length_error(baseline_table["error_counts"])  # classify which error type this table is included
            table_list.append(baseline_table)
            
        return table_list
    
    def baseline_tab_gen(
        self, paper_list: List[Dict[str, Any]], model: str, 
        index: int, tab_id: str, column_num: int, 
        gold_caption: Optional[str] = None, source: str = "abs"
    ) -> Dict[str, Any]:
        """ Generate a table based on the given paper data with baseline prompting method.
        
         Args:
            paper_list (List[Dict[str, Any]]): List of papers
            model (str): Model type
            index (int): Index
            tab_id (str): Table ID
            column_num (int): Number of gold columns
            gold_caption (Optional[str]): Gold caption if there is
            source (str): Source of the paper information (abstract or introduction)
            
        Returns:
            Dict[str, Any], a predicted table
        """
        # when column_num * paper_num is less than 120, generate one table
        # when column_num * paper_num is more than 120, give half column number and generate two tables
        
        # set up error counts
        error_counts = {
            "length_error": 0,
            "json_error": 0,
            "paper_num_error": 0,
            "column_num_error": 0
        }
        
        # make a paper text that will be inputted to the prompt
        paper_text = ""
        for p_idx, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, p_idx, paper, source, "multiple")
        paper_text = f"[Start of the Paper Content]\n{paper_text}[End of the Paper Content]"

        merged_table_list = []  # list to store the tables generated over multiple turns if generated tables in prvious turns have not enough columns
        while (error_counts["length_error"] < 5):   # if the error count is less than 5, generate the table
            tmp_prompt = baseline_create_json_format_template(self.template["baseline_paper_to_table"]["prompt_simple"],
                                                 column_num, paper_list, paper_text)
            table = generate(tmp_prompt, model, "generation", "json", self.template["baseline_paper_to_table"])  
               
            is_valid, error_type = validate_table(table, paper_list, column_num)    # check if the table satisfies the requirements (no length issue, follow json format, matches paper and column number)
            if is_valid:
                merged_table_list.append(table)
                if len(merged_table_list) > 1:
                    baseline_table = merge_tables(merged_table_list)    
                else: 
                    baseline_table = {
                        "id": int(index), 
                        "tabid": str(tab_id), 
                        "schema": list(table.keys()), 
                        "table": table,
                        "gold_col": column_num, 
                        "predicted_col_num":len(list(table.keys())), 
                        "type": "single_call", 
                        "caption": "N/A", 
                        "error_counts": error_counts
                    }
                break
            elif error_type == "column_num_error":  # generate another table with the remaining number of columns (new_column_num = column_num - len(table.keys()))
                merged_table_list.append(table)
                column_num = column_num - len(table.keys())
                error_counts[error_type] += 1
            else:
                error_counts[error_type] += 1
                print(f"\t\t{tab_id} - Table Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                baseline_table = {
                    "id": int(index), 
                    "tabid": str(tab_id), 
                    "text": table, 
                    "error_counts": error_counts
                }
        return baseline_table

class HierarchicalEndToEnd(BaseEndToEnd):
    def __init__(self, args: DictConfig):
        self.template = load_json_file(args.endtoend.prompt_path)

    def __call__(self, args, sample, tab_id, index, column_num, gold_caption = None) -> List[Table]:
        """when the number of columns and papers is enough to be generated in one table, predict table in one turn.
           when the number of columns and papers is large, divide the total number of columns to be created into multiple turns to generate the columns over several turns.
           
            Args:
                args (DictConfig): Arguments
                sample (Dict[str, Any]): a sample of a set of paper (sample["x"] is a list of papers)
                tab_id (str): Table ID 
                index (int): Index
                column_num (int): Number of gold columns
                gold_caption (Optional[str]): Gold caption (default: None)
            Returns:
                List[Table]: List of tables
        """
        if column_num * len(sample["x"]) < args.endtoend.max_length:
            tables = self.table_prediction(args, tab_id, index, sample["x"], column_num, gold_caption=gold_caption)
        else:
            tables = self.table_prediction(args, tab_id, index, sample["x"], column_num, args.endtoend.max_length, gold_caption=gold_caption) 
        for table in tables:
            table["error_counts"] = mark_length_error(table["error_counts"]) 
        return tables
        
    def generate_commonality_attribute(self, tabid, paper_list: List[Dict[str, Any]], model:str, 
                                       num_column: int, difficulty: str = "hard", research_topic: Optional[str] = None, 
                                       source: str = "abstract", num_commonality: Optional[int] = 3
                                       ) -> Dict[str, Any]:
        """ Make a dictionary of commonality (kesy) and attribute (values) in one turn.
        
        Args:
            paper_list (List[Dict[str, Any]]): List of papers
            model (str): Model type
            num_column (int): Number of columns
            source (str): Source of the paper information (abstract or introduction)
        
        Returns:
            Dict[str], containing the commonality as key and the attributes as value ({"<commonality>":["<attribute>", ..]})
        """
        
        error_counts = {
            "scheme_length_error": 0,
            "scheme_json_error": 0,
            "scheme_unknown_error": 0,
        }

        paper_text = ""
        for p_idx, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, p_idx, paper, source, "multiple")
        paper_text = f"[Start of the Paper Content]\n{paper_text}[End of the Paper Content]"
        paper_titles = ""
        for p_idx, paper in enumerate(paper_list):
            paper_titles = make_paper_list_input(paper_titles, p_idx, paper, "title", "multiple")
        paper_titles = f"[Start of the Paper Content]\n{paper_titles}[End of the Paper Content]"
      
        max_try = 5
        for i in range(max_try):     # if the error count is less than 5, keep generating the scheme when the sceme errors occur
            if difficulty == "hard":
                
                # original commonality
                loaded_data = load_json_file("paper_comparison/test.json")
                # if tabid in loaded_data["attribute_gold_num"]:
                #     print("Already generated")
                #     continue
                # else:
                result = {}
                tmp_prompt = self.template["scheme_attribute_generation"]['prompt_abstract'].format(num_attributes=num_commonality, input_info=paper_text)
                print("we are here")
                original_commonality = generate(tmp_prompt, model, "generation", "json", self.template["scheme_attribute_generation"])
                for key, value in original_commonality.items():
                    tmp_prompt = self.template["hierarchical_scheme_generation"]['prompt_abstract'].format(num_attributes=num_column, 
                                                                                                        input_info=paper_titles, caption = key)
                    res = generate(tmp_prompt, model, "generation", "json", self.template["hierarchical_scheme_generation"])
                    print("ORIGINAL:\n", res, "\n\n")
                    result[key] = res
                
                loaded_data["attribute_gold_num"][tabid] = result
                with open("paper_comparison/test.json", "w") as f:
                    json.dump(loaded_data, f, indent=4)
                result = {}
                for key, value in result.items():
                    result[key] = expand_hierarchy(value)
                
            elif difficulty == "medium":
                tmp_prompt = self.template["hierarchical_scheme_generation"]['prompt_abstract'].format(num_attributes=num_column, input_info=paper_text, 
                                                                                                       caption = research_topic)
                res = generate(tmp_prompt, model, "generation", "json", self.template["hierarchical_scheme_generation"])
                # if res is not a list format, regenerate the scheme
                result[research_topic] = expand_hierarchy(res)
                
            is_valid, error_type = validate_scheme(result)
            if is_valid:
                print("Scheme generation success")
                scheme = result
                
            else:
                error_counts[error_type] += 1
                print(f"\t\tScheme Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                scheme = {"text": result, "error_type": error_type, "error_counts": error_counts}
        return scheme, error_counts
    
    def value_generation(
        self, paper_list: List[Dict[str, Any]], model: str, similarity: str, attributes: List[str], attribute_num: int, 
        id: int, tab_id: str, save_type: str, scheme_error_counts: Dict[str, Any]={}, source: str = "abstract", paper_loop: str = "multiple") -> Dict[str, Any]:
        """ Given a list of papers, model_type, similarity, attributes, source of each paper's information,
            make a table that is filled with values for each paper and each attribute.
        
         Args:
            paper_list (List[Dict[str, Any]]): List of papers
            model (str): Model type
            similarity (str): Generated commonality of the papers
            attributes (List[str]): List of attributes for the similarity
            attribute_num (int): Number of gold columns
            id (int): Index
            tab_id (str): Table ID
            save_type (str): Save type (experiment setup)
            scheme_error_counts (Dict[str, Any]): Error counts for the scheme generation
            source (str): Source of the paper information (abstract or introduction)
        
        Returns:
            Dict[str], table containing the attributes as key and the values from each paper as value ({"attribute": {"paper_1": ["value", ..], ..}})
        """
        prompt_type = "value_generation_qa"    # value_generation_qa or value_generation
        temp_template = "[Start of the Paper Content]\n{paper}[End of the Paper Content]\n\n[Start of Table Caption]\n{similarity}\n[End of Table Caption]\n\n[Start of Table Columns]\n{columns}\n[End of Table Columns]\n\n"
        col_names = '\n'.join([f'Column name {index+1}: {att}' for index, att in enumerate(attributes)])
        paper_text = ""
        error_counts = {
            "length_error": 0,
            "json_error": 0,
            "paper_num_error": 0,
            "column_num_error": 0
        }
        
        # add the scheme error counts to the error counts
        for key, value in scheme_error_counts.items():
            error_counts[key] = value
            
        for index, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, index, paper, source, paper_loop)
            
        merged_table_list = []
        while(error_counts["length_error"] < 5):
            # make prompt for the table generation (combine the paper text, commonalities, and column names)
            combined_prompt = ours_create_json_format_template(temp_template, self.template[prompt_type][f'prompt_{paper_loop}_{source}'], 
                                                                paper_text, len(paper_list), similarity, attributes)
            table = generate(combined_prompt, model, "generation", "json", self.template[prompt_type])
            is_valid, error_type = validate_table(table, paper_list, attribute_num)   # check if the table satisfies the requirements (no length issue, follow json format, matches paper and column number)
            if is_valid:
                merged_table_list.append(table)
                if len(merged_table_list) > 1:
                    our_table = merge_tables(merged_table_list)
                else:
                    our_table = {
                        "id": int(id), 
                        "tabid": str(tab_id), 
                        "schema": list(table.keys()), 
                        "table": table, 
                        "caption": similarity, 
                        "gold_col": attribute_num,
                        "predicted_col_num": len(list(table.keys())),
                        "type": save_type, 
                        "error_counts": error_counts
                    }
                    break
            elif error_type == "column_num_error":  # generate another table with the remaining number of columns (new_column_num = column_num - len(table.keys()))
                merged_table_list.append(table)
                attributes = [item for item in attributes if item not in list(table.keys())]
                attribute_num = len(attributes)
                error_counts[error_type] += 1
            else:   # if the error is not column_num_error, increase the error count and save output as an error
                error_counts[error_type] += 1
                print(f"\t\t{tab_id} - Table Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                print(table)
                our_table = {
                    "id": int(id), 
                    "tabid": str(tab_id),
                    "text": table, 
                    "error_counts": error_counts
                }
        return our_table

    def table_prediction(
        self, args, tab_id, 
        id, paper_list: List[Dict[str, Any]], col_num: int, 
        max_length: int = None, gold_caption: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """ Make a list of tables based on the paper data with our prompting method.
        
         Args:
            args: Arguments
            paper_list (List[Dict[str, Any]]): List of papers
            col_num (int): Number of columns
            max_length (int): Maximum length of the table
            gold_caption (Optional[str]): Gold caption
            id (int): Index
            tab_id (str): Table ID
            
        Returns:
            List[Dict[str, Any]], a list of predicted tables 
        """
        table_set = []
        scheme_set = {} 
        save_type = "{}_{}_{}_{}".format(args.endtoend.model_type, args.endtoend.num_commonality, args.endtoend.attribute_gen_method, args.endtoend.paper_loop)
        commonality_attribute_sets, scheme_error_counts = self.generate_commonality_attribute(tab_id, paper_list, args.endtoend.model_type, col_num, difficulty=args.difficulty, research_topic=gold_caption, source=args.endtoend.scheme_source)
        scheme_valid = False if "text" in list(commonality_attribute_sets.keys()) else True
        commonalities = list(commonality_attribute_sets.keys())
        # commonalities = self.filter_out(commonality_attribute_sets, "random", num_commonalities=args.endtoend.num_commonality)  #If more commonalities are generated, randomly sample commonalities

        if scheme_valid:
            # make a table for each commonality
            scheme_set = {} 
            for commonality in commonalities:
                attributes = commonality_attribute_sets[commonality]
                scheme_set[commonality] = attributes
                if max_length != None:     # when the value generation needs to be divided into multiple calls
                    start = 0
                    merged_table_set = []
                    
                    # Divide the total number of columns to be created into multiple turns to generate the columns over several turns.
                    column_list = divide_column_num(len(attributes), len(paper_list), max_length)  
                    for partial_column_num in column_list:
                        partial_attributes = attributes[start:start+partial_column_num]
                        start += partial_column_num
                        tab = self.value_generation(
                            paper_list, args.endtoend.model_type, 
                            commonality, partial_attributes, partial_column_num, 
                            id, tab_id, save_type, scheme_error_counts,
                            args.endtoend.value_source, args.endtoend.paper_loop
                        )
                        if "text" in tab:
                            break
                        merged_table_set.append(tab)
                        
                    table = merge_tables(merged_table_set)
                    
                else:   # when the value generation can be done in one call
                    table = self.value_generation(
                        paper_list, args.endtoend.model_type, 
                        commonality, attributes, len(attributes), 
                        id, tab_id, save_type, scheme_error_counts,
                        args.endtoend.value_source, args.endtoend.paper_loop
                    )
                table_set.append(table)
        else:
            print("Scheme generation failed")
            commonality_attribute_sets["id"] = int(id)
            table_set.append(commonality_attribute_sets)
        return table_set

class ComputedOutputsEndToEnd(BaseEndToEnd):
    def __init__(self, args: DictConfig):
        self.template = load_json_file(args.endtoend.prompt_path)

    def __call__(self, args, sample, tab_id, index, column_num, gold_caption = None) -> List[Table]:
        """when the number of columns and papers is enough to be generated in one table, predict table in one turn.
           when the number of columns and papers is large, divide the total number of columns to be created into multiple turns to generate the columns over several turns.
           
            Args:
                args (DictConfig): Arguments
                sample (Dict[str, Any]): a sample of a set of paper (sample["x"] is a list of papers)
                tab_id (str): Table ID 
                index (int): Index
                column_num (int): Number of gold columns
                gold_caption (Optional[str]): Gold caption (default: None)
            Returns:
                List[Table]: List of tables
        """
        
        if column_num * len(sample["x"]) < args.endtoend.max_length:
            tables = self.table_prediction(args, tab_id, index, sample, column_num, gold_caption=gold_caption)
        else:
            tables = self.table_prediction(args, tab_id, index, sample, column_num, args.endtoend.max_length, gold_caption=gold_caption) 
        for table in tables:
            table["error_counts"] = mark_length_error(table["error_counts"]) 
        return tables
        
    def generate_commonality_attribute(self, paper_list: List[Dict[str, Any]], model:str, 
                                       num_column: int, difficulty: str = "hard", research_topic: Optional[str] = None, source: str = "abstract"
                                       ) -> Dict[str, Any]:
        """ Make a dictionary of commonality (kesy) and attribute (values) in one turn.
        
        Args:
            paper_list (List[Dict[str, Any]]): List of papers
            model (str): Model type
            num_column (int): Number of columns
            source (str): Source of the paper information (abstract or introduction)
        
        Returns:
            Dict[str], containing the commonality as key and the attributes as value ({"commonality":["attribute", ..]})
        """
        
        error_counts = {
            "scheme_length_error": 0,
            "scheme_json_error": 0,
            "scheme_unknown_error": 0,
        }

        paper_text = ""
        for p_idx, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, p_idx, paper, source, "multiple")
        paper_text = f"[Start of the Paper Content]\n{paper_text}[End of the Paper Content]"

        # if difficulty == "hard":
        #     tmp_prompt = self.template["scheme_attribute_generation"]['prompt_abstract'].format(num_attributes=num_column, input_info=paper_text)
        # elif difficulty == "medium":
        #     tmp_prompt = self.template["find_attributes"]['prompt2'].format(input_info=paper_text, research_topic=research_topic, num_attributes=num_column)
        
        max_try = 5
        for i in range(max_try):     # if the error count is less than 5, keep generating the scheme when the sceme errors occur
            if difficulty == "hard":
                tmp_prompt = self.template["scheme_attribute_generation"]['prompt_abstract'].format(num_attributes=num_column, input_info=paper_text)
                result = generate(tmp_prompt, model, "generation", "json", self.template["scheme_attribute_generation"])
            elif difficulty == "medium":
                tmp_prompt = self.template["find_attributes"]['prompt2'].format(input_info=paper_text, research_topic=research_topic, num_attributes=num_column)
                res = generate(tmp_prompt, model, "generation", "json", self.template["find_attributes"])
                # if res is not a list format, regenerate the scheme
                result = {research_topic: res["attributes"]}

            is_valid, error_type = validate_scheme(result)
            if is_valid:
                print("Scheme generation success")
                scheme = result
                break
            else:
                error_counts[error_type] += 1
                print(f"\t\tScheme Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                scheme = {"text": result, "error_type": error_type, "error_counts": error_counts}
        return scheme, error_counts

    def generate_caption_based_attribute(self, sample, model:str, 
                                       num_column: int, difficulty: str = "caption", research_topic: Optional[str] = None, 
                                       num_commonality: Optional[int] = 3, source: str = "abstract"
                                       ) -> Dict[str, Any]:
        """ Make a dictionary of commonality (kesy) and attribute (values) in one turn.
        
        Args:
            paper_list (List[Dict[str, Any]]): List of papers
            model (str): Model type
            num_column (int): Number of columns
            source (str): Source of the paper information (abstract or introduction)
        
        Returns:
            Dict[str], containing the commonality as key and the attributes as value ({"<commonality>":["<attribute>", ..]})
        """
        
        error_counts = {
            "scheme_length_error": 0,
            "scheme_json_error": 0,
            "scheme_unknown_error": 0,
        }
        paper_list = sample["x"]
        paper_text = ""
        in_text_references = ""
        ics_captions = ""
        ics_papers = ""
        print(f"caption: {research_topic}\n")
        for p_idx, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, p_idx, paper, source, "multiple")
        paper_text = f"[Paper Content]\n{paper_text}"
        for item in sample["y"].intextref:
            new_text = re.sub(r'\{\{table:[\w-]+\}\} ', '', item["text"])
            in_text_references += f'{item["section"]}: {new_text}'
        for i, tab in enumerate(sample["y"].icspaper):
            tab["table_json"].pop("References", None)
            ics_papers += f'[Example Table {i}]\n{tab["table_json"]}\n\n'
        for i, tab in enumerate(sample["y"].icscaption):
            tab["table_json"].pop("References", None)
            ics_captions += f'[Example Table {i}]\n{tab["table_json"]}\n\n'

        max_try = 5
        for i in range(max_try):     # if the error count is less than 5, keep generating the scheme when the sceme errors occur
            if difficulty == "caption":
                tmp_prompt = self.template["gold_caption_in_text_ref"]['prompt_abstract'].format(num_attributes=num_column, input_info=paper_text, 
                                                                                                caption = research_topic, in_text_ref=in_text_references)
                
                # if res is not a list format, regenerate the scheme
            elif difficulty == "ics_caption":
                tmp_prompt = self.template["incontext_examples"]['prompt_abstract'].format(num_attributes=num_column, input_info=paper_text, 
                    example_tables = ics_captions)
                print(tmp_prompt)
            elif difficulty == "ics_paper":
                tmp_prompt = self.template["incontext_examples"]['prompt_abstract'].format(num_attributes=num_column, input_info=paper_text, 
                    example_tables = ics_papers)
                print(tmp_prompt)
            
            res = generate(tmp_prompt, model, "generation", "list", self.template["gold_caption_in_text_ref"])
            
            is_valid, error_type = validate_list_scheme(res)
            if is_valid:
                print("Scheme generation success")
                scheme = res
                break
            else:
                error_counts[error_type] += 1
                print(f"\t\tScheme Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                scheme = {"text": res, "error_type": error_type, "error_counts": error_counts}
        return scheme, error_counts
    
        
    def value_generation(
        self, paper_list: List[Dict[str, Any]], model: str, similarity: str, attributes: List[str], attribute_num: int, 
        id: int, tab_id: str, save_type: str, scheme_error_counts: Dict[str, Any]={}, source: str = "abstract", paper_loop: str = "multiple") -> Dict[str, Any]:
        """ Given a list of papers, model_type, similarity, attributes, source of each paper's information,
            make a table that is filled with values for each paper and each attribute.
        
         Args:
            paper_list (List[Dict[str, Any]]): List of papers
            model (str): Model type
            similarity (str): Generated commonality of the papers
            attributes (List[str]): List of attributes for the similarity
            attribute_num (int): Number of gold columns
            id (int): Index
            tab_id (str): Table ID
            save_type (str): Save type (experiment setup)
            scheme_error_counts (Dict[str, Any]): Error counts for the scheme generation
            source (str): Source of the paper information (abstract or introduction)
        
        Returns:
            Dict[str], table containing the attributes as key and the values from each paper as value ({"attribute": {"paper_1": ["value", ..], ..}})
        """
        prompt_type = "value_generation_qa"    # value_generation_qa or value_generation
        temp_template = "[Start of the Paper Content]\n{paper}[End of the Paper Content]\n\n[Start of Table Caption]\n{similarity}\n[End of Table Caption]\n\n[Start of Table Columns]\n{columns}\n[End of Table Columns]\n\n"
        col_names = '\n'.join([f'Column name {index+1}: {att}' for index, att in enumerate(attributes)])
        paper_text = ""
        error_counts = {
            "length_error": 0,
            "json_error": 0,
            "paper_num_error": 0,
            "column_num_error": 0
        }
        
        # add the scheme error counts to the error counts
        for key, value in scheme_error_counts.items():
            error_counts[key] = value
            
        for index, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, index, paper, source, paper_loop)
            
        merged_table_list = []
        while(error_counts["length_error"] < 5):
            # make prompt for the table generation (combine the paper text, commonalities, and column names)
            combined_prompt = ours_create_json_format_template(temp_template, self.template[prompt_type][f'prompt_{paper_loop}_{source}'], 
                                                                paper_text, len(paper_list), similarity, attributes)
            table = generate(combined_prompt, model, "generation", "json", self.template[prompt_type])
            is_valid, error_type = validate_table(table, paper_list, attribute_num)   # check if the table satisfies the requirements (no length issue, follow json format, matches paper and column number)
            if is_valid:
                merged_table_list.append(table)
                if len(merged_table_list) > 1:
                    our_table = merge_tables(merged_table_list)
                else:
                    our_table = {
                        "id": int(id), 
                        "tabid": str(tab_id), 
                        "schema": list(table.keys()), 
                        "table": table, 
                        "caption": similarity, 
                        "gold_col": attribute_num,
                        "predicted_col_num": len(list(table.keys())),
                        "type": save_type, 
                        "error_counts": error_counts
                    }
                    break
            elif error_type == "column_num_error":  # generate another table with the remaining number of columns (new_column_num = column_num - len(table.keys()))
                merged_table_list.append(table)
                attributes = [item for item in attributes if item not in list(table.keys())]
                attribute_num = len(attributes)
                error_counts[error_type] += 1
            else:   # if the error is not column_num_error, increase the error count and save output as an error
                error_counts[error_type] += 1
                print(f"\t\t{tab_id} - Table Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                print(table)
                our_table = {
                    "id": int(id), 
                    "tabid": str(tab_id),
                    "text": table, 
                    "error_counts": error_counts
                }
        return our_table
        
    def filter_out(self, commonality_attribute_sets: Dict, method: str, num_commonalities: Optional[int] = 1) -> List[Any]:
        commonalities = list(commonality_attribute_sets.keys())
        if len(commonalities) < num_commonalities:
            return commonalities
        else:
            if method == "random":
                # random sampling one of the commonalities
                return random.sample(commonalities, num_commonalities)
                
            elif method == "best":   # need revision: identify elements that are different from each other
                # based on the information of the papers, commonalities, and attributes, return the best commonality index
                # idx = generate_best_commonality(commonality_attribute_sets)
                # commonalities = commonalities[idx]
                return []
            
    def empty_value_generation(
        self, paper_list: List[Dict[str, Any]], attributes: List[str], attribute_num: int, caption: str, 
        id: int, tab_id: str, save_type: str, scheme_error_counts: Dict[str, Any]={}) -> Dict[str, Any]:
        """ Given a list of papers and attributes,
            make a table that is filled with empty values for each paper and each attribute.
        
         Args:
            paper_list (List[Dict[str, Any]]): List of papers
            attributes (List[str]): List of attributes for the similarity
            attribute_num (int): Number of gold columns
            caption: Gold caption
            id (int): Index
            tab_id (str): Table ID
            save_type (str): Save type (experiment setup)
            scheme_error_counts (Dict[str, Any]): Error counts for the scheme generation
        
        Returns:
            Dict[str], table containing the attributes as key and the empty values from each paper as value ({"attribute": {"paper_1": [], ..}})
        """
        error_counts = {
            "length_error": 0,
            "json_error": 0,
            "paper_num_error": 0,
            "column_num_error": 0
        }
        
        # add the scheme error counts to the error counts
        for key, value in scheme_error_counts.items():
            error_counts[key] = value
                     
        our_table = {
            "id": int(id), 
            "tabid": str(tab_id), 
            "schema": attributes, 
            "table": {att: {f"paper_{index}": [] for index in range(len(paper_list))} for att in attributes}, 
            "caption": caption, 
            "gold_col": attribute_num,
            "predicted_col_num": len(attributes),
            "type": save_type, 
            "error_counts": error_counts
        }
        return our_table
        
        
    def table_prediction(
        self, args, tab_id, 
        id, sample: Dict[str, Any], col_num: int, 
        max_length: int = None, gold_caption: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """ Make a list of tables based on the paper data with our prompting method.
        
         Args:
            args: Arguments
            sample (Dict[str, Any]): "x": list of papers, "y": table data
            col_num (int): Number of columns
            max_length (int): Maximum length of the table
            gold_caption (Optional[str]): Gold caption
            id (int): Index
            tab_id (str): Table ID
            
        Returns:
            List[Dict[str, Any]], a list of predicted tables 
        """
        paper_list = sample["x"]
        table_set = []
        scheme_set = {} 
        save_type = "{}_{}_{}_{}".format(args.endtoend.model_type, args.endtoend.num_commonality, args.endtoend.attribute_gen_method, args.endtoend.paper_loop)
        if args.difficulty == "hard":
            commonality_attribute_sets, scheme_error_counts = self.generate_commonality_attribute(paper_list, args.endtoend.model_type, col_num, difficulty=args.difficulty, research_topic=gold_caption, source=args.endtoend.scheme_source)
            scheme_valid = False if "text" in list(commonality_attribute_sets.keys()) else True
            commonalities = list(commonality_attribute_sets.keys())
            commonalities = self.filter_out(commonality_attribute_sets, "random", num_commonalities=args.endtoend.num_commonality)  #If more commonalities are generated, randomly sample commonalities
        else:   # caption and ics
            attributes, scheme_error_counts = self.generate_caption_based_attribute(sample, args.endtoend.model_type, 
                                       col_num, args.difficulty, gold_caption, 
                                       num_commonality= 1, source=args.endtoend.scheme_source)
            
            scheme_valid = False if isinstance(attributes, dict) else True
            commonalities = [gold_caption]
            commonality_attribute_sets = {gold_caption: attributes}
            
        if scheme_valid:
            # make a table for each commonality
            scheme_set = {} 
            for commonality in commonalities:
                if args.difficulty == "hard":
                    attributes = commonality_attribute_sets[commonality]
                # else:
                #     attributes, scheme_error_counts = self.generate_caption_based_attribute(sample, args.endtoend.model_type, 
                #                        col_num, args.difficulty, gold_caption, 
                #                        source="abstract", num_commonality= 1)
                scheme_set[commonality] = attributes   
                table = self.empty_value_generation(
                    paper_list, attributes, col_num, commonality, 
                    id, tab_id, save_type, scheme_error_counts
                )
                table_set.append(table)
        else:
            print("Scheme generation failed")
            if args.difficulty == "hard":
                commonality_attribute_sets["id"] = int(id)
                table_set.append(commonality_attribute_sets)
            else:
                attributes["id"] = int(id)
                table_set.append(attributes)
        return table_set
        
class DebugAbstractsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> List[Table]:
        table_values = {
            "Studies decontextualization?": {"choi21": ["yes"], "newman23": ["yes"], "potluri23": ["no"]},
            "What is their data source?": {
                "choi21": ["Wikipedia"],
                "newman23": ["Scientific Papers"],
                "potluri23": ["ELI5"],
            },
            "What field are they in?": {"choi21": ["NLP"], "newman23": ["NLP"], "potluri23": ["NLP"]},
            "What task do they study?": {
                "choi21": ["decontextualization"],
                "newman23": ["decontextualization"],
                "potluri23": ["long-answer summarization"],
            },
        }
        return [
            Table(
                tabid="0",
                schema=set(table_values.keys()),
                values=table_values,
            )
        ]
    

class PrecomputedOutputsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> list[Table]:
        with open(args.endtoend.path) as f:
            table_values = json.load(f)

        # the baselines (baseline_paper_to_table_max.json, baseline_paper_to_cc_tab_max.json) contain only the tables
        # so, we don't need to extract them, but for Our algorithm (ours_output_decontext.json) the file contains other
        # info
        if "final_table" in table_values:
            table_values = table_values["final_table"]
            for attribute in table_values:
                del table_values[attribute]["type"]
                del table_values[attribute]["presup"]

        return [
            Table(
                tabid="0",
                schema=table_values.keys(),
                values=table_values,
            )
        ]


class OracleEndToEnd(BaseEndToEnd):
    """Returns the gold tables"""

    def __call__(self, args, data) -> list[Table]:
        return [sample["y"] for sample in data]


def load_endtoend(args: DictConfig):
    if args.endtoend.name == "debug_abstracts":
        return DebugAbstractsEndToEnd(args)
    elif args.endtoend.name == "precomp_outputs":
        return PrecomputedOutputsEndToEnd(args)
    elif args.endtoend.name == "oracle":
        return OracleEndToEnd(args)
    elif args.endtoend.name == "ours_outputs":
        return ComputedOutputsEndToEnd(args)
    elif args.endtoend.name == "baseline_outputs":
        return BaselineEndToEnd(args)
    elif args.endtoend.name =="hierarchical_outputs":
        return HierarchicalEndToEnd(args)
    return BaseEndToEnd(args)
