import json
import requests
import pandas as pd
import parseS2ORC
import get_full_text
import get_library

def bring_paper_data(paper_id):
    url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=abstract,year,authors,corpusId,title,tldr,venue'
    
    # Send a GET request to the API endpoint
    response = requests.get(url)

    # Check if the request was successful (status code 200 indicates success)
    if response.status_code == 200:
        data = response.json()
    else:
        print('Error occurred:', response.status_code)
    return data

def finding_introduction(data):
    introduction = []
    for item in data:
        introduction.append(item.get('text'))
    # concatenated = "\n\n".join(introduction)
    return introduction

def get_library_folder_data(folder_title, userId, id_type="long"):
    if id_type == "long":
        user_id = get_library.get_user_ids(userId)
        print(user_id[0])
        folder_list = get_library.get_library_folders(user_id[0])
        print(folder_list)
    else:
        user_id = [int(userId)]
        folder_list = get_library.get_library_folders(user_id[0])
        print(folder_list)
        
    library_id = next((item.get('library_folder_id') for item in folder_list if item.get('name') == folder_title), None)
    folder_papers = get_library.get_library_papers(user_id[0], [str(library_id)])
    paperId_list = [element["paper_id"] for element in folder_papers if element["status"] == "Created"]
    return paperId_list, library_id

def prepare_data(userId, folder_title):
    short_userId = userId.split("@")[0]
    paper_longidList, folder_id = get_library_folder_data(folder_title, userId)
    paper_data = []
    for index, paper_id in enumerate(paper_longidList):
        paper = {}
        paper_info = bring_paper_data(paper_id)
        paper["id"] = f"paper{index}"
        paper["title"] = paper_info["title"]
        paper["abstract"] = paper_info["abstract"]
        paper["authors"] = paper_info["authors"]
        paper["corpusId"] = paper_info["corpusId"]
        paper["year"] = paper_info["year"]
        paper["tldr"] = paper_info["tldr"]
        paper["venue"] = paper_info["venue"]
        paper["introduction"] = False
        paper_data.append(paper)
        
    lib_file_path = f"library_data/{short_userId}_{folder_title}.json"
    with open(lib_file_path, "w") as json_file:
        json.dump(paper_data, json_file)
    print("saved library list")
    return paper_data