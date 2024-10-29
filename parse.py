from eml_parser import EmlParser
from pandas import read_json, read_html, DataFrame
from json import load
from DATAMANIPULATION.data_clean import replace_whitespaces, replace_url, process_text, join_list
from functools import reduce
from bs4 import BeautifulSoup

def parse_eml(email_path: list):
    """
    Parses eml files into dictionary
    """
    emails = []

    for i in email_path:

        email = EmlParser(True,True,include_href=True).decode_email(i)
        emails.append(email)
    
    return emails

def soup(str):

    return BeautifulSoup(str, "html.parser").get_text()

def replace(str: str):
    """
    Combined functions used on text
    """
    return reduce(lambda x, func: func(x), [replace_url,replace_whitespaces,soup,process_text],str)

def read_output(json_path: list):
    """
    Takes in list of json paths
    """
    for i in json_path:
        with open(i,"r") as file:
            json_list: list[dict] = load(file)
    
    results = []
    for item_no, item in enumerate(json_list):
        result = {}
        result["id"] = item_no + 1

        try:
            result["subject"] = item["header"]["subject"]
            result["cleaned_subject"] = process_text(item["header"]["subject"])
        except Exception as e:
            print(e)
            result["subject"] = ""
            result["cleaned_subject"] = ""
        
        try:
            result["sender"] = item["header"]["from"]
        except Exception as e:
            print(e)
            result["sender"] = ""
        
        try:
            result["body"] = join_list([replace(i["content"]) for i in item["body"]])
        except Exception as e:
            print(e)
            result["body"] = ""

        try:
            result["url"] = [i["uri"] for i in item["body"]]
        except Exception as e:
            print(e)
            result["url"] = ""

        try:
            result["file_hash"] = [i["hash"] for i in item["body"]]
        except Exception as e:
            print(e)
            result["file_hash"] = ""
        
        # try:
        #     result["attachments"] = []
        # except Exception as e:
        #     print(e)
        
        results.append(result)
    
    return results
        

        

