from eml_parser import EmlParser
from pandas import read_json, read_html, DataFrame
from json import load

def parse_eml(email_path: list):
    """
    Parses eml files into dictionary
    """
    emails = []

    for i in email_path:

        email = EmlParser(True,True,include_href=True).decode_email(i)
        emails.append(email)
    
    return emails

def read_output(json_path: list):
    """
    Takes in list of json paths
    """
    for i in json_path:
        json_list: list[dict] = load(i)

        

