# Inspired by https://github.com/Hermie-Wormie/Phishing-Email-Detector

import base64
import requests
from constants import API_KEY, VT_FILE, VT_URL


def check_hash(file_hash: str):
    """
    Sends a request to the VirusTotal API to check if the file with the provided hash 
    has been analyzed before and returns the result.

    Parameters:
    file_hash (str): The hash of the file to be checked (SHA-256)
        Note - MD5 or SHA-1 will work as well but file_to_hash only hashes to SHA-256

    Returns [Tied to hash_scan_results(result)]:
    tuple: If the file is found and detected as malicious, returns (number_of_positives, VirusTotal_permalink).
    str: If the file is clean or not found in VirusTotal's database, returns an explanatory string.
    str: If there is an error with the request, returns an error message.

    How it works:
        - Prepares a request with the API key and file hash as parameters.
        - An HTTP GET request is sent to VirusTotal's API endpoint using the file hash as the resource.
        - If the response is successful (HTTP status 200), parses the JSON result and processes it.
        - The function then calls `hash_scan_results(result)` to interpret the result.
        - If the response is empty, a message is returned indicating no data was received from VirusTotal.
        - If the request fails, an error message with the HTTP status code is returned.

    Side Effects:
        - Sends an HTTP request to VirusTotal's API.

    Exceptions:
        - If the request fails, it returns an HTTP error status code message.
    """
    headers = {'accept': 'application/json', 'x-apikey': API_KEY}
    url = VT_FILE + file_hash
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        result = response.json()
        if result:
            return hash_scan_results(result)

        else:
            return "Empty response from VirusTotal."
    else:
        return f"Error: {response.status_code} from VirusTotal"


def hash_scan_results(result):
    """
    Interprets the result from VirusTotal API for a given file hash and determines whether the file is malicious.

    Parameters:
    result (dict): The JSON response from VirusTotal containing scan information for the file.

    Returns:
    tuple: (positives, permalink) if the file is malicious, where:
        - positives (int): The number of antivirus engines that flagged the file as malicious.
        - permalink (str): A direct link to the VirusTotal report for this file.

    str: "Hash of file not found in VirusTotal database." 
        - if the file hash is not present in VirusTotal's database.
    """
    results = {}
    try:
        if result["data"]["attributes"]:
            if result["data"]["attributes"]["last_analysis_stats"]["malicious"] > 0:
                for x,y in result["data"]["attributes"]["last_analysis_results"].items():
                    if y["category"] == "malicious":
                        results.update({x:y})
                
                return results
            
            elif result["data"]["attributes"]["last_analysis_stats"]["suspicious"] > 0:
                for x,y in result["data"]["attributes"]["last_analysis_results"].items():
                    if y["category"] == "suspicious":
                        results.update({x:y})
                
                return results

            else:
                for x,y in result["data"]["attributes"]["last_analysis_results"].items():
                    if y["category"] == "harmless":
                        results.update({x:y})
            
                return results
        
    except Exception as e:
        return "Hash of file not found in VirusTotal database."


def check_url(url: str):
    """
    Sends a request to the VirusTotal API to check the status of the given URL and returns the scan result.

    Parameters:
    url (str): The URL to be checked for malicious activity.

    Returns [Tied to url_scan_results(result)]:
    tuple: If the URL is flagged as malicious, returns (number_of_positives, VirusTotal_permalink).
    str: If the URL is clean or not found, returns an explanatory string.
    str: If there is an error with the request, returns an error message with the HTTP status code.

    How it works:
        - Prepares a request with the API key and URL as parameters.
        - Sends an HTTP GET request to the VirusTotal API endpoint for URL analysis.
        - If the response is successful (HTTP status 200), parses the JSON result and processes it.
        - The function then calls `url_scan_results(result)` to interpret the result.
        - If the response is empty, a message is returned indicating no data was received from VirusTotal.
        - If the request fails, an error message with the HTTP status code is returned.

    Side Effects:
    - Sends an HTTP request to VirusTotal's API.

    Exceptions:
    - If the request fails, returns the HTTP status code indicating the error.
    """
    headers = {'accept': 'application/json', 'x-apikey': API_KEY}
    url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    vt_url = VT_URL + url_id
    response = requests.get(vt_url, headers=headers)

    if response.status_code == 200:
        result = response.json()
        if result:
            return url_scan_results(result)

        else:
            return "Empty response from VirusTotal."
    else:
        return f"Error: {response.status_code} from VirusTotal"


def url_scan_results(result):
    """
    Interprets the result from VirusTotal for the given URL and determines whether it is malicious.

    Parameters:
    result (dict): The JSON response from VirusTotal containing scan information for the URL.

    Returns:
    tuple: (positives, permalink) if the URL is flagged as malicious, where:
        - positives (int): The number of antivirus engines that flagged the URL as malicious.
        - permalink (str): A direct link to the VirusTotal report for this URL.

    str: "URL not found in VirusTotal database" 
        - if the URL is not present in VirusTotal's database.

    How it works:
    - Checks if data attributes is present in the response, indicating the information found associated with the URL.
    - If detected as malicious, returns the results pertaining to malicious detections.
    - Else if detected as suspicious, returns the results pertaining to suspicious detections.
    - Else if detected as harmless, returns the results pertaining to malicious detections.
    - If the URL is not found in the database or if no result is returned, handles those cases appropriately.

    Side Effects:
    - None.
    """
    results = {}
    try:
        if result["data"]["attributes"]:
            if result["data"]["attributes"]["last_analysis_stats"]["malicious"] > 0:
                for x,y in result["data"]["attributes"]["last_analysis_results"].items():
                    if y["category"] == "malicious":
                        results.update({x:y})
                
                return results
            
            elif result["data"]["attributes"]["last_analysis_stats"]["suspicious"] > 0:
                for x,y in result["data"]["attributes"]["last_analysis_results"].items():
                    if y["category"] == "suspicious":
                        results.update({x:y})
                
                return results

            else:
                for x,y in result["data"]["attributes"]["last_analysis_results"].items():
                    if y["category"] == "harmless":
                        results.update({x:y})
            
                return results
        
    except Exception as e:
        print(e)
        return "URL not found in VirusTotal database"


if __name__ == "__main__":

    # Test malicious url:
    url = "http://www.myetherevvalliet.com/"
    print(check_url(url))

    # Test malicious hash:
    hash = "b1b74ff5c67cfdc16f1cb30db9b885046c4c5d71af575a2d3de4a79918c1ce89"
    # print(check_hash(hash))

    # Hash file then check VirusTotal
    # hash_of_file = file_to_hash(file_path)
    # print(check_hash(hash_of_file))
