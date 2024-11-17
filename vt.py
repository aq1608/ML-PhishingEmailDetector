# Adapted from another github https://github.com/Hermie-Wormie/Phishing-Email-Detector
# Will be changed soon

import hashlib
import requests

API_KEY = 'Your_VT_API_Key'
VT_URL = 'https://www.virustotal.com/vtapi/v2/url/report'
VT_FILE = 'https://www.virustotal.com/vtapi/v2/file/report'


# def file_to_hash(file_path):
#     """
#     Computes the SHA-256 hash of the file located at the given file path.

#     Parameters:
#     file_path (str): The path to the file that needs to be hashed.

#     Returns:
#     str: The SHA-256 hash of the file, represented as a hexadecimal string.

#     How it works:
#     - Opens the file in binary mode to ensure it reads the raw bytes.
#     - Reads the file in chunks (4096 bytes) to avoid loading large files entirely into memory.
#     - Updates the hash object incrementally with each chunk of data.
#     - After reading the entire file, returns the final SHA-256 hash as a hex string.

#     Exceptions:
#     - If the file does not exist or cannot be opened, an IOError (or FileNotFoundError) will be raised.
#     """
#     sha256_hash = hashlib.sha256()

#     with open(file_path, "rb") as file:
#         for byte_block in iter(lambda: file.read(4096), b""):
#             sha256_hash.update(byte_block)

#     return sha256_hash.hexdigest()


def check_hash(file_hash):
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
    params = {'apikey': API_KEY, 'resource': file_hash}
    response = requests.get(VT_FILE, params=params)

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

    str: "The file is clean." 
        - if no antivirus engines flagged the file as malicious.

    str: "Hash of file not found in VirusTotal database." 
        - if the file hash is not present in VirusTotal's database.
    """
    if result['response_code'] == 1:

        if result['positives'] > 0:
            num_positives = result['positives']
            vt_link = result['permalink']
            return num_positives, vt_link

        else:
            return "The file is clean."
        
    else:
        return "Hash of file not found in VirusTotal database."


def check_url(url):
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
    params = {'apikey': API_KEY, 'resource': url}
    response = requests.get(VT_URL, params=params)

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

    str: "Clean URL" 
        - if the URL is not flagged by any antivirus engines.

    str: "URL not found in VirusTotal database" 
        - if the URL is not present in VirusTotal's database.

    str: "No result returned from VirusTotal" 
        - if the result is unexpectedly empty.

    How it works:
    - Checks if the `response_code` is 1, indicating the URL was found in the VirusTotal database.
    - If positive detections exist, returns the number of positives and the report link.
    - If no positives, returns that the URL is clean.
    - If the URL is not found in the database or if no result is returned, handles those cases appropriately.

    Side Effects:
    - None.
    """
    if result['response_code'] == 1:
        if result['positives'] > 0:
            num_positive = result['positives']
            vt_link = result['permalink']
            return num_positive, vt_link

        else:
            return "Clean URL"
        
    else:
        return "URL not found in VirusTotal database"


if __name__ == "__main__":

    # Test malicious url:
    url = "http://www.myetherevvalliet.com/"
    # print(check_url(url))

    # Test malicious hash:
    hash = "b1b74ff5c67cfdc16f1cb30db9b885046c4c5d71af575a2d3de4a79918c1ce89"
    # print(check_hash(hash))

    # Hash file then check VirusTotal
    # hash_of_file = file_to_hash(file_path)
    # print(check_hash(hash_of_file))
