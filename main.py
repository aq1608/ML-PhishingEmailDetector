import shutil
from pathlib import Path
from DATAMANIPULATION.parse import parse_eml, read_output
from json import dumps,loads, dump
from datetime import datetime

def extract_data():
    """
    Recreates CLEANDATA folder and child items
    Returns a list of all child items in DATASET
    """
    dataset_dir = Path("DATASET")
    cleandata_dir = Path("CLEANDATA")
    upload_dir = Path("UPLOADED")
    
    ext = "*.csv"
    datasets: list[Path] = []
    datasets.extend(dataset_dir.glob(ext))

    sep_data = read_dirfiles(cleandata_dir, ext)

    if cleandata_dir.exists(): shutil.rmtree(cleandata_dir) #in case of corruption
    cleandata_dir.mkdir()
    for path in sep_data:
        path.touch()
    
    # if upload_dir.exists(): shutil.rmtree(upload_dir) #in case of corruption
    # upload_dir.mkdir()
    # for path in sep_data:
    #     path.touch()

    return datasets

def read_dirfiles(dir: Path, ext: str):
    """
    Reads a directory
    Returns all items in directory with specific extension
    """
    uploads: list[Path] = []
    uploads.extend(dir.glob(ext))

    return uploads

def json_serial(obj):
    """
    Serialise object into json format
    """
    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial

def upload_file():
    """
    Copies eml files into UPLOADED dir
    Reads the eml files into json format
    """

    files = parse_eml(read_dirfiles(dir=Path("UPLOADED"), ext="*.eml"))
    with open(Path("UPLOADED/test.json"),"w") as fp:
        dump(files,fp, default=json_serial)

def main():
    
    upload_file()
    
    json_file = read_output(read_dirfiles(dir=Path("UPLOADED"), ext="*.json"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting programme ...")
        exit()
    except Exception as e:
        print(e)

# import tkinter as tk
# from tkinter import filedialog
# import re
# import webbrowser
# from VirusTotal import *
# from model_test import *
# from model_train import *

# #=========================== Functions ===========================

# def show_home():
#     phishingResult_frame.pack_forget()
#     home_frame.pack(fill='both', expand=True)

# def show_model():
#     classification_report()

# def show_phishingResult():
#     home_frame.pack_forget()
#     phishingResult_frame.pack(fill='both', expand=True)

# def extract_url(text):
#     url_pattern = r"https?://[^\s]+"    # Regular expression to match URLs
#     urls = re.findall(url_pattern, text)
#     return urls if urls else None

# def check_phishing():
#     # Get inputs
#     hash = hash_entry.get().strip()
#     sender = sender_entry.get().strip()
#     subject = subject_entry.get().strip()
#     content = content_entry.get().strip()
#     # Run inputs to functions to check for possible phishing/malicious activity
#     url = extract_url(content)
#     emailPositive = check_email(sender, subject, content)
#     urlPositive = check_url(url)
#     hashPositive = check_hash(hash)

#     urlPositive_button.grid_forget()
#     hashPositive_button.grid_forget()

#     emailPositive_label.config(text=f"Based on our dataset, your email is likely to be a:     {emailPositive}")
    
#     # Different display for the different results from URL
#     if isinstance(urlPositive, tuple):
#         urlPositive_label.config(text=f"Number of positive results for malicious URL:           {urlPositive[0]}")
#         urlPositive_button.grid(row=2, column=2, padx=5, pady=5, sticky='w')
#     elif urlPositive == "Clean URL":
#         urlPositive_label.config(text="Number of positive results for malicious URL:           Clean URL")
#     elif urlPositive == "URL not found in VirusTotal database":
#         urlPositive_label.config(text="Number of positive results for malicious URL:           URL not found in VirusTotal")
#     elif urlPositive == "Error: 204 from VirusTotal":
#         urlPositive_label.config(text="Number of positive results for malicious URL:           Limit of 4 searches/min reached")
#     else:
#         urlPositive_label.config(text="Number of positive results for malicious URL:           No URL found in email")

#     # Different display for the different results from hash
#     if isinstance(hashPositive, tuple):
#         hashPositive_label.config(text=f"Number of positive results for malicious Hash:         {hashPositive[0]}")
#         hashPositive_button.grid(row=3, column=2, padx=5, pady=5, sticky='w')
#     elif hashPositive == "The file is clean.":
#         hashPositive_label.config(text="Number of positive results for malicious Hash:         Clean File")
#     elif hashPositive == "Hash of file not found in VirusTotal database.":
#         hashPositive_label.config(text="Number of positive results for malicious Hash:         File Hash not found in VirusTotal")
#     elif hashPositive == "Error: 204 from VirusTotal":
#         hashPositive_label.config(text="Number of positive results for malicious Hash:         Limit of 4 searches/min reached")   
#     else:
#         hashPositive_label.config(text="Number of positive results for malicious Hash:         No Hash provided")
#     show_phishingResult()

# # Run URL against VirusTotal function and get hyperlink if found malicious
# def view_URL_positives():   
#     content = content_entry.get().strip()
#     url = extract_url(content)
#     urlPositive = (check_url(url))
#     hyperlink = urlPositive[1]
#     webbrowser.open(hyperlink)

# # Run Hash against VirusTotal function and get hyperlink if found malicious
# def view_hash_positives():
#     hash = hash_entry.get().strip()
#     hashPositive = (check_hash(hash))
#     hyperlink = hashPositive[1]
#     webbrowser.open(hyperlink)

# # Input validation to check for proper email, and inputs if empty
# def check_entries():
#     sender = sender_entry.get().strip()
#     subject = subject_entry.get().strip()
#     content = content_entry.get().strip()

#     # Email validation RegEx pattern
#     email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

#     # Check if sender is a valid email
#     if email_pattern != "" and subject !="" and content != "":
#         result_label.grid_forget()
#         checkResults_button.grid(row=7, column=2, pady=5, padx=15, sticky='w')
#         return
    
#     elif not re.match(email_pattern, sender):
#         result_label.grid(row=7, column=2, pady=5, padx=9, sticky='nw', columnspan=2)
#         result_label.config(text="Invalid sender email!", fg='red')
#         checkResults_button.grid_forget()
#         return

#     # Check if subject and content are non-empty
#     elif subject == "":
#         result_label.grid(row=7, column=2, pady=5, padx=9, sticky='nw', columnspan=2)
#         result_label.config(text="Subject cannot be empty!", fg='red')
#         checkResults_button.grid_forget()
#         return

#     elif content == "":
#         result_label.grid(row=7, column=2, pady=5, padx=9, sticky='nw', columnspan=2)
#         result_label.config(text="Content cannot be empty!", fg='red')
#         checkResults_button.grid_forget()
#         return
        
# def bind_entries():
#     sender_entry.bind('<KeyRelease>', lambda event: check_entries())
#     subject_entry.bind('<KeyRelease>', lambda event: check_entries())
#     content_entry.bind('<KeyRelease>', lambda event: check_entries())

# # Function for user to select file, get the hash and automatically add into hash input
# def upload_file():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         hash = file_to_hash(file_path)
#         hash_entry.delete(0, tk.END)  # Clear any existing text in the entry
#         hash_entry.insert(0, hash) # Add hashed into input

# # Main GUI application window
# root = tk.Tk()
# root.geometry("800x600")
# root.title("Phishing Email Detector") # Title of GUI
# root.resizable(False, False)

# # Title
# title_label = tk.Label(root, text="Phishing Email Detector", font=("system", 24, "bold"))
# title_label.pack(pady=1)

# #=========================== Home Page Content ===========================

# # Create a frame to hold the home content
# content_frame = tk.Frame(root)
# content_frame.pack(fill='both', expand=True)

# # Create the home frame
# home_frame = tk.Frame(content_frame, bg='navy', borderwidth=2, relief='solid')

# # Show the home frame initially
# home_frame.pack(fill='both', expand=True)

# # Grid weights for centering
# home_frame.grid_columnconfigure(0, weight=1)
# home_frame.grid_columnconfigure(1, weight=1)
# home_frame.grid_columnconfigure(5, weight=1)
# home_frame.grid_columnconfigure(2, weight=1)
# home_frame.grid_rowconfigure(0, weight=1)
# home_frame.grid_rowconfigure(7, weight=1)
# home_frame.grid_rowconfigure(8, weight=1)

# # Labels and entries for the inputs
# sender_label = tk.Label(home_frame, text="Enter the Sender:", bg='navy', fg='white') # Sender text and sender input
# sender_label.grid(row=1, column=1, padx=10, pady=5, sticky='e')
# sender_entry = tk.Entry(home_frame, width=45)
# sender_entry.grid(row=1, column=2, padx=10, pady=5, sticky='w')

# subject_label = tk.Label(home_frame, text="Enter the Subject:", bg='navy', fg='white') # Subject text and subject input
# subject_label.grid(row=2, column=1, padx=10, pady=5, sticky='e')
# subject_entry = tk.Entry(home_frame, width=45)
# subject_entry.grid(row=2, column=2, padx=10, pady=5, sticky='w')

# content_label = tk.Label(home_frame, text="Enter the Content:", bg='navy', fg='white') # Content text and content input
# content_label.grid(row=3, column=1, padx=10, pady=5, sticky='e')
# content_entry = tk.Entry(home_frame, width=45)
# content_entry.grid(row=3, column=2, padx=10, pady=5, sticky='w')

# hash_label = tk.Label(home_frame, text="Enter Hash (optional):", bg='navy', fg='white') # Hash text and hash input
# hash_label.grid(row=5, column=1, padx=10, pady=5, sticky='e')
# hash_entry = tk.Entry(home_frame, width=45)
# hash_entry.grid(row=5, column=2, padx=10, pady=5, sticky='w')

# upload_button = tk.Button(home_frame, text="Upload File to get Hash", command=upload_file) # Button for uploading file
# upload_button.grid(row=7, column=1, sticky='e')

# checkResults_button = tk.Button(home_frame, text="Check for Phishing", command=check_phishing, width=35) # Button to check if phishing

# # Label to display validation results
# result_label = tk.Label(home_frame, text="", bg='navy', fg='white', anchor='w', width=20)
# result_label.grid(row=7, column=2, pady=5, padx=9, sticky='nw', columnspan=2)

# # Button to view classification model
# to_model_button = tk.Button(home_frame, text="See classification Model", command=show_model)
# to_model_button.grid(row=10, column=1, columnspan=3, pady=5, sticky='s')


# #=========================== Phishing Results Page Content ===========================

# # Phishing Results page
# phishingResult_frame = tk.Frame(content_frame, bg='navy', borderwidth=2, relief='solid')

# # Text for Email result based on dataset, URL result and Hash result based on VirusTotal
# emailPositive_label = tk.Label(phishingResult_frame, bg='navy', fg='white', text="Based on our dataset, your email is likely to be: ")
# emailPositive_label.grid(row=1, column=1, padx=5, pady=5, sticky='w')

# urlPositive_label = tk.Label(phishingResult_frame, bg='navy', fg='white', text="Number of positive results for malicious URL: ")
# urlPositive_label.grid(row=2, column=1, padx=5, pady=5, sticky='w')
# urlPositive_button = tk.Button(phishingResult_frame, text="View malicious URL Flags", command=view_URL_positives) # Button to URL link if malicious
# urlPositive_button.grid(row=2, column=2, padx=5, pady=5, sticky='w')

# hashPositive_label = tk.Label(phishingResult_frame, bg='navy', fg='white', text="Number of positive results for malicious Hash: ")
# hashPositive_label.grid(row=3, column=1, padx=5, pady=5, sticky='w')
# hashPositive_button = tk.Button(phishingResult_frame, text="View malicious Hash Flags", command=view_hash_positives) # Button to Hash link if malicious
# hashPositive_button.grid(row=3, column=2, padx=5, pady=5, sticky='w')

# to_home_button = tk.Button(phishingResult_frame, text="Back to Home Page", command=show_home) # Button to return to home page
# to_home_button.grid(row=0, column=0, pady=10, padx=10, sticky="nw")

# # Configure grid weights for centering
# phishingResult_frame.grid_columnconfigure(0, weight=1)
# phishingResult_frame.grid_columnconfigure(1, weight=1)
# phishingResult_frame.grid_columnconfigure(3, weight=1)
# phishingResult_frame.grid_columnconfigure(4, weight=1)
# phishingResult_frame.grid_rowconfigure(0, weight=1)
# phishingResult_frame.grid_rowconfigure(4, weight=1)

# # Bind entry validation
# bind_entries()

# # Run the application
# root.mainloop()