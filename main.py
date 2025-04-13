import shutil
from pathlib import Path
from json import dump
from datetime import datetime
from parse import parse_eml, read_output, read_dirfiles
from vt import check_url, check_hash
from model import plot_precision_recall_curve, plot_confusion_matrix, plot_roc_curve, classification_report, check_email, read_pkl

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


def json_serial(obj):
    """
    Serialise object into json format
    """
    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial

def check_file(file):
    print(file)

def upload_file():
    """
    Copies eml files into UPLOADED dir
    Reads the eml files into json format
    """

    files = parse_eml(read_dirfiles(dir=Path("UPLOADED"), ext="*.eml"))
    with open(Path("UPLOADED/test.json"),"w") as fp:
        dump(files,fp, default=json_serial)

def parse_results(r_list: list, log_reg_model, tfidf_vectorizer):
    """
    Parses results into list
    """

    subject = []
    email_prediction = []
    url_check = []
    file_check = []

    for i in r_list:
        subject.append(i["subject"])
        email_prediction.append(check_email(i["sender"], i["cleaned_subject"], i["body"], log_reg_model, tfidf_vectorizer))
        url_check.append([check_url(j) for j in i["url"]])
        file_check.append([check_hash(k) for k in i["file_hash"]])

    return subject, email_prediction, url_check, file_check

def main():
    
    # upload_file()
    
    uploaded_results = read_output(read_dirfiles(dir=Path("UPLOADED"), ext="*.json"))
    
    log_reg_model, tfidf_vectorizer = read_pkl(read_dirfiles(dir=Path("PKL"), ext="*.pkl"))
    
    subject, email_prediction, url_check, file_check = parse_results(uploaded_results, log_reg_model, tfidf_vectorizer)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting programme ...")
        exit()
    except Exception as e:
        print(e)
