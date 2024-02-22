import requests
import pandas as pd
import os
from tqdm import tqdm
import time

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def search(q, save_path=None, databases=["US-PGPUB", "USPAT", "USOCR"], pageCount=1000, keys=[], n_limit=None, start_index=0, verbose=False):
    """
    검색 쿼리의 결과로 나온 특허들의 식별자를 반환함.

    q: 검색 쿼리 (USPTO 사이트에서처럼 동일하게 사용하면 됨)
    save_path: 검색 결과 저장 경로
    databases: 검색할 데이터베이스, 복수 가능
        - "US-PGPUB": 출원신청 특허
        - "USPAT": 등록 특허
        - "USOCR": OCR 시스템으로 읽은 옛날 특허 포함
    pageCount: 한번에 return 받을 검색결과 수
    keys: 추가로 받을 필드들
    """
    url = "https://ppubs.uspto.gov/dirsearch-public/searches/searchWithBeFamily"

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        "x-requested-with": "XMLHttpRequest",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    }

    start = start_index
    data = {
        "start": start_index,
        "pageCount": pageCount,
        "sort": "date_publ desc",
        "docFamilyFiltering": "noFiltering", # "noFiltering", "familyIdFiltering", "appNumFiltering"
        "searchType": 1,
        "familyIdEnglishOnly": True,
        "familyIdFirstPreferred": "US-PGPUB",
        "familyIdSecondPreferred": "USPAT",
        "familyIdThirdPreferred": "FPRS",
        "showDocPerFamilyPref": "showEnglish",
        "queryId": 0,
        "tagDocSearch": False,
        "query": {
            "caseId": 1,
            "hl_snippets": "2",
            "op": "OR",
            "q": q,
            "queryName": q,
            "highlights": "1",
            "qt": "brs",
            "spellCheck": False,
            "viewName": "tile",
            "plurals": True,
            "britishEquivalents": True,
            "databaseFilters": [
                {
                    "databaseName": database,
                    "countryCodes": [

                    ]
                } for database in databases
            ],
            "searchType": 1,
            "ignorePersist": True,
            "userEnteredQuery": q
        }
    }

    keys = ['guid', 'publicationReferenceDocumentNumber', "type"] + keys
    rows = []
    iter_index = 0
    while True:
        # 504 Gateway Time-out issue 있음 -> 단순 시간초과 오류인듯
        # 403 Forbidden error -> 쿼리가 이상한듯?
        response = requests.post(url, headers=headers, json=data)
        res_status = response.status_code

        try:
            response = response.json()
        except:
            # print(f"ERROR: {res_status}")
            return {"status": "error", "status_code": res_status, "results": pd.DataFrame(rows), "start_index": data["start"], "full_response": response}
            # raise Exception(f"{res_status}")

        # if data["start"] == 0:
        if iter_index == 0:
            # totalPages = response["totalPages"]
            numFound = response["numFound"]
            if n_limit is not None:
                numFound = n_limit
            if verbose:
                pbar = tqdm(total=numFound)
                if start_index != 0: pbar.update(data["start"])

        patents = response["patents"]

        for patent in patents:
            row = {key: patent[key] for key in keys}
            rows.append(row)

        if save_path is not None: pd.DataFrame(rows).to_csv(save_path, index=False)

        if verbose: pbar.update(min(pageCount, numFound-data["start"]))
        iter_index += 1
        if data["start"] + pageCount < numFound: data["start"] += pageCount
        else: break

    return {"status": "normal", "results": pd.DataFrame(rows)}

def get_patent_fulltext(guid, database):
    """
    guid에 해당하는 특허의 fulltext 정보를 json 형태로 반환함.

    quid: search에서 얻은 특허 고유 번호
    databases: 검색할 데이터베이스, 복수 가능
        - "US-PGPUB": 출원신청 특허
        - "USPAT": 등록 특허
        - "USOCR": OCR 시스템으로 읽은 옛날 특허 포함
    """

    # database = {USPAT, US-PGPUB, USOCR}
    url = f"https://ppubs.uspto.gov/dirsearch-public/patents/{guid}/highlight?queryId=27581344&source={database}&includeSections=true&uniqueId="
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        "x-requested-with": "XMLHttpRequest",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    }
    response = requests.get(url, headers=headers)

    result = response.json()
    return result

def get_filepath(dir, pn, make_directory=False):
    pn = str(pn)
    pn_split = [pn[:len(pn) % 3]] + [pn[i:i+3] for i in range(len(pn) % 3, len(pn), 3)]

    dir = os.path.join(dir, "/".join(pn_split[:-1]))
    if make_directory: makedirs(dir)
    filepath = os.path.join(dir,  f"{pn}.json")

    return filepath
