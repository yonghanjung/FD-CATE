# === build_fdcate_nhanes.py  ================================================
# Purpose: Prepare NHANES 1999–2004 for FD‑CATE (X=chronic pain, Z=opioids, Y=mortality)
# Author: ChatGPT (all code here is new / updated)
# Requirements: pandas, numpy, requests, urllib3, pyreadstat OR pandas.read_sas
# Tested with: pandas>=2.0, requests>=2.31, urllib3>=1.26 (works with v2 too)

import os, io, sys, time
from typing import Dict, List
import numpy as np
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ------------------------------- CONFIG --------------------------------------

OUTDIR = "./nhanes_fd_data"
os.makedirs(OUTDIR, exist_ok=True)

# Stable base endpoints
BASE_DATAFILES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"   # stable, year-specific “DataFiles”
BASE_PRETTY    = "https://wwwn.cdc.gov/Nchs/Nhanes"               # pretty pages (fallback only)

# Cycles and metadata
#   key: visible label; begin_year: folder under /Public/{BeginYear}/DataFiles
#   tag: filename suffix; sddsrvyr: cycle id used by NHANES
CYCLES = {
    "1999-2000": {"begin_year": 1999, "tag": "",   "sddsrvyr": 1,
                  "files": {"DEMO": "DEMO",    "MPQ": "MPQ",     "RXQ_RX": "RXQ_RX"}},
    "2001-2002": {"begin_year": 2001, "tag": "_B", "sddsrvyr": 2,
                  "files": {"DEMO": "DEMO_B", "MPQ": "MPQ_B",   "RXQ_RX": "RXQ_RX_B"}},
    "2003-2004": {"begin_year": 2003, "tag": "_C", "sddsrvyr": 3,
                  "files": {"DEMO": "DEMO_C", "MPQ": "MPQ_C",   "RXQ_RX": "RXQ_RX_C"}},
}

# All-years file: RXQ_DRUG (DataFiles live under begin_year=1988 for this release)
RXQ_DRUG_BEGIN_YEAR = 1988
RXQ_DRUG_FILE = "RXQ_DRUG"

# Linked Mortality File (public-use 2015 follow-up, CSV)
LMF_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/PUF/2015_Linked_Mortality_File_2019_public_use.csv"

# Multum ingredient-level L3 category codes for opioids
OPIOID_CODES = {60, 191}

# ------------------------------- HTTP CLIENT ---------------------------------

def make_session() -> requests.Session:
    sess = requests.Session()
    # Retry on common transient statuses
    retry = Retry(
        total=7,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Connection": "keep-alive",
    })
    return sess

SESSION = make_session()

def http_download(url: str, save_path: str):
    r = SESSION.get(url, timeout=90)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code} for {url}", response=r)
    with open(save_path, "wb") as f:
        f.write(r.content)

# ------------------------------- FETCHERS ------------------------------------

def candidate_urls_for(stem: str, tag: str, cycle_label: str, begin_year: int) -> List[str]:
    """
    Return an ordered list of candidate URLs to try for a given XPT file.
    """
    # 1) Stable DataFiles path (preferred)
    urls = [f"{BASE_DATAFILES}/{begin_year}/DataFiles/{stem}.xpt"]
    # 2) Pretty paths (two case variants for extension)
    urls += [
        f"{BASE_PRETTY}/{cycle_label}/{stem}.XPT",
        f"{BASE_PRETTY}/{cycle_label}/{stem}.xpt",
    ]
    return urls

def fetch_xpt(save_path: str, stem: str, tag: str, cycle_label: str, begin_year: int):
    """
    Try multiple URL patterns with retry/backoff; raise with context if all fail.
    """
    if os.path.exists(save_path):
        return
    errors = []
    for url in candidate_urls_for(stem, tag, cycle_label, begin_year):
        try:
            http_download(url, save_path)
            return
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    msg = " | ".join(errors[-3:])
    raise RuntimeError(f"Failed to fetch {stem} for {cycle_label}. Tried: "
                       f"{' , '.join(candidate_urls_for(stem, tag, cycle_label, begin_year))}. "
                       f"Errors: {msg}")

def read_xpt(path: str) -> pd.DataFrame:
    # pandas can read XPT via read_sas(format="xport")
    df = pd.read_sas(path, format="xport")
    # Normalize colnames to uppercase strings
    df.columns = [str(c).upper() for c in df.columns]
    return df

# ------------------------------- PIPELINE ------------------------------------

def load_and_append() -> Dict[str, pd.DataFrame]:
    demo_list, mpq_list, rx_list = [], [], []
    for cyc_label, meta in CYCLES.items():
        beg = meta["begin_year"]
        tag = meta["tag"]
        sdd = meta["sddsrvyr"]
        files = meta["files"]

        # Fetch
        for comp, stem in files.items():
            save_path = os.path.join(OUTDIR, f"{stem}.xpt")
            fetch_xpt(save_path, stem=stem, tag=tag, cycle_label=cyc_label, begin_year=beg)

        # Read
        df_demo = read_xpt(os.path.join(OUTDIR, f"{files['DEMO']}.xpt"))
        df_mpq  = read_xpt(os.path.join(OUTDIR, f"{files['MPQ']}.xpt"))
        df_rx   = read_xpt(os.path.join(OUTDIR, f"{files['RXQ_RX']}.xpt"))

        # Tag cycle
        for d in (df_demo, df_mpq, df_rx):
            d["SDDSRVYR_SRC"] = sdd

        demo_list.append(df_demo); mpq_list.append(df_mpq); rx_list.append(df_rx)

    DEMO_all = pd.concat(demo_list, axis=0, ignore_index=True, sort=False)
    MPQ_all  = pd.concat(mpq_list,  axis=0, ignore_index=True, sort=False)
    RX_all   = pd.concat(rx_list,   axis=0, ignore_index=True, sort=False)
    return {"DEMO_all": DEMO_all, "MPQ_all": MPQ_all, "RX_all": RX_all}

def build_weights(DEMO_all: pd.DataFrame) -> pd.Series:
    # MEC 6-year weight for 1999–2004 per CDC tutorial:
    # if SDDSRVYR in {1,2}: (2/3)*WTMEC4YR, if SDDSRVYR==3: (1/3)*WTMEC2YR
    if "SDDSRVYR" not in DEMO_all.columns and "SDDSRVYR_SRC" in DEMO_all.columns:
        DEMO_all["SDDSRVYR"] = DEMO_all["SDDSRVYR_SRC"]
    sdd = pd.to_numeric(DEMO_all["SDDSRVYR"], errors="coerce")
    w_4yr = pd.to_numeric(DEMO_all.get("WTMEC4YR"), errors="coerce")
    w_2yr = pd.to_numeric(DEMO_all.get("WTMEC2YR"), errors="coerce")
    mec6 = np.where(sdd.isin([1,2]), (2.0/3.0)*w_4yr, (1.0/3.0)*w_2yr)
    return pd.Series(pd.to_numeric(mec6, errors="coerce"), index=DEMO_all.index, name="WTMEC6YR")

def construct_X_from_MPQ(mpq: pd.DataFrame) -> pd.Series:
    # X=1 if MPQ100==1 and MPQ110 in {3,4}; else 0 (NaN if both missing)
    m100 = pd.to_numeric(mpq.get("MPQ100"), errors="coerce")
    m110 = pd.to_numeric(mpq.get("MPQ110"), errors="coerce")
    X = ((m100 == 1) & (m110.isin([3,4]))).astype(float)
    return X.rename("X")

def fetch_rxq_drug() -> str:
    # Prefer the DataFiles location (begin_year=1988 for RXQ_DRUG)
    local = os.path.join(OUTDIR, f"{RXQ_DRUG_FILE}.xpt")
    if os.path.exists(local):
        return local
    # Try DataFiles first, then pretty
    urls = [
        f"{BASE_DATAFILES}/{RXQ_DRUG_BEGIN_YEAR}/DataFiles/{RXQ_DRUG_FILE}.xpt",
        f"{BASE_PRETTY}/1988-2020/{RXQ_DRUG_FILE}.XPT",
        f"{BASE_PRETTY}/1988-2020/{RXQ_DRUG_FILE}.xpt",
    ]
    errors = []
    for url in urls:
        try:
            http_download(url, local)
            return local
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise RuntimeError(f"Failed to fetch {RXQ_DRUG_FILE}. Tried: {', '.join(urls)}. "
                       f"Errors: {' | '.join(errors[-3:])}")

def construct_Z_from_RX(rx_all: pd.DataFrame, rxq_drug: pd.DataFrame) -> pd.Series:
    # Merge RXQ_RX with RXQ_DRUG (on RXDDRGID) and flag presence of any ingredient L3 class in {60,191}.
    rx_all = rx_all.copy()
    rxq_drug = rxq_drug.copy()
    for df in (rx_all, rxq_drug):
        df.columns = [str(c).upper() for c in df.columns]

    key = "RXDDRGID"
    rx_all[key] = rx_all[key].astype(str).str.strip()
    rxq_drug[key] = rxq_drug[key].astype(str).str.strip()

    # Keep only the ingredient L3 id columns (RXDICI?C)
    ing_l3_cols = [c for c in rxq_drug.columns if c.startswith("RXDICI") and c.endswith("C")]
    rxq_sub = rxq_drug[[key] + ing_l3_cols].copy()

    merged = rx_all[["SEQN", key]].merge(rxq_sub, on=key, how="left", validate="m:1")

    # Flag if any ingredient-level L3 category equals opioid codes
    opioid_mask = np.zeros(len(merged), dtype=bool)
    for c in ing_l3_cols:
        opioid_mask |= pd.to_numeric(merged[c], errors="coerce").isin(list(OPIOID_CODES))

    opioids_by_person = merged.loc[opioid_mask].groupby("SEQN").size()
    Z = merged[["SEQN"]].drop_duplicates().set_index("SEQN")
    Z["Z"] = (Z.index.isin(opioids_by_person.index)).astype(float)
    return Z["Z"]

def prepare_dataset():
    # 1) Load/append cycles
    mats = load_and_append()
    DEMO_all, MPQ_all, RX_all = mats["DEMO_all"], mats["MPQ_all"], mats["RX_all"]

    # 2) Adult restriction (MPQ target 20+)
    DEMO_all["RIDAGEYR"] = pd.to_numeric(DEMO_all["RIDAGEYR"], errors="coerce")
    adults = DEMO_all["RIDAGEYR"] >= 20

    # 3) Build combined MEC 6-year weights
    DEMO_all["WTMEC6YR"] = build_weights(DEMO_all)

    # 4) Treatment X
    X_series = construct_X_from_MPQ(MPQ_all)
    X = MPQ_all[["SEQN"]].assign(X=X_series.values)

    # 5) Mediator Z (needs RXQ_DRUG)
    p_rxq = fetch_rxq_drug()
    RXQ_DRUG_df = read_xpt(p_rxq)
    Z_series = construct_Z_from_RX(RX_all, RXQ_DRUG_df)
    Z = pd.DataFrame({"SEQN": Z_series.index, "Z": Z_series.values})

    # 6) Base covariates
    base = DEMO_all.loc[adults, ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","DMDEDUC2","INDFMPIR",
                                 "SDMVSTRA","SDMVPSU","WTMEC6YR"]].copy()

    # 7) Merge X, Z
    df = base.merge(X, on="SEQN", how="left").merge(Z, on="SEQN", how="left")

    # 8) Mortality (LMF)
    lmf_path = os.path.join(OUTDIR, "lmf_2015_public_use.csv")
    if not os.path.exists(lmf_path):
        http_download(LMF_URL, lmf_path)
    LMF = pd.read_csv(lmf_path, dtype={"SEQN": np.int64})
    LMF = LMF[["SEQN","ELIGSTAT","MORTSTAT","PERMTH_EXM"]].copy()
    df = df.merge(LMF, on="SEQN", how="left")

    # 9) Construct modeling matrices
    df["SEX_FEMALE"] = (df["RIAGENDR"]==2).astype(float)
    df["AGE"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df["PIR"] = pd.to_numeric(df["INDFMPIR"], errors="coerce")
    df["EDU"] = pd.to_numeric(df["DMDEDUC2"], errors="coerce")
    for k in [1,2,3,4,5]:
        df[f"RACE_{k}"] = (df["RIDRETH1"]==k).astype(float)

    # Respect LMF eligibility for vital status / time
    df.loc[df["ELIGSTAT"]!=1, ["MORTSTAT","PERMTH_EXM"]] = np.nan

    C_cols = ["AGE","SEX_FEMALE","PIR","EDU","RACE_1","RACE_2","RACE_3","RACE_4","RACE_5"]
    C = df[C_cols].astype(float).values
    Xv = df["X"].astype(float).values
    Zv = df["Z"].astype(float).values
    Y_bin = df["MORTSTAT"].astype(float).values
    Y_time = df["PERMTH_EXM"].astype(float).values

    meta = df[["SEQN","WTMEC6YR","SDMVSTRA","SDMVPSU"]].copy()

    # Save artifacts
    np.savez(os.path.join(OUTDIR, "fdcate_arrays_1999_2004.npz"),
             C=C, X=Xv, Z=Zv, Y_bin=Y_bin, Y_time=Y_time)
    df.to_parquet(os.path.join(OUTDIR, "fdcate_long_1999_2004.parquet"), index=False)
    meta.to_csv(os.path.join(OUTDIR, "fdcate_design_1999_2004.csv"), index=False)

    print("Saved:",
          "\n  arrays:  ", os.path.join(OUTDIR, "fdcate_arrays_1999_2004.npz"),
          "\n  long df: ", os.path.join(OUTDIR, "fdcate_long_1999_2004.parquet"),
          "\n  design:  ", os.path.join(OUTDIR, "fdcate_design_1999_2004.csv"))

if __name__ == "__main__":
    prepare_dataset()
# ============================================================================#
