"""
RxGuard AI - AI Drug-Drug Interaction Checker
HayMedics Academy | Data | Research | Innovation

Run:  pip install streamlit reportlab    (one-time)   then   streamlit run app.py
Expects: meddose_model_v2.joblib, interaction_type_labels.csv,
         assets/logo.png  (a white-background horizontal HayMedics logo)
"""
import os, io, base64, itertools
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import requests
import urllib.parse
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, Crippen, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors as rlc
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
RDLogger.DisableLog("rdApp.*")

APP_NAME = "RxGuard AI"
TAGLINE  = "Drug Interaction Intelligence"
PILL     = "DRUG-INTERACTION INTELLIGENCE"

NAVY, BLUE, AMBER = "#1E2563", "#2E57A6", "#FEA621"
MODEL_FILE  = "meddose_model_v2.joblib"
# Uses the local model file if present; otherwise downloads from this GitHub
# Release asset (this is what happens on Streamlit Cloud).
MODEL_URL   = "https://github.com/HayMedics/rxguard-ai/releases/download/v1.0/meddose_model_v2.joblib"
LABELS_FILE = "interaction_type_labels.csv"
LOGO_CANDIDATES = ["assets/logo.png", "assets/HMA__Tagline.jpg", "assets/HMA.jpg",
                   "assets/HMA__Tagline_PNG.png", "assets/HMA_PNG.png"]
ICON_CANDIDATES = ["assets/HMA_ICON_PNG.png", "assets/HMA_ICON.jpg", "assets/icon.png"]

def _find(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

LOGO, ICON = _find(LOGO_CANDIDATES), _find(ICON_CANDIDATES)

_page_icon = "💊"
try:
    if ICON:
        from PIL import Image
        _page_icon = Image.open(ICON)
except Exception:
    pass

st.set_page_config(page_title=f"{APP_NAME} · HayMedics Academy",
                   page_icon=_page_icon, layout="centered")

st.markdown("""
<style>
  .block-container { padding-top: 1.4rem; max-width: 900px; }
  h1,h2,h3,h4 { color:#1E2563; }
  .rx-header { display:flex; justify-content:space-between; align-items:center;
    background:#fff; border:1px solid #E6EAF3; border-bottom:3px solid #FEA621;
    border-radius:16px; padding:.9rem 1.4rem; box-shadow:0 3px 16px rgba(30,37,99,.07);
    margin-bottom:1.1rem; }
  .rx-logo { height:46px; width:auto; }
  .rx-wordmark { font-weight:800; font-size:1.3rem; color:#1E2563; }
  .rx-pill { border:1.5px solid #1E2563; color:#1E2563; border-radius:999px;
    padding:.35rem .9rem; font-weight:700; font-size:.72rem; letter-spacing:.6px; white-space:nowrap; }
  .rx-eyebrow { color:#FEA621; font-weight:800; letter-spacing:1.5px; font-size:.8rem; margin-top:.3rem; }
  .rx-title { font-size:2.3rem; font-weight:800; color:#1E2563; line-height:1.1; margin:.1rem 0 .9rem 0; }
  .rx-disc { background:#FFF7E8; border:1px solid #F7D58B; color:#7a5a12;
    border-radius:10px; padding:.7rem 1rem; font-size:.88rem; margin:.2rem 0 1.1rem 0; }
  .rx-redflag { background:#FCE9E7; border:1px solid #E6B0AA; border-left:5px solid #C0392B;
    color:#7B241C; border-radius:10px; padding:.8rem 1rem; margin:.4rem 0 .9rem 0; font-size:.92rem; }
  .rx-top { background:linear-gradient(135deg,#1E2563 0%,#2E57A6 100%); color:#fff;
    border-radius:16px; padding:1.2rem 1.4rem; margin:.5rem 0; box-shadow:0 4px 18px rgba(30,37,99,.18); }
  .rx-top h4 { color:#fff; margin:0 0 .5rem 0; font-size:1.3rem; }
  .rx-watch { background:rgba(255,255,255,.13); border-radius:8px; padding:.5rem .75rem;
    font-size:.92rem; margin-top:.7rem; }
  .rx-pair { background:#fff; border:1px solid #E6EAF3; border-left:5px solid #2E57A6;
    border-radius:12px; padding:.85rem 1.1rem; margin:.5rem 0; }
  .rx-pair b { color:#1E2563; }
  .rx-badge { display:inline-block; background:#FEA621; color:#1E2563; font-weight:800;
    padding:.22rem .7rem; border-radius:999px; font-size:.9rem; }
  .rx-badge-sm { display:inline-block; background:#EEF2FB; color:#2E57A6; font-weight:700;
    padding:.12rem .55rem; border-radius:999px; font-size:.8rem; }
  .bar-wrap { background:#E6EAF3; border-radius:8px; height:10px; width:100%; overflow:hidden; margin:.2rem 0; }
  .bar-fill { background:#2E57A6; height:10px; }
  .imp-fill { background:#FEA621; height:10px; }
  .muted { color:#5b6480; font-size:.85rem; }
  .stButton>button, .stDownloadButton>button { background:#2E57A6; color:#fff; border:0; border-radius:10px; font-weight:700; }
  .stButton>button:hover, .stDownloadButton>button:hover { background:#FEA621; color:#1E2563; }
</style>
""", unsafe_allow_html=True)

DRUGS = {
    'Allopurinol': 'O=c1[nH]cnc2[nH]ncc12',
    'Alprazolam': 'Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2',
    'Amitriptyline': 'CN(C)CCC=C1c2ccccc2CCc2ccccc21',
    'Amlodipine': 'CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl',
    'Amoxicillin': 'CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O',
    'Ampicillin': 'CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O',
    'Apixaban': 'COc1ccc(-n2nc(C(N)=O)c3c2CCN(c2ccc(N4CCCCC4=O)cc2)C3=O)cc1',
    'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'Atenolol': 'CC(C)NCC(O)COc1ccc(CC(N)=O)cc1',
    'Atorvastatin': 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O',
    'Bisoprolol': 'CC(C)NCC(O)COc1ccc(COCCOC(C)C)cc1',
    'Caffeine': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
    'Carbamazepine': 'NC(=O)N1c2ccccc2C=Cc2ccccc21',
    'Carvedilol': 'COc1ccccc1OCCNCC(O)COc1cccc2[nH]c3ccccc3c12',
    'Celecoxib': 'Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1',
    'Cephalexin': 'CC1=C(C(=O)O)N2C(=O)C(NC(=O)C(N)c3ccccc3)C2SC1',
    'Cetirizine': 'OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1',
    'Ciprofloxacin': 'O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O',
    'Citalopram': 'CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21',
    'Clonazepam': 'O=C1CN=C(c2ccccc2Cl)c2cc([N+](=O)[O-])ccc2N1',
    'Clopidogrel': 'COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1',
    'Dexamethasone': 'CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO',
    'Diazepam': 'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21',
    'Diclofenac': 'OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl',
    'Diflunisal': 'O=C(O)c1cc(-c2ccc(F)cc2F)ccc1O',
    'Diltiazem': 'CC(=O)OC1C(c2ccc(OC)cc2)Sc2ccccc2N(CCN(C)C)C1=O',
    'Diphenhydramine': 'CN(C)CCOC(c1ccccc1)c1ccccc1',
    'Enalapril': 'CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O',
    'Fexofenadine': 'CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1',
    'Fluconazole': 'OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F',
    'Fluoxetine': 'CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1',
    'Furosemide': 'NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl',
    'Gabapentin': 'NCC1(CC(=O)O)CCCCC1',
    'Glibenclamide': 'COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1',
    'Gliclazide': 'Cc1ccc(S(=O)(=O)NC(=O)NN2CC3CCCC3C2)cc1',
    'Glimepiride': 'CCC1=C(C)CN(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCC(C)CC3)cc2)C1=O',
    'Haloperidol': 'O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1',
    'Hydrochlorothiazide': 'NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O',
    'Hydrocortisone': 'CC12CCC3C(CCC4=CC(=O)CCC34C)C1CC(O)C2(O)C(=O)CO',
    'Ibuprofen': 'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
    'Indomethacin': 'COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1',
    'Ketoprofen': 'CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1',
    'Lamotrigine': 'Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1',
    'Lansoprazole': 'Cc1ccnc(CS(=O)c2nc3ccccc3[nH]2)c1OCC(F)(F)F',
    'Levetiracetam': 'CCC(C(N)=O)N1CCCC1=O',
    'Levofloxacin': 'CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23',
    'Linezolid': 'CC(=O)NCC1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1',
    'Lisinopril': 'NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O',
    'Loratadine': 'CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1',
    'Lorazepam': 'OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O',
    'Losartan': 'CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1',
    'Mefenamic acid': 'Cc1cccc(C)c1Nc1ccccc1C(=O)O',
    'Metformin': 'CN(C)C(=N)NC(=N)N',
    'Methotrexate': 'CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1',
    'Metoclopramide': 'CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC',
    'Metoprolol': 'CC(C)NCC(O)COc1ccc(CCOC)cc1',
    'Metronidazole': 'Cc1ncc([N+](=O)[O-])n1CCO',
    'Naproxen': 'COc1ccc2cc(C(C)C(=O)O)ccc2c1',
    'Nifedipine': 'COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]',
    'Nitrofurantoin': 'O=C1CN(/N=C/c2ccc([N+](=O)[O-])o2)C(=O)N1',
    'Olanzapine': 'Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1',
    'Omeprazole': 'COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1',
    'Ondansetron': 'Cc1nccn1CC1CCc2c(c3ccccc3n2C)C1=O',
    'Pantoprazole': 'COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC(F)F)c3C)nc2c1',
    'Paracetamol': 'CC(=O)Nc1ccc(O)cc1',
    'Paroxetine': 'Fc1ccc(C2CCNCC2COc2ccc3c(c2)OCO3)cc1',
    'Phenytoin': 'O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1',
    'Pioglitazone': 'CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1',
    'Piroxicam': 'CN1C(C(=O)Nc2ccccn2)=C(O)c2ccccc2S1(=O)=O',
    'Prednisolone': 'CC12CC(O)C3C(CCC4=CC(=O)C=CC43C)C1CCC2(O)C(=O)CO',
    'Pregabalin': 'CC(C)CC(CN)CC(=O)O',
    'Propranolol': 'CC(C)NCC(O)COc1cccc2ccccc12',
    'Quetiapine': 'OCCOCCN1CCN(C2=Nc3ccccc3Sc3ccccc32)CC1',
    'Ranitidine': 'CN/C(=C\\[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1',
    'Risperidone': 'CC1=C(CCN2CCC(c3noc4cc(F)ccc34)CC2)C(=O)N2CCCCC2=N1',
    'Rivaroxaban': 'O=C(NCC1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1',
    'Rosuvastatin': 'CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1C=CC(O)CC(O)CC(=O)O',
    'Salbutamol': 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',
    'Sertraline': 'CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21',
    'Sildenafil': 'CCCc1nn(C)c2c1nc(-c1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC)[nH]c2=O',
    'Simvastatin': 'CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12',
    'Sitagliptin': 'NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F',
    'Spironolactone': 'CC(=O)SC1CC2C3CCC4=CC(=O)CCC4(C)C3CCC2(C)C12CCC(=O)O2',
    'Sulfamethoxazole': 'Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1',
    'Theophylline': 'Cn1c(=O)c2[nH]cnc2n(C)c1=O',
    'Tramadol': 'CN(C)CC1CCCCC1(O)c1cccc(OC)c1',
    'Trimethoprim': 'COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC',
    'Valproic acid': 'CCCC(CCC)C(=O)O',
    'Valsartan': 'CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)C(C(C)C)C(=O)O',
    'Venlafaxine': 'COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1',
    'Verapamil': 'COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC',
    'Warfarin': 'CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O',
    'Zolpidem': 'Cc1ccc(-c2c(CC(=O)N(C)C)nc3ccc(C)cn23)cc1',
}


DESC_FUNCS = {
    "MW": Descriptors.MolWt, "LogP": Crippen.MolLogP, "TPSA": Descriptors.TPSA,
    "HBD": Descriptors.NumHDonors, "HBA": Descriptors.NumHAcceptors,
    "RotB": Descriptors.NumRotatableBonds, "Rings": Descriptors.RingCount,
    "HeavyAt": Descriptors.HeavyAtomCount, "FracCsp3": Descriptors.FractionCSP3,
    "QED": Descriptors.qed,
}

# ---------- mechanism-based risk layer (curated, NOT a model output) ----------
def _match(d, groups):
    return all(any(o in d for o in g) for g in groups)

RISK_RULES = [
 ([("qtc", "qt-prolong", "qt prolong")], "high", "Heart-rhythm risk (QT prolongation): palpitations, dizziness, fainting — seek urgent care if these occur."),
 ([("serotonergic", "serotonin")], "high", "Serotonin syndrome risk: agitation, fast heartbeat, high temperature, muscle stiffness, confusion — seek urgent care."),
 ([("anticoagulant", "bleeding", "hemorrhage", "haemorrhage")], "high", "Increased bleeding risk: unusual bruising, blood in urine/stool, prolonged bleeding."),
 ([("respiratory depress",)], "high", "Slowed-breathing risk: seek urgent care for very slow/shallow breathing or unresponsiveness."),
 ([("cns depress", "central nervous system depress")], "high", "Additive CNS depression: excessive drowsiness, confusion, slowed breathing."),
 ([("hyperkalemi", "hyperkalaemi")], "high", "High-potassium risk: muscle weakness, irregular heartbeat."),
 ([("hypoglycem", "hypoglycaem")], "high", "Low-blood-sugar risk: shakiness, sweating, confusion, palpitations."),
 ([("nephrotoxic",)], "high", "Kidney-toxicity risk: monitor kidney function and stay hydrated."),
 ([("hepatotoxic",)], "high", "Liver-toxicity risk: jaundice, dark urine, abdominal pain, nausea."),
 ([("myelosuppress", "bone marrow", "neutropeni")], "high", "Low blood-cell-count risk: infections, fatigue, unusual bleeding."),
 ([("bradycard", "cardiotoxic", "arrhythm")], "high", "Heart rate/rhythm effects: very slow or irregular pulse, fainting."),
 ([("hypertensive", "hypertension")], "high", "Raised blood-pressure risk: severe headache — monitor blood pressure."),
 ([("neuromuscular block",)], "high", "Prolonged muscle weakness / breathing effects."),
 ([("hypotensive", "antihypertensive")], "moderate", "Additive blood-pressure lowering: dizziness or fainting, especially on standing."),
 ([("sedat", "drowsi", "somnolen")], "moderate", "Additive drowsiness: avoid driving/machinery until you know the effect."),
 ([("anticholinergic",)], "moderate", "Additive anticholinergic effects: dry mouth, constipation, blurred vision, confusion (esp. older adults)."),
 ([("serum concentration",), ("increas",)], "moderate", "Drug levels may rise, increasing side-effect risk — monitoring or dose review may be needed."),
 ([("serum concentration",), ("decreas",)], "low", "Drug levels may fall, possibly reducing effectiveness."),
 ([("metabolism",), ("decreas",)], "moderate", "Slower clearance can raise drug levels — monitoring may be needed."),
 ([("metabolism",), ("increas",)], "low", "Faster clearance may reduce effectiveness of the affected drug."),
 ([("therapeutic efficacy", "effectiveness")], "low", "Reduced effectiveness of the affected drug is possible."),
 ([("absorption",), ("decreas",)], "low", "Reduced absorption — separating dosing times may help."),
]

def assess_risk(desc):
    d = desc.lower()
    for groups, level, note in RISK_RULES:
        if _match(d, groups):
            return level, note
    return "moderate", "Potential interaction — monitor and check with a pharmacist or clinician."

RISK_META = {"high": ("🔴", "High-risk mechanism", "#C0392B", "#FCE9E7"),
             "moderate": ("🟠", "Monitor", "#B9770E", "#FFF3E0"),
             "low": ("🟡", "Lower concern", "#8A6D0B", "#FFFBEA")}
RISK_RANK = {"high": 0, "moderate": 1, "low": 2}

def risk_badge(level):
    e, lbl, fg, bg = RISK_META[level]
    return (f'<span style="background:{bg};color:{fg};font-weight:800;padding:.15rem .6rem;'
            f'border-radius:999px;font-size:.76rem;white-space:nowrap;">{e} {lbl}</span>')

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        if not MODEL_URL:
            return None
        with st.spinner("Downloading model (first run only, ~80 MB)…"):
            r = requests.get(MODEL_URL, stream=True, timeout=180)
            r.raise_for_status()
            with open(MODEL_FILE, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_FILE):
        return {}
    s = pd.read_csv(LABELS_FILE, index_col=0).iloc[:, 0]
    return {int(k): str(v) for k, v in s.items()}

@st.cache_data(show_spinner="Looking up structure on PubChem…")
def name_to_smiles(name):
    enc = urllib.parse.quote(name.strip())
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    for prop in ("SMILES", "CanonicalSMILES", "IsomericSMILES", "ConnectivitySMILES"):
        try:
            r = requests.get(f"{base}{enc}/property/{prop}/TXT", timeout=15)
            if r.status_code == 200:
                smi = r.text.strip().splitlines()[0].strip()
                if smi and Chem.MolFromSmiles(smi):
                    return smi
        except Exception:
            continue
    return None

def mol_uri(smiles, size=(220, 150)):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    img = Draw.MolToImage(m, size=size)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def img_data_uri(path):
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def _gen(bundle):
    return rdFingerprintGenerator.GetMorganGenerator(radius=bundle["radius"], fpSize=bundle["n_bits"])

def featurize_pair(smi1, smi2, bundle):
    m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    if m1 is None or m2 is None:
        return None
    names = bundle["descriptors"]
    gen = _gen(bundle)
    d1 = np.array([DESC_FUNCS[n](m1) for n in names], dtype=np.float64)
    d2 = np.array([DESC_FUNCS[n](m2) for n in names], dtype=np.float64)
    a1 = np.zeros(bundle["n_bits"], np.uint8); ConvertToNumpyArray(gen.GetFingerprint(m1), a1)
    a2 = np.zeros(bundle["n_bits"], np.uint8); ConvertToNumpyArray(gen.GetFingerprint(m2), a2)
    x = np.concatenate([d1 + d2, np.abs(d1 - d2), d1 * d2, (a1 + a2)])
    return x.astype(np.float32).reshape(1, -1)

def predict_top(x, bundle, labels, k=5):
    model, le = bundle["model"], bundle["label_encoder"]
    proba = model.predict_proba(x)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1][:k]
    return [(labels.get(int(le.inverse_transform([classes[i]])[0]),
            f"Interaction type {int(le.inverse_transform([classes[i]])[0])}"), float(proba[i]))
            for i in order]

def with_names(text, a, b):
    return (text.replace("#Drug1", a).replace("#Drug2", b)
                .replace("#drug1", a).replace("#drug2", b))

def global_importances(bundle):
    d = bundle["descriptors"]
    fn = ([f"{c} (sum)" for c in d] + [f"{c} (difference)" for c in d] + [f"{c} (product)" for c in d])
    imp = bundle["model"].feature_importances_
    items = list(zip(fn, imp[:30])) + [("Molecular fingerprint (overall structure)", float(imp[30:].sum()))]
    items.sort(key=lambda t: -t[1])
    return items[:6]

def explain_pair(sa, sb, bundle):
    m1, m2 = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    gen = _gen(bundle)
    tani = DataStructs.TanimotoSimilarity(gen.GetFingerprint(m1), gen.GetFingerprint(m2))
    diffs = []
    for n in bundle["descriptors"]:
        a, b = DESC_FUNCS[n](m1), DESC_FUNCS[n](m2)
        sc = max(abs(a), abs(b), 1e-6)
        diffs.append((n, a, b, abs(a - b) / sc))
    diffs.sort(key=lambda t: -t[3])
    return tani, diffs[:4]

def build_pdf(meds, rows, logo_path):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16*mm, bottomMargin=16*mm,
                            leftMargin=16*mm, rightMargin=16*mm)
    ss = getSampleStyleSheet()
    navy = rlc.HexColor(NAVY); amber = rlc.HexColor(AMBER); blue = rlc.HexColor(BLUE)
    H = ParagraphStyle("H", parent=ss["Title"], textColor=navy, fontSize=17, spaceAfter=2)
    sub = ParagraphStyle("s", parent=ss["Normal"], textColor=blue, fontSize=10)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.5, leading=11)
    story = []
    if logo_path and os.path.exists(logo_path):
        try:
            im = RLImage(logo_path); im.drawHeight = 12*mm
            im.drawWidth = 12*mm * im.imageWidth / im.imageHeight
            story += [im, Spacer(1, 3*mm)]
        except Exception:
            pass
    story.append(Paragraph(f"{APP_NAME} — Drug Interaction Report", H))
    story.append(Paragraph(datetime.now().strftime("Generated %d %b %Y, %H:%M"), sub))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("<b>Medications screened:</b> " + ", ".join(n for n, _ in meds), body))
    story.append(Spacer(1, 4*mm))
    if any(r[4] == "high" for r in rows):
        warn = ParagraphStyle("w", parent=ss["Normal"], fontSize=9.5, textColor=rlc.HexColor("#C0392B"))
        story.append(Paragraph("<b>HIGH-RISK MECHANISM</b> flagged in one or more combinations "
                               "(see the Risk column). A flag reflects the predicted interaction type, "
                               "not a verified severity rating; its absence is not proof of safety.", warn))
        story.append(Spacer(1, 3*mm))
    rlabel = {"high": "High", "moderate": "Monitor", "low": "Lower"}
    rcol = {"high": rlc.HexColor("#C0392B"), "moderate": rlc.HexColor("#B9770E"), "low": rlc.HexColor("#8A6D0B")}
    data = [["Drug A", "Drug B", "Predicted interaction", "Risk", "Conf."]]
    cmds = [("BACKGROUND", (0,0), (-1,0), navy), ("TEXTCOLOR", (0,0), (-1,0), rlc.white),
            ("FONTSIZE", (0,0), (-1,-1), 8), ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rlc.white, rlc.HexColor("#F4F6FB")]),
            ("LINEBELOW", (0,0), (-1,0), 1.2, amber), ("GRID", (0,0), (-1,-1), 0.3, rlc.HexColor("#E6EAF3")),
            ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]
    for i, (na, nb, desc, conf, level, note) in enumerate(rows):
        data.append([Paragraph(na, body), Paragraph(nb, body), Paragraph(desc, body),
                     rlabel[level], f"{conf*100:.0f}%"])
        cmds.append(("TEXTCOLOR", (3, i+1), (3, i+1), rcol[level]))
        cmds.append(("FONTNAME", (3, i+1), (3, i+1), "Helvetica-Bold"))
    t = Table(data, colWidths=[24*mm, 24*mm, 80*mm, 18*mm, 12*mm], repeatRows=1)
    t.setStyle(TableStyle(cmds))
    story.append(t)
    highs = [(na, nb, note) for na, nb, desc, conf, level, note in rows if level == "high"]
    if highs:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("<b>What to watch for (high-risk combinations):</b>", body))
        for na, nb, note in highs:
            story.append(Paragraph(f"• {na} + {nb}: {note}", body))
    story.append(Spacer(1, 6*mm))
    disc = ParagraphStyle("d", parent=ss["Normal"], fontSize=7.5, textColor=rlc.HexColor("#7a5a12"))
    story.append(Paragraph("Educational demo — not medical advice. Predictions come from a machine-learning "
                           "model and may be wrong. Risk levels are mechanism-based, not validated severity "
                           "scores or personal risk. Always consult a qualified pharmacist or clinician. "
                           "Built by HayMedics Academy · Data | Research | Innovation.", disc))
    doc.build(story)
    return buf.getvalue()

# ---------- header ----------
logo_html = (f'<img src="{img_data_uri(LOGO)}" class="rx-logo"/>' if LOGO
             else '<span class="rx-wordmark">HayMedics Academy</span>')
st.markdown(f'<div class="rx-header">{logo_html}<span class="rx-pill">{PILL}</span></div>',
            unsafe_allow_html=True)
_t = TAGLINE.rsplit(" ", 1)
st.markdown(f'<div class="rx-eyebrow">{APP_NAME.upper()}</div>'
            f'<div class="rx-title">{_t[0]} <span style="color:{AMBER}">{_t[1]}</span></div>',
            unsafe_allow_html=True)
st.markdown('<div class="rx-disc">⚠️ <b>Educational demo — not medical advice.</b> '
            'Predictions come from a machine-learning model and may be wrong. '
            'Always consult a qualified pharmacist or clinician.</div>', unsafe_allow_html=True)

bundle, labels = load_model(), load_labels()
if bundle is None:
    st.error(f"Couldn't find **{MODEL_FILE}**. Run `python train_v2_fast.py`, "
             "then launch the app from the same folder.")
    st.stop()

st.session_state.setdefault("extras", [])
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)

def render_check(entry):
    meds, plist = entry["meds"], entry["pairs"]
    n_high = sum(1 for p in plist if p["risk"] == "high")
    if n_high:
        st.markdown(f'<div class="rx-redflag">🚩 <b>{n_high} combination(s) involve a high-risk '
                    'interaction mechanism.</b> Review carefully and consult a pharmacist or clinician. '
                    'A flag is based on the predicted interaction type — its absence does <u>not</u> '
                    'mean a combination is safe.</div>', unsafe_allow_html=True)
    st.markdown(f"##### {len(meds)} medications → {len(plist)} pair(s) analysed")
    st.caption("Sorted by risk, then model confidence. Risk reflects the predicted interaction "
               "*mechanism* — not a validated severity score or your personal risk.")
    for idx, p in enumerate(plist):
        border = RISK_META[p["risk"]][2]
        if idx == 0:
            st.markdown(f'<div class="rx-top"><h4>{p["na"]} ↔ {p["nb"]}</h4>'
                        f'<span class="rx-badge">{p["conf"]*100:.1f}% confidence</span> '
                        f'{risk_badge(p["risk"])}'
                        f'<p style="margin-top:.7rem;font-size:1.04rem;">{p["desc"]}</p>'
                        f'<div class="rx-watch">⚠️ <b>What to watch for:</b> {p["note"]}</div></div>',
                        unsafe_allow_html=True)
            if len(meds) == 2 and p["others"]:
                st.markdown("###### Other possible interactions")
                for d, pr in p["others"]:
                    st.markdown(f'<div class="muted">{d}</div><div class="bar-wrap"><div class="bar-fill" '
                                f'style="width:{max(pr*100,1):.0f}%;"></div></div>'
                                f'<div class="muted">{pr*100:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="rx-pair" style="border-left-color:{border};">'
                        f'<b>{p["na"]} ↔ {p["nb"]}</b> '
                        f'<span class="rx-badge-sm">{p["conf"]*100:.1f}%</span> {risk_badge(p["risk"])}'
                        f'<div style="margin-top:.35rem;color:#33384d;">{p["desc"]}</div>'
                        f'<div class="muted" style="margin-top:.3rem;">What to watch for: {p["note"]}</div>'
                        f'</div>', unsafe_allow_html=True)

    with st.expander("🔬  Why these predictions? (model explainability)"):
        st.markdown("**What the model relies on overall**")
        imp = global_importances(bundle)
        mx = max(v for _, v in imp) or 1
        for name, val in imp:
            st.markdown(f'<div class="muted">{name} — {val*100:.1f}%</div>'
                        f'<div class="bar-wrap"><div class="imp-fill" '
                        f'style="width:{max(val/mx*100,1):.0f}%;"></div></div>', unsafe_allow_html=True)
        top = plist[0]
        tani, diffs = explain_pair(top["sa"], top["sb"], bundle)
        interp = ("very similar structures" if tani > 0.6 else
                  "moderately similar structures" if tani > 0.3 else "structurally distinct")
        st.markdown(f"**Top pair — {top['na']} vs {top['nb']}**")
        st.markdown(f"Structural (Tanimoto) similarity: **{tani:.2f}** — {interp}. "
                    "Shared substructure is a strong driver of how the model groups interactions.")
        dd = pd.DataFrame([{"Property": n, top['na']: f"{a:.2f}", top['nb']: f"{b:.2f}"}
                           for n, a, b, _ in diffs]).set_index("Property")
        st.markdown("Largest physicochemical differences:")
        st.table(dd)

    rows = [(p["na"], p["nb"], p["desc"], p["conf"], p["risk"], p["note"]) for p in plist]
    st.download_button("📄  Download report (PDF)", build_pdf(meds, rows, LOGO),
                       file_name=f"{APP_NAME.replace(' ', '_')}_report.pdf",
                       mime="application/pdf", key="dlpdf")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍  Interaction Checker", "💊  Drug Profile", "🕘  History", "ℹ️  About"])

# ============================ TAB 1 =============================
with tab1:
    st.markdown("#### Build a medication list")
    picks = st.multiselect("Pick from common medications (type to filter)",
                           sorted(DRUGS), default=["Aspirin", "Warfarin"])
    with st.container(border=True):
        st.markdown("**Not listed? Add any drug**")
        c1, c2 = st.columns([4, 1])
        nm = c1.text_input("By name (PubChem)", key="addname",
                           placeholder="e.g. clarithromycin", label_visibility="collapsed")
        if c2.button("Add", key="addnamebtn"):
            if nm.strip():
                smi = name_to_smiles(nm.strip())
                if smi:
                    st.session_state.extras.append((nm.strip().title(), smi))
                else:
                    st.warning("That name wasn't found on PubChem.")
        c3, c4 = st.columns([4, 1])
        sm = c3.text_input("Or by SMILES", key="addsmi",
                           placeholder="or paste a SMILES string", label_visibility="collapsed")
        if c4.button("Add", key="addsmibtn"):
            s = sm.strip()
            if s and Chem.MolFromSmiles(s):
                st.session_state.extras.append(("Custom molecule", s))
            elif s:
                st.warning("That SMILES couldn't be parsed.")
        if st.session_state.extras:
            st.caption("Added: " + ", ".join(n for n, _ in st.session_state.extras))
            if st.button("Clear added drugs"):
                st.session_state.extras = []

    meds, seen = [], set()
    for n, s in [(n, DRUGS[n]) for n in picks] + list(st.session_state.extras):
        if s not in seen:
            seen.add(s); meds.append((n, s))

    if meds:
        st.markdown("**Selected medications**")
        cols = st.columns(min(len(meds), 5))
        for i, (n, s) in enumerate(meds):
            with cols[i % 5]:
                uri = mol_uri(s, (170, 120))
                if uri:
                    st.markdown(f'<img src="{uri}" style="width:100%;border:1px solid #E6EAF3;'
                                f'border-radius:8px;background:#fff;"/>', unsafe_allow_html=True)
                st.caption(n)

    go = st.button("🔍  Check All Interactions", type="primary", disabled=len(meds) < 2)
    if len(meds) < 2:
        st.caption("Add at least two medications to run a check.")

    if go:
        plist = []
        for (na, sa), (nb, sb) in itertools.combinations(meds, 2):
            x = featurize_pair(sa, sb, bundle)
            if x is None:
                continue
            top = predict_top(x, bundle, labels, k=5)
            desc = with_names(top[0][0], na, nb)
            level, note = assess_risk(desc)
            plist.append({"na": na, "sa": sa, "nb": nb, "sb": sb,
                          "desc": desc, "conf": top[0][1], "risk": level, "note": note,
                          "others": [(with_names(d, na, nb), p) for d, p in top[1:]]})
        plist.sort(key=lambda r: (RISK_RANK[r["risk"]], -r["conf"]))
        entry = {"time": datetime.now().strftime("%d %b, %H:%M"), "meds": meds, "pairs": plist}
        st.session_state.last = entry
        st.session_state.history.insert(0, entry)

    if st.session_state.last:
        render_check(st.session_state.last)

# ============================ TAB 2 =============================
with tab2:
    st.markdown("#### Single-drug profile")
    how = st.radio("Choose drug by", ["List", "Name (PubChem)", "SMILES"], horizontal=True)
    nm2, smi2 = "Custom molecule", None
    if how == "List":
        nm2 = st.selectbox("Drug", sorted(DRUGS)); smi2 = DRUGS[nm2]
    elif how == "Name (PubChem)":
        q = st.text_input("Drug name", placeholder="e.g. atorvastatin")
        if q.strip():
            smi2 = name_to_smiles(q.strip()); nm2 = q.strip().title()
            if smi2 is None:
                st.warning("That name wasn't found on PubChem.")
    else:
        smi2 = (st.text_input("SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O").strip() or None)

    if smi2 and Chem.MolFromSmiles(smi2):
        m = Chem.MolFromSmiles(smi2)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f'<img src="{mol_uri(smi2, (320, 250))}" style="width:100%;border:1px solid '
                        f'#E6EAF3;border-radius:10px;background:#fff;"/>', unsafe_allow_html=True)
            st.caption(nm2)
        with c2:
            props = {k: DESC_FUNCS[k](m) for k in
                     ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "QED"]}
            st.table(pd.DataFrame({"Property": list(props),
                                   "Value": [f"{v:.2f}" for v in props.values()]}).set_index("Property"))
            ro5 = sum([props["MW"] <= 500, props["LogP"] <= 5, props["HBD"] <= 5, props["HBA"] <= 10])
            st.markdown(f"**Lipinski rule-of-5:** {'✅ passes' if ro5 == 4 else f'⚠️ {4-ro5} violation(s)'} "
                        f"<span class='muted'>({ro5}/4 criteria met)</span>", unsafe_allow_html=True)
    elif smi2:
        st.warning("Couldn't read that structure.")

# ============================ TAB 3 =============================
with tab3:
    st.markdown("#### Check history")
    st.caption("Your checks this session. (History clears when you refresh or restart the app.)")
    if not st.session_state.history:
        st.info("No checks yet — run one in the Interaction Checker tab.")
    else:
        if st.button("Clear history"):
            st.session_state.history = []
        else:
            for entry in st.session_state.history:
                top = entry["pairs"][0] if entry["pairs"] else None
                lbl = f"{entry['time']} · {len(entry['meds'])} meds"
                if top:
                    flag = "🚩 " if any(p["risk"] == "high" for p in entry["pairs"]) else ""
                    lbl += f" · {flag}top: {top['na']} ↔ {top['nb']} ({top['conf']*100:.0f}%)"
                with st.expander(lbl):
                    for p in entry["pairs"]:
                        st.markdown(f"**{p['na']} ↔ {p['nb']}** "
                                    f"<span class='rx-badge-sm'>{p['conf']*100:.0f}%</span> "
                                    f"{risk_badge(p['risk'])}<br>"
                                    f"<span class='muted'>{p['desc']}</span>", unsafe_allow_html=True)

# ============================ TAB 4 =============================
with tab4:
    st.markdown(f"#### About {APP_NAME}")
    st.markdown(
        f"**{APP_NAME}** estimates how pairs of medications may interact, using a machine-learning "
        "model trained on molecular structure. Built by **HayMedics Academy** to demonstrate "
        "responsible, transparent ML in healthcare.\n\n"
        "**How it works**\n"
        "- Trained on DrugBank's ~191,000 drug-pair interactions across 86 interaction types.\n"
        f"- Each drug → 10 physicochemical descriptors + a {bundle['n_bits']}-bit Morgan/ECFP "
        f"fingerprint (radius {bundle['radius']}); pairs combined order-invariantly so (A,B) == (B,A).\n"
        "- Model: RandomForest. Held-out performance ≈ **0.83 accuracy, 0.76 macro-F1**.\n"
        "- For more than two drugs, every pair is screened and ranked.\n\n"
        "**Explainability** — the checker shows the model's feature importances plus, for the top pair, "
        "structural (Tanimoto) similarity and the biggest physicochemical differences.\n\n"
        "**Risk flags & side-effect notes** — each predicted interaction *type* is mapped to its known "
        "clinical risk category (e.g. QT prolongation, bleeding, serotonin syndrome, CNS depression) "
        "and the symptoms to watch for. This is a curated educational layer over the prediction — **not** "
        "a validated severity score, and not personalised. A missing flag is **not** proof of safety.\n\n"
        "**Limitations**\n"
        "- Predicts the interaction *type*, not its true *severity* or *direction*.\n"
        "- Trained on known interactions, so absence of a flag isn't proof of safety.\n"
        "- An educational/research tool — **not a medical device**.\n\n"
        "_HayMedics Academy · Data | Research | Innovation._")

st.markdown(f'<p style="text-align:center;color:#5b6480;font-size:.8rem;margin-top:1.6rem;">'
            f'{APP_NAME} · HayMedics Academy</p>', unsafe_allow_html=True)
