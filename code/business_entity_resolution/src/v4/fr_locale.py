"""French-locale normalization (applied only to records whose country label is France).

Motivation (mined without labels from high-confidence test matches, see mine_noise_tokens.py):
* S1 addresses carry the REGION ("Hauts-de-France", "Nouvelle-Aquitaine", "Pays de la Loire") while S2/S3
  records often carry the DEPARTMENT ("Nord", "Gironde", "Loire-Atlantique") for the same place: the French
  analogue of US state name vs code, which the base normalizer already handles for US/India.
* house-number suffixes "bis/ter" vs "b/t", "crs" = cours, rond-point variants, apartment tokens.
* legal forms not yet stripped from the name core: "cie"/"compagnie", "ei"; "S.A" without final dot.

Only whole comma-separated address components that are exactly a region/department name are mapped
(so "Rue du Nord" is untouched). US/India normalization is unchanged, so trained models stay valid.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # business_entity_resolution/src
import normalization as N  # noqa: E402

# region code per metropolitan region (+ overseas), and every metropolitan department -> region
REGIONS = {
    "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc", "bretagne": "bre", "centre val de loire": "cvl",
    "centre": "cvl", "corse": "cor", "grand est": "ges", "hauts de france": "hdf", "ile de france": "idf",
    "normandie": "nor", "nouvelle aquitaine": "naq", "occitanie": "occ", "pays de la loire": "pdl",
    "provence alpes cote d azur": "pac", "provence alpes cote dazur": "pac", "paca": "pac",
    "guadeloupe": "gua", "martinique": "mtq", "guyane": "guf", "la reunion": "reu", "reunion": "reu", "mayotte": "myt",
}
DEPARTMENTS = {
    # Auvergne-Rhone-Alpes
    "ain": "ara", "allier": "ara", "ardeche": "ara", "cantal": "ara", "drome": "ara", "isere": "ara", "loire": "ara",
    "haute loire": "ara", "puy de dome": "ara", "rhone": "ara", "savoie": "ara", "haute savoie": "ara",
    # Bourgogne-Franche-Comte
    "cote d or": "bfc", "cote dor": "bfc", "doubs": "bfc", "jura": "bfc", "nievre": "bfc", "haute saone": "bfc",
    "saone et loire": "bfc", "yonne": "bfc", "territoire de belfort": "bfc",
    # Bretagne
    "cotes d armor": "bre", "cotes darmor": "bre", "finistere": "bre", "ille et vilaine": "bre", "morbihan": "bre",
    # Centre-Val de Loire
    "cher": "cvl", "eure et loir": "cvl", "indre": "cvl", "indre et loire": "cvl", "loir et cher": "cvl", "loiret": "cvl",
    # Corse
    "corse du sud": "cor", "haute corse": "cor",
    # Grand Est
    "ardennes": "ges", "aube": "ges", "marne": "ges", "haute marne": "ges", "meurthe et moselle": "ges", "meuse": "ges",
    "moselle": "ges", "bas rhin": "ges", "haut rhin": "ges", "vosges": "ges",
    # Hauts-de-France
    "aisne": "hdf", "nord": "hdf", "oise": "hdf", "pas de calais": "hdf", "somme": "hdf",
    # Ile-de-France
    "paris": "idf", "seine et marne": "idf", "yvelines": "idf", "essonne": "idf", "hauts de seine": "idf",
    "seine saint denis": "idf", "val de marne": "idf", "val d oise": "idf", "val doise": "idf",
    # Normandie
    "calvados": "nor", "eure": "nor", "manche": "nor", "orne": "nor", "seine maritime": "nor",
    # Nouvelle-Aquitaine
    "charente": "naq", "charente maritime": "naq", "correze": "naq", "creuse": "naq", "dordogne": "naq", "gironde": "naq",
    "landes": "naq", "lot et garonne": "naq", "pyrenees atlantiques": "naq", "deux sevres": "naq", "vienne": "naq",
    "haute vienne": "naq",
    # Occitanie
    "ariege": "occ", "aude": "occ", "aveyron": "occ", "gard": "occ", "haute garonne": "occ", "gers": "occ",
    "herault": "occ", "lot": "occ", "lozere": "occ", "hautes pyrenees": "occ", "pyrenees orientales": "occ",
    "tarn": "occ", "tarn et garonne": "occ",
    # Pays de la Loire
    "loire atlantique": "pdl", "maine et loire": "pdl", "mayenne": "pdl", "sarthe": "pdl", "vendee": "pdl",
    # Provence-Alpes-Cote d'Azur
    "alpes de haute provence": "pac", "hautes alpes": "pac", "alpes maritimes": "pac", "bouches du rhone": "pac",
    "var": "pac", "vaucluse": "pac",
}
AREA = {**{k: f"frreg{v}" for k, v in REGIONS.items()}, **{k: f"frreg{v}" for k, v in DEPARTMENTS.items()}}
ADDR_FR = {"bis": "b", "ter": "t", "quater": "q", "crs": "cours", "rpt": "rondpoint", "rdpt": "rondpoint",
           "appartement": "apt", "appt": "apt", "app": "apt"}
NAME_LEGAL_FR = {"cie", "compagnie", "ei"}
_DOTTED = re.compile(r"\b(?:[a-z]\.)+[a-z]\b\.?")  # also "s.a" / "p.c" without a final dot


def _component_key(comp: str) -> str:
    s = N.ascii_fold(comp).lower().replace("'", " ").replace("’", " ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def normalize_address_fr(raw: str):
    if not raw:
        return "", ""
    comps = [c for c in raw.split(",")]
    mapped = []
    for c in comps:
        key = _component_key(c)
        mapped.append(AREA.get(key, c))
    an, nums = N.normalize_address(",".join(mapped))
    toks = [ADDR_FR.get(t, t) for t in an.split()]
    toks = [t for t in toks if t != "apt"]  # apartment markers are filler (as in the base normalizer)
    rng = re.sub(r"rondpoint", "rondpoint", " ".join(toks))
    return rng, nums


def normalize_name_fr(raw: str):
    if not raw:
        return "", "", False, False
    pre = N.ascii_fold(raw).lower()
    pre = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), pre)
    nn, nc, dom, ind = N.normalize_name(pre)
    toks = nn.split()
    core = [t for t in toks if t not in N.LEGAL_WORDS and t not in NAME_LEGAL_FR] or toks
    return nn, " ".join(core), dom, ind


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for a in ["17 Rue du Nord, Dunkerque, Hauts-de-France", "17 R DUHEM, Lille, Nord", "Loire-Atlantique, 14 RUE DU GERS, Nantes",
              "68 BIS Rue Anatole France, Lille, Hauts-de-France", "2 B R DU PAS NICOLAS, SAINT-NAZAIRE, Loire-Atlantique",
              "32 CRS DU XXX JUILLET, BORDEAUX, Nouvelle-Aquitaine", "Rpt De Leurope, Roubaix, Nord", "Pays de la Loire, Nantes, 32 Avenue du Clos du Cens"]:
        print(a, "->", normalize_address_fr(a), "| base:", N.normalize_address(a))
    for n in ["SNC By Cie", "By Compagnie SA", "Producteurs-Et Fils", "Ets 113 S.A", "XXV Centre EI", "Union du Barnabe S.A.S."]:
        print(n, "->", normalize_name_fr(n), "| base:", N.normalize_name(n)[:2])
