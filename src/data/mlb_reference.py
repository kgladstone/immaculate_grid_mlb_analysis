import pandas as pd
import re
from rapidfuzz import process
import unicodedata
import unidecode
from pybaseball import chadwick_register

# Known aliases for common Grid-style issues
MANUAL_ALIASES = {
    "Henry Aaron": "Hank Aaron",
    "Larry Berra": "Yogi Berra",
    "Robert Gibson": "Bob Gibson",
    "Lawrence Jackson": "Bo Jackson",
    "Fred Mcgriff": "Fred McGriff",
    "Robert Dickey": "R.A. Dickey",
    "Allan Burnett": "A.J. Burnett",
    "David Parker": "Dave Parker",
    "William Madlock": "Bill Madlock",
    "Melvin Cabrera": "Miguel Cabrera",
    "George Ruth": "Babe Ruth",
    "John Pedro Gonzalez": "Pedro Gonzalez",
    "Happ": "Ian Happ",
    "Jarrod": "Jarrod Saltalamacchia",
    "Grover Alexander": "Grover Cleveland Alexander",
    "Bobby Witt Jr.": "Bobby Witt Jr.",
    "Fernando Tatis Jr.": "Fernando Tatis Jr.",
    "Vladimir Guerrero Jr.": "Vladimir Guerrero Jr.",
    # Add more as needed...
}

def load_mlb_player_names():
    print("🔁 Loading MLB names from Chadwick register...")
    df = chadwick_register()

    df['name_first'] = df['name_first'].apply(lambda x: unidecode.unidecode(str(x)).strip())
    df['name_last'] = df['name_last'].apply(lambda x: unidecode.unidecode(str(x)).strip())

    if 'name_given' in df.columns:
        df['name_given'] = df['name_given'].apply(lambda x: unidecode.unidecode(str(x)).strip() if pd.notna(x) else '')

    names = set()

    for _, row in df.iterrows():
        first, last = row['name_first'], row['name_last']
        if first and last:
            base = f"{first} {last}"
            names.add(base)
            suffix = str(row.get('name_suffix', '')).strip()
            if suffix:
                names.add(f"{base} {suffix}")

        # Add full legal name if available
        if 'name_given' in row and row['name_given']:
            given = row['name_given']
            if len(given.split()) >= 2:
                names.add(given)

    print(f"✅ Loaded {len(names):,} canonical player names")
    return names

def clean_name(name):
    if not isinstance(name, str):
        return name

    # Normalize accents (e.g., José → Jose)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')

    # Replace problematic characters
    name = name.replace('!', 'l')
    name = name.replace('|', ' ')  # << this is the fix: treat pipe as space
    name = name.strip()

    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name)

    # Fix suffixes like Ji/J./Sr./etc
    suffix = ''
    if re.search(r'\b(Jr|Ji|J\.|Sr|S\.?)\b$', name, re.IGNORECASE):
        suffix = 'Jr' if re.search(r'\bJ', name, re.IGNORECASE) else 'Sr'
        name = re.sub(r'\b(Jr|Ji|J\.|Sr|S\.?)\b$', '', name, flags=re.IGNORECASE).strip()

    # Remove anything not part of a legit name
    name = re.sub(r'[^A-Za-z.\-\' ]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    if suffix:
        name = f"{name} {suffix}"

    return name.title()



def correct_typos_with_fuzzy_matching(df, response_column, similarity_threshold=0.85):
    print(f"Correcting typos in column '{response_column}'...")
    mlb_names = load_mlb_player_names()
    canonical_mapping = {}

    def get_corrected(name):
        cleaned = clean_name(name)
        if cleaned in canonical_mapping:
            return canonical_mapping[cleaned]

        # Apply manual override if available
        if cleaned in MANUAL_ALIASES:
            canonical_mapping[cleaned] = MANUAL_ALIASES[cleaned]
            return MANUAL_ALIASES[cleaned]

        # Fuzzy match to best MLB name
        result = process.extractOne(cleaned, mlb_names)
        if result:
            match, _, _ = result
        else:
            match = cleaned
        canonical_mapping[cleaned] = match
        return match

    corrected_rows, changes_log = [], []

    for _, row in df.iterrows():
        original = row[response_column]
        corrected = original

        if isinstance(original, dict):
            corrected = {}
            for k, v in original.items():
                if isinstance(v, str) and len(v.strip()) > 0:
                    new = get_corrected(v)
                    if new != v:
                        changes_log.append({
                            "grid_number": row['grid_number'],
                            "submitter": row['submitter'],
                            "original_name": v,
                            "corrected_name": new,
                            "response_location": k
                        })
                    corrected[k] = new
                else:
                    corrected[k] = v
        elif isinstance(original, str) and len(original.strip()) > 0:
            new = get_corrected(original)
            if new != original:
                changes_log.append({
                    "grid_number": row['grid_number'],
                    "submitter": row['submitter'],
                    "original_name": original,
                    "corrected_name": new,
                    "response_location": None
                })
            corrected = new

        row_copy = row.copy()
        row_copy[response_column] = corrected
        corrected_rows.append(row_copy)

    return pd.DataFrame(corrected_rows), pd.DataFrame(changes_log)
