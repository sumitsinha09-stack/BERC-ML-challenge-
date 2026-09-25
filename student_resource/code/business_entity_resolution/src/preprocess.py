import pandas as pd
import re
import gc
import os

LEGAL_SUFFIXES = [
    r'\bprivate\s+limited\b', r'\bpvt\s+ltd\b', r'\bpvt\s+limited\b',
    r'\blimited\b', r'\bltd\b',
    r'\bcorporation\b', r'\bcorp\b',
    r'\bincorporated\b', r'\binc\b',
    r'\bcompany\b', r'\bco\b',
    r'\bllc\b', r'\bllp\b',
    r'\bsarl\b', r'\bsa\b', r'\bsas\b', r'\bsarl\b',  # French
    r'\bgmbh\b', r'\bag\b',  # German (just in case)
    r'\benterprises?\b', r'\bsolutions?\b', r'\bservices?\b',
    r'\bgroup\b', r'\bholdings?\b',
]

ADDRESS_ABBREVIATIONS = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive', r'\bln\b': 'lane',
    r'\bct\b': 'court', r'\bpl\b': 'place', r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway', r'\bapt\b': 'apartment', r'\bste\b': 'suite',
    r'\bbldg\b': 'building', r'\bfl\b': 'floor', r'\bdist\b': 'district',
    r'\bnr\b': 'near', r'\bopp\b': 'opposite',
}

US_STATE_ABBR = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}


def normalize_name(name):
    if pd.isna(name) or not isinstance(name, str):
        return ''
    text = name.lower().strip()
    text = text.replace('&', ' and ')
    text = text.replace('.', ' ').replace(',', ' ').replace('-', ' ')
    text = re.sub(r"['\"]", '', text)
    for suffix in LEGAL_SUFFIXES:
        text = re.sub(suffix, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_address(address):
    if pd.isna(address) or not isinstance(address, str):
        return ''
    text = address.lower().strip()
    text = text.replace(',', ' ').replace('.', ' ').replace('-', ' ')
    for abbr, full in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(abbr, full, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_name_tokens(name):
    if not name:
        return ''
    tokens = name.split()
    tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens)


def preprocess_dataframe(df):
    df['name_clean'] = df['business_name'].apply(normalize_name)
    df['addr_clean'] = df['business_address'].apply(normalize_address)
    df['name_tokens'] = df['name_clean'].apply(extract_name_tokens)
    return df


def load_and_preprocess(data_dir, prefix, output_dir):
    """Load source files, preprocess, and save per-country chunks."""
    os.makedirs(output_dir, exist_ok=True)

    for source_num in [1, 2, 3]:
        filepath = os.path.join(data_dir, f'{prefix}_source{source_num}.tsv')
        print(f'Loading {filepath}...')
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        df = df.fillna('')
        print(f'  Loaded {len(df)} rows')

        print(f'  Preprocessing...')
        df = preprocess_dataframe(df)

        countries = df['country'].unique()
        for country in countries:
            country_df = df[df['country'] == country]
            out_path = os.path.join(
                output_dir,
                f'{prefix}_source{source_num}_{country.lower().replace(" ", "_")}.tsv'
            )
            country_df.to_csv(out_path, sep='\t', index=False)
            print(f'  Saved {len(country_df)} rows for {country} → {out_path}')

        del df
        gc.collect()


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    processed_dir = os.path.join(base_dir, 'processed_data')

    print('=== Preprocessing TRAINING data ===')
    load_and_preprocess(
        os.path.join(base_dir, 'dataset', 'train'),
        'train',
        processed_dir
    )

    print('\n=== Preprocessing TEST data ===')
    load_and_preprocess(
        os.path.join(base_dir, 'dataset', 'test'),
        'test',
        processed_dir
    )

    print('\nDone! Preprocessed files saved to:', processed_dir)
