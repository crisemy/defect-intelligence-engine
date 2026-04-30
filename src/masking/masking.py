import pandas as pd
import hashlib
from pathlib import Path

RAW_PATH = Path("data/raw")
MASKED_PATH = Path("data/masked")

def hash_value(value: str) -> str:
    """Return SHA256 hash of a string value."""
    if pd.isna(value):
        return value
    return hashlib.sha256(value.encode()).hexdigest()

def mask_version(value):
    """Mask version strings and special patterns by replacing known prefixes and versions."""
    if pd.isna(value):
        return value
    
    # Preservar Planning tal cual
    if value.strip() == 'Planning':
        return value
    
    # Reemplazos específicos de patrones completos o parciales
    replacements = {
        'TMC Powershell': 'APP_PS',
        'Test Automation': 'APP_Automation',
        'TMC_': 'APP_',
        'TMI_': 'APP_',
    }
    
    # Aplicar reemplazos exactos primero
    for old, new in replacements.items():
        if old in value:
            value = value.replace(old, new)
    
    # Reemplazar versiones 6.x → 1.x (después de los reemplazos anteriores)
    value = value.replace('6.', '1.')
    
    return value

def abbreviate_component(value):
    """Abbreviate component names to obscure origin while keeping them short and distinguishable."""
    if pd.isna(value):
        return value
    # Split into words
    words = value.split()
    if len(words) > 1:
        # Para nombres con varias palabras: inicial de cada palabra + .
        return '.'.join(w[0].upper() for w in words)
    else:
        # Para una sola palabra: primeras dos letras
        return value[:2].capitalize()

def mask_dataset(input_filename: str, output_filename: str) -> None:
    """
    Load raw dataset, anonymize sensitive fields,
    and save masked dataset.
    """
    input_path = RAW_PATH / input_filename
    output_path = MASKED_PATH / output_filename

    df = pd.read_csv(input_path)

    # ---- Drop sensitive columns ----
    columns_to_drop = [
        "Summary",
        "Assignee Id",
        "Issue id"
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # ---- Hash Assignee ----
    if "Assignee" in df.columns:
        df["Assignee"] = df["Assignee"].apply(hash_value)

    # ---- Replace Issue key with internal numeric id ----
    if "Issue key" in df.columns:
        df["Issue key"] = range(1, len(df) + 1)

    # ---- Mask Fix versions and Affects versions columns ----
    fix_cols = [col for col in df.columns if col.startswith('Fix versions')]
    affects_cols = [col for col in df.columns if col.startswith('Affects versions')]
    for col in fix_cols + affects_cols:
        df[col] = df[col].apply(mask_version)

    # ---- Abbreviate Components columns ----
    component_cols = [col for col in df.columns if col.startswith('Components')]
    for col in component_cols:
        df[col] = df[col].apply(abbreviate_component)

    # Ensure output directory exists
    MASKED_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Masked dataset saved to {output_path}")

if __name__ == "__main__":
    mask_dataset("jira_dataset.csv", "masked_dataset.csv")