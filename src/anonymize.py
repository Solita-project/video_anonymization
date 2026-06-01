# Removes personal data from transcription.json
# Keeps the original transcription.json unchanged
# Saves cleaned transcript to data/output/cleaned_transcription.json
# Usage:
# source venvs/transcript/Scripts/activate
# python src/anonymize.py

from pathlib import Path
import json
import re

import spacy


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define input transcript file
INPUT_FILE = ROOT_DIR / "data" / "output" / "transcription.json"

# Define cleaned transcript output file
OUTPUT_FILE = ROOT_DIR / "data" / "output" / "cleaned_transcription.json"


# Regex patterns for Finnish personal data
PATTERNS = [
    # Finnish personal identity code in normal written form, for example 120580-123X
    ("HENKILÖTUNNUS", r"\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}[-+A]\d{3}[0-9A-FHJ-NPR-Y]\b"),

    # Finnish personal identity code after hetu or henkilötunnus words
    ("HENKILÖTUNNUS", r"\b(?:hetu|henkilö?tunnus\w*|henki\S{0,20}tunn\w*|sotu|sosiaaliturvatunnus\w*)\s*(?:on|oli|:|=)?\s*((?:0?[1-9]|[12]\d|3[01])\s*(?:0?[1-9]|1[0-2])\s*\d{2,4}\s*(?:[-+A]|viiva|miinus|plus|a)?\s*\d{1,3}\s*[0-9A-ZÅÄÖ]{1,5})\b"),

    # Patient ID when written as one word or spoken as separate words
    ("POTILASTUNNUS", r"\b(?:potilasnumero|potilasnro|potilastunnus|potilas\s+numero|potilas\s+nro|potilas\s+tunnus)\s*(?:on|oli|:|=)?\s*([A-ZÅÄÖ]{0,4}\s*[-/]?\s*\d{1,8}(?:\s*[-/]?\s*\d{1,8})*)\b"),

    # Other healthcare-related identifying numbers
    ("POTILASTUNNUS", r"\b(?:asiakasnumero|asiakasnro|käyntinumero|käyntinro|lähetenumero|lähetenro|tutkimusnumero|näytenumero)\s*(?:on|oli|:|=)?\s*([A-ZÅÄÖ]{0,4}\s*[-/]?\s*\d{2,12}(?:\s*[-/]?\s*\d{1,12})*)\b"),

    # Email after email keyword when ASR misses @, for example anna.kurhonen.example.fi
    ("SÄHKÖPOSTI", r"\b(?:sähköposti|sähköpostiosoite|email|e-mail)\s*(?:on|oli|:|=)?\s*([A-Za-zÅÄÖåäö0-9._%+-]+\.[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),

    # Normal email address, for example name@example.com
    ("SÄHKÖPOSTI", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),

    # Spoken email address, for example mattivirtanen at example.com
    ("SÄHKÖPOSTI", r"\b[A-Za-zÅÄÖåäö0-9._%+-]+\s+(?:at|ät|miukumauku)\s+[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),

    # Spoken email address with spoken dot, for example matti at example piste fi
    ("SÄHKÖPOSTI", r"\b[A-Za-zÅÄÖåäö0-9._%+-]+\s+(?:at|ät|miukumauku)\s+[A-Za-z0-9-]+\s+(?:piste|dot)\s+[A-Za-z]{2,}\b"),

    # Finnish phone number in numeric form
    ("PUHELINNUMERO", r"\b(?:\+358|0)\s?(?:\d[\s-]?){6,12}\d\b"),

    # Finnish phone number after a phone keyword, including spoken number words
    ("PUHELINNUMERO", r"\b(?:puhelinnumero|puhelin\s+numero|puhelin|puh\.?|gsm)\s*(?:on|oli|:|=)?\s*((?:\+358|0|nolla)\s*(?:(?:\d+|nolla|yksi|kaksi|kolme|neljä|viisi|kuusi|seitsemän|kahdeksan|yhdeksän)[\s-]+){4,14}(?:\d+|nolla|yksi|kaksi|kolme|neljä|viisi|kuusi|seitsemän|kahdeksan|yhdeksän))\b"),

    # Date of birth
    ("SYNTYMÄAIKA", r"\b(?:syntynyt|syntymäaika|s\.aika|syntymäpäivä)\s*(?:on|oli|:|=)?\s*(\d{1,2}\.\d{1,2}\.\d{2,4})\b"),

    # Street address, for example Esimerkkikatu 12A
    ("OSOITE", r"\b[A-ZÅÄÖa-zåäö-]+(?:katu|tie|kuja|polku|kaari|rinne|raitti|aukio|väylä|puisto|tori|ranta|penger|rinne)\s+\d+[A-Za-z]?\b"),

    # Street address after address keyword, for example osoitteessa Koivutie 8 B
    ("OSOITE", r"\b(?:osoite|osoitteessa|asuu\s+osoitteessa)\s*(?:on|oli|:|=)?\s*([A-ZÅÄÖa-zåäö-]+(?:katu|tie|kuja|polku|kaari|rinne|raitti|aukio|väylä|puisto|tori|ranta|penger)\s+\d+\s*[A-Za-z]?)\b"),

    # Postal code and city in normal form, for example 00100 Helsinki
    ("OSOITE", r"\b\d{5}\s+(?!Hän\b|Potilas\b|potilas\b|on\b)[A-ZÅÄÖ][A-ZÅÄÖa-zåäö-]{2,}\b"),

    # Postal code and city in spoken or split form, for example 00 100 Helsinki
    ("OSOITE", r"\b\d{2}\s+\d{3}\s+(?!Hän\b|Potilas\b|potilas\b|on\b)[A-ZÅÄÖ][A-ZÅÄÖa-zåäö-]{2,}\b"),

    # Hospital or healthcare unit names, for example Meilahden sairaalassa
    ("ORGANISAATIO", r"\b(?!Potilas\b|Hän\b)[A-ZÅÄÖ][A-ZÅÄÖa-zåäö-]+\s+(?:yliopistollinen|yliopistollisessa|keskus|keskusessa|keskusairaala|keskussairaalassa)?\s*(?:sairaala|sairaalassa|sairaalaan|sairaalasta|terveysasema|terveysasemalla|terveysasemalle|terveyskeskus|terveyskeskuksessa|klinikka|klinikalla|poliklinikka|poliklinikalla)\b"),

    # Common Finnish hospital abbreviations
    ("ORGANISAATIO", r"\b(?:HUS|HYKS|TAYS|TYKS|KYS|OYS)\b"),
]


# spaCy entity labels that are personal or identifying
SPACY_LABELS = {
    "PER": "NIMI",
    "PERSON": "NIMI",
    "LOC": "SIJAINTI",
    "GPE": "SIJAINTI",
    "ORG": "ORGANISAATIO",
}


# Words that should not be removed even if spaCy thinks they are entities
SAFE_SPACY_WORDS = {
    "potilas",
    "potilaan",
    "potilaalla",
    "potilaalle",
    "potilasta",
    "aikuinen",
    "mies",
    "nainen",
    "lapsi",
    "poika",
    "tyttö",
    "vauva",
    "vastasyntynyt",
}


def load_model():
    # Load Finnish spaCy model
    return spacy.load("fi_core_news_sm")


def load_json(path):
    # Read JSON file
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    # Create output folder
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def add_regex_spans(text, spans):
    # Find personal data with regex patterns
    for label, pattern in PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):

            # If regex has a capture group, anonymize only that group
            if match.lastindex:
                start = match.start(1)
                end = match.end(1)

            # Otherwise anonymize the full match
            else:
                start = match.start()
                end = match.end()

            spans.append({
                "start": start,
                "end": end,
                "label": label,
            })


def add_spacy_spans(text, spans, nlp):
    # Find names, locations and organizations with spaCy
    doc = nlp(text)

    for ent in doc.ents:
        # Clean entity text before checking it
        ent_text = ent.text.strip(".,:;!? ").lower()

        # Keep safe medical words visible
        if ent_text in SAFE_SPACY_WORDS:
            continue

        # Get replacement label
        label = SPACY_LABELS.get(ent.label_)

        # Save entity span if it should be anonymized
        if label:
            spans.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "label": label,
            })


def clean_spans(spans):
    # Sort spans by start position and length
    spans = sorted(spans, key=lambda item: (item["start"], -(item["end"] - item["start"])))

    # Remove overlapping spans
    cleaned = []
    last_end = -1

    for span in spans:
        if span["start"] >= last_end:
            cleaned.append(span)
            last_end = span["end"]

    return cleaned


def find_sensitive_spans(text, nlp):
    # Store sensitive text positions
    spans = []

    # Add regex matches
    add_regex_spans(text, spans)

    # Add spaCy matches
    add_spacy_spans(text, spans, nlp)

    # Return cleaned spans
    return clean_spans(spans)


def replace_text(text, spans):
    # Return original text if nothing was found
    if not spans:
        return text

    # Build anonymized text
    parts = []
    last = 0

    for span in spans:
        parts.append(text[last:span["start"]])
        parts.append(span["label"])
        last = span["end"]

    parts.append(text[last:])

    return "".join(parts)


def overlaps(start, end, span):
    # Check if word overlaps sensitive span
    return start < span["end"] and end > span["start"]


def find_word_positions(text, words):
    # Store word positions inside segment text
    positions = []
    cursor = 0

    for word in words:
        word_text = word.get("word", "").strip()

        if not word_text:
            positions.append(None)
            continue

        index = text.find(word_text, cursor)

        if index == -1:
            positions.append(None)
            continue

        start = index
        end = index + len(word_text)
        positions.append((start, end))
        cursor = end

    return positions


def anonymize_words(segment, spans, nlp):
    # Read word list
    words = segment.get("words", [])

    # Read original segment text
    text = segment.get("text", "")

    # Match words to text positions
    positions = find_word_positions(text, words)

    # Keep track of already written labels
    used_spans = set()

    # Anonymize word-level data
    for index, word in enumerate(words):
        position = positions[index]

        # If word position was not found, check the word alone
        if position is None:
            word_spans = find_sensitive_spans(word.get("word", ""), nlp)
            word["word"] = replace_text(word.get("word", ""), word_spans)
            continue

        start, end = position

        # Check if this word overlaps sensitive text
        for span_index, span in enumerate(spans):
            if overlaps(start, end, span):

                # Write the anonymization label only once
                if span_index not in used_spans:
                    word["word"] = span["label"]
                    used_spans.add(span_index)

                # Remove the rest of the words inside the same sensitive span
                else:
                    word["word"] = ""

                break


def anonymize_segment(segment, nlp):
    # Read original segment text
    text = segment.get("text", "")

    # Find personal data
    spans = find_sensitive_spans(text, nlp)

    # Anonymize word-level transcript first
    anonymize_words(segment, spans, nlp)

    # Anonymize full segment text
    segment["text"] = replace_text(text, spans)


def anonymize():
    # Load Finnish NLP model
    nlp = load_model()

    # Load original transcript
    transcript = load_json(INPUT_FILE)

    # Anonymize each segment
    for segment in transcript:
        anonymize_segment(segment, nlp)

    # Save cleaned transcript as a new file
    save_json(transcript, OUTPUT_FILE)

    # Show output file
    print(f"Cleaned transcript saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    # Start anonymization
    anonymize()