"""Conservative matching views for the four-column challenge record schema.

The raw columns are returned unchanged.  Derived text is normalized to NFC,
lowercased, and spaced at punctuation while retaining Unicode letters, marks,
and numbers.  Accent folding is conditional: after punctuation cleanup it is
applied only to Latin-only text.  Mixed-script text and scripts such as
Devanagari keep their combining marks.

``name_core`` removes a sequence of known legal suffix tokens only at the end.
``address_canonical`` is an additional token-abbreviation view, not an address
parser.  ``address_numbers`` contains every maximal Unicode number run in input
order, including runs inside alphanumeric tokens, with repeats and leading
zeros preserved.  It makes no claim about which number is a house number,
unit, floor, or postcode.

This module imports no third-party package and performs no I/O.  The returned
SQL targets DuckDB 1.3.2 and later.  The caller supplies a trusted relation SQL
fragment with VARCHAR columns entity_id, business_name, business_address, and
country; it may be a quoted table name, table function, or parenthesized query.
Do not pass untrusted user text as the relation fragment.
"""

from __future__ import annotations


PREPROCESS_VERSION = "aj-preprocess-v1-unicode-latin-fold"

# These are whole tokens, and only a trailing sequence is removed.  In
# particular, generic words such as services, group, trade, and company remain.
_LEGAL_SUFFIXES = (
    "incorporated", "inc", "corporation", "corp", "limited", "ltd",
    "llc", "llp", "private", "pvt", "plc",
)
_SUFFIX_ALTERNATION = "|".join(_LEGAL_SUFFIXES)
_TRAILING_SUFFIX_PATTERN = (
    rf"(^| )({_SUFFIX_ALTERNATION})( ({_SUFFIX_ALTERNATION}))*$"
)


def normalized_sql(input_relation: str) -> str:
    """Return a SELECT adding separate, explicitly named matching views.

    The original fields, including NULLs and country labels, are not changed.
    NULL or punctuation-only names/addresses become empty derived strings.
    ``name_is_empty`` and ``address_is_empty`` refer to those folded strings;
    ``name_core_is_empty`` separately identifies a suffix-only name whose
    folded text still exists.  ``rid`` is exactly ``entity_id``.

    The relation argument is trusted SQL, not a path or a SQL value to quote.
    """
    if not isinstance(input_relation, str):
        raise TypeError("input_relation must be a trusted SQL relation string")
    relation = input_relation.strip()
    if not relation:
        raise ValueError("input_relation must not be empty")

    return rf"""
WITH _aj_preprocess_unicode AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        lower(nfc_normalize(coalesce(business_name, ''))) AS name_unicode,
        lower(nfc_normalize(coalesce(business_address, ''))) AS address_unicode
    FROM {relation}
),
_aj_preprocess_spaced AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        trim(regexp_replace(name_unicode, '[^\p{{L}}\p{{M}}\p{{N}}]+', ' ', 'g'))
            AS name_spaced,
        trim(regexp_replace(address_unicode, '[^\p{{L}}\p{{M}}\p{{N}}]+', ' ', 'g'))
            AS address_spaced
    FROM _aj_preprocess_unicode
),
_aj_preprocess_folded AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        CASE WHEN NOT regexp_matches(name_spaced, '[^\p{{Latin}}\p{{N}} ]')
            THEN strip_accents(name_spaced) ELSE name_spaced END AS name_folded,
        CASE WHEN NOT regexp_matches(address_spaced, '[^\p{{Latin}}\p{{N}} ]')
            THEN strip_accents(address_spaced) ELSE address_spaced END AS address_folded
    FROM _aj_preprocess_spaced
),
_aj_preprocess_views AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        name_folded,
        trim(regexp_replace(name_folded, '{_TRAILING_SUFFIX_PATTERN}', '')) AS name_core,
        address_folded,
        array_to_string(
            list_transform(string_split(address_folded, ' '), tok -> CASE tok
                WHEN 'road' THEN 'rd'
                WHEN 'street' THEN 'st'
                WHEN 'avenue' THEN 'ave'
                WHEN 'boulevard' THEN 'blvd'
                WHEN 'lane' THEN 'ln'
                WHEN 'drive' THEN 'dr'
                ELSE tok END),
            ' '
        ) AS address_canonical,
        regexp_extract_all(address_folded, '\p{{N}}+') AS address_numbers
    FROM _aj_preprocess_folded
)
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    entity_id AS rid,
    name_folded,
    name_core,
    address_folded,
    address_canonical,
    address_numbers,
    name_folded = '' AS name_is_empty,
    address_folded = '' AS address_is_empty,
    name_core = '' AS name_core_is_empty
FROM _aj_preprocess_views
""".strip()
