# Agent Guardrails

## Fuzzy Matching Safety

- Never overwrite `bin/json/images_parser.json` with fuzzy-corrected values.
- `images_parser.json` is parser/source metadata and must remain OCR/parsing output.
- Fuzzy normalization is allowed only in derived outputs:
  - `bin/csv/images_metadata.csv`
  - `bin/csv/images_metadata_fuzzy_log.csv`
- Any new pipeline step that applies typo/fuzzy correction must preserve this separation.

## Report Bank Verification

- Before and after editing `bin/json/report_bank_data.json` or `src/app/services/report_bank.py`, run:
  - `python src/scripts/verify_report_bank_completeness.py`
- When adding a specific new report, also require its title:
  - `python src/scripts/verify_report_bank_completeness.py --expect-title "Median Year Histogram by Submitter (Unique Players)"`
- Do not mark report-bank work complete if this check reports an error.
