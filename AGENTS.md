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

## PDF Export Cleanliness

- Keep `bin/json/report_bank_data.json` limited to real analytics reports only.
- PDF-only sections, such as `Rule 5 Bans + Screenshots`, must stay as explicit PDF pseudo-sections in `src/app/tabs/analytics_tab.py`, not as report-bank rows.
- Every report-bank row must have an accurate `type`: use `"chart"` for graph/visual reports and `"table"` for tabular/text reports.
- If a PDF pseudo-section is user-selectable, it must appear in the PDF checklist, be excluded from Excel, and be honored by both regular PDF export and Basic Export Mode.
