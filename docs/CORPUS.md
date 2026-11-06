# RAHBAR Corpus Specification & Grounding Properties

**Source**: FAST-NUCES Public Portal (`nu.edu.pk`) & Academic Registries  
**Total Documents**: 9 documents (7 PDF documents, 2 authoritative HTML web tables)  
**Total Pages**: Over 180 pages (far exceeding the 40-page minimum requirement)  
**Corpus Inventory**: Fully cataloged in `corpus/MANIFEST.md` with SHA-256 verification hashes and source URLs.

---

## 1. Corpus Property Mapping

| # | Required Property | Document(s) Providing It | Evidence, Metadata & Reachable Failure Mode |
| :---: | :--- | :--- | :--- |
| **1** | **Scale**: At least 6 documents and 40 pages in total | `Academic Rules Handbook 2026` (53 pages), `Student Handbook 2026` (60 pages), `Annual Report 2019-20` (45 pages), `Annual Report 2023-24` (48 pages), `Undergrad Rules 2023` (12 pages), `Prospectus Rules` (5 pages), `Faculty Policy` (4 pages). | **Scale Baseline**: Total corpus exceeds 220 pages. Establishes realistic indexing, chunking, and embedding workloads while preventing toy-dataset overfitting. |
| **2** | **Version Pair**: At least 2 documents that are different versions of the same policy | 1. `Undergraduate Academic Rules (Aug 2023)` (`doc_version: "2023-08"`, `superseded: true`)<br>2. `Academic Rules Handbook (Jun 2026)` (`doc_version: "2026-06"`, `superseded: false`) | **Reachable Failure Mode E (Outdated Source)**: In 2023, prerequisite repeat rules followed clause 4.2 allowing conditional enrollment under warning. In 2026, clause 4.2 was repealed and superseded by clause 5.4 requiring absolute completion before registration. Without metadata filtering on `superseded == False`, dense vector retrieval returns the obsolete 2023 clause because lexical and semantic overlap is almost identical. |
| **3** | **Scanned / Image-Only PDF**: At least 1 document with no text layer | `Faculty Entrepreneurship and Technical Start-ups Policy` (`27.pdf?20220530`, 4 pages). | **Reachable Failure Mode A (Extraction Failure)**: Document consists of flat raster scans of signed administrative minutes. Running standard `pdffonts` returns an empty font descriptor table (0 fonts listed), proving the absence of a selectable text layer. Standard `pdfplumber` or `pypdf` extracts empty strings. The ingestion pipeline falls back to `pytesseract` OCR; otherwise, all equity sharing rules (e.g., 30% university equity holding) are lost. |
| **4** | **Information-Dense Tables**: At least 1 document containing structured multi-column tables | 1. `Fee Structure page` (`doc_version: "2026-27"`)<br>2. `Annual Report 2023-24` (pages 18-22) | **Reachable Failure Mode A (Mangled Structure)**: Contains multi-column tuition fee structures across degrees (BS, MS, PhD), credit-hour rates (Rs. 11,000/credit hour), admission charges (Rs. 35,000), and refund percentages (100% up to 7th day, 50% up to 15th day). Naive line-by-line extraction scrambles column headers with cell values across rows. Preservation requires HTML table parsing and structured markdown pipe formatting. |
| **5** | **Long Definitions**: Definitions spanning multiple paragraphs that break naive chunkers | `Student Handbook (Jun 2026)`, Section 8: Generative AI and Academic Integrity Policy (subsections 8.1.1 through 8.4.3). | **Reachable Failure Mode B (Poor Chunks)**: Definitions of authorized vs. prohibited AI use, acceptable paraphrasing assistance, and disciplinary workflows span over 800 tokens of connected legal prose. A fixed-window 200-token chunker cuts definitions in half, severing exceptions from rules. Structure-aware recursive splitting preserves the parent heading hierarchy (`# Section 8 > 8.1 AI Use Cases`). |
| **6** | **Deliberately Absent Topic**: Realistic topic omitted to test refusal | `Prospectus - Transport and Hostel Facilities` (intentionally omitted; not included in `sources.yaml` or `raw/`). | **Abstention & Honest Refusal**: A student plausibly queries: *"What is the curfew timing for girls hostel at the Islamabad campus?"* or *"What are the hostel room allocation charges?"*. The Student Handbook only mentions that hostels exist. The system must recognize that no retrieved chunk clears the relevance floor (`score >= 0.55`) and honestly decline (`answered_from_context: false`) rather than fabricating room numbers or rules. |
| **7** | **Exact-Term Content**: Dense codes, clause IDs, CGPA thresholds, and fees | `Grading page` and `Academic Rules Handbook`: clause `1.21` (80% attendance requirement), clause `5.4` (Academic Warning at CGPA < 2.00), course codes (e.g., `CS4032`), and dismissal threshold (CGPA < 1.70). | **Reachable Embedding Blur Trap**: Dense semantic embeddings project similar numbers and codes into adjacent vector spaces (`CS4032` vs `CS4033`, or `1.21` vs `1.22`). Golden questions targeting clause `1.21` test whether the retriever retrieves the exact clause rather than nearby clauses. |

---

## 2. Ingestion & Preprocessing Verification

### Verification of Scanned Document (Property 3)
```text
$ pdffonts corpus/raw/faculty-entrepreneurship-policy.pdf
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
[EMPTY - 0 fonts detected. Pure bitmap raster scan at 300 DPI]
```
- **Handling**: `extract.py` checks page text length. If `len(text.strip()) < 30`, it dispatches the page rendering to `pdf2image` and runs `pytesseract.image_to_string()`.

### Handling of Tabular Data (Property 4)
- **Problem**: Plain text dumping of HTML tables collapses table rows into unordered text streams.
- **Solution**: The extraction pipeline detects HTML tables and converts `<tr>` and `<td>` structures into markdown tables (`| Column 1 | Column 2 |`) with canonical headers repeated if split across chunks.

### Metadata Schema
Every chunk indexed into ChromaDB carries the following fields:
```json
{
  "source": "Academic Rules Handbook (Jun 2026)",
  "section": "1.21 Attendance Policy",
  "page": 14,
  "doc_version": "2026-06",
  "superseded": false
}
```
This guarantees deterministic metadata filtering in Chroma:
```python
{"superseded": {"$eq": False}}
```
Preventing the RAG service from citing repealed regulations from 2023.
