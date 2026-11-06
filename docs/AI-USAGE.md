# AI Usage and Attribution Disclosure

In accordance with course academic integrity policies and section 8 of the assignment handout, this document transparently records all AI-assisted tools utilized during the development, verification, and hardening of RAHBAR.

| Tool | What it wrote or shaped | What we changed afterwards, and why |
| :--- | :--- | :--- |
| **Google Antigravity / Gemini Assistant** | Assisted in drafting initial boilerplate structures for Pydantic models, FastAPI routes, and K8s manifest formatting. | Rigorously reviewed and refactored the extraction pipeline to incorporate custom OCR detection thresholds and markdown table formatting. Hand-tuned cosine distance-to-similarity transformations and added strict citation matching logic to prevent hallucinated references. |
| **GitHub Copilot CLI** | Provided auto-completions for regex matching patterns for legal clause headings (`\d+\.\d+`) and HTML table tags in extraction. | Replaced generic regex patterns with strict structure-aware paragraph and clause delimiters to ensure long policy definitions (like Generative AI guidelines) were not severed across chunk boundaries. |
| **Claude / Cursor Analysis** | Assisted in architecting the NetworkPolicy YAML templates and Calico CNI egress rule structures. | Adapted rules specifically to allow port 53 DNS egress to `kube-dns` in the `kube-system` namespace, preventing DNS resolution failure loops after default-deny policies were applied. |
| **Python Standard Library AST Tools** | Used to cross-check code against `scripts/check_submission.py` AST linter rules. | Removed all comments inside critical inspection blocks to ensure code semantics directly satisfied AST inspection without docstring ambiguity. |
