# The archive's knowledge

Every `.md` file in this folder is a document the archive can answer from. Replace the placeholder files with verified information about Dhruv Bendre and the archive starts answering; nothing else changes.

## File format

```markdown
---
category: Projects
title: Projects
source: knowledge/projects.md
date: 2026-01-01
project:
document_type: catalogue
status: placeholder
---

# Projects

## Project name

What it is, what problem it solved, what was built, what it taught.
```

- **Frontmatter** (the block between `---` lines) is metadata. `category` and `title` appear in the "From the archive" line under answers. `date`, `project` and `document_type` are free text kept with every chunk. Everything is optional except that the file must be markdown.
- **`status: placeholder`** keeps a file out of the index entirely. Delete that line (or change it to `status: live`) once the file holds real content. A file whose body is only `TODO:` markers is treated as a placeholder even without the line.
- **Headings** are chunk boundaries. Write one `##` per project, role, paper or topic and the archive will retrieve that topic on its own. Keep one idea per section; long sections are packed into ~220-word chunks with a small overlap.
- **Facts only.** Write what is true and verified. The archive never infers, so anything that is not written here does not exist for it.

## Suggested documents

| File | What it should hold |
|---|---|
| `profile.md` | Who Dhruv is, in his own words. A short story, roles, what he cares about. |
| `experience.md` | Roles, teams, dates, responsibilities, outcomes. One `##` per role. |
| `projects.md` | One `##` per project: problem, what was built, stack, outcome, what it taught. |
| `research.md` | Publications and research: title, venue, year, co-authors, abstract in plain words, contribution. |
| `education.md` | Degrees, institutions, dates, notable coursework or theses. |
| `achievements.md` | Awards, competitions, recognitions, with dates and context. |
| `skills.md` | Languages, frameworks, tools, and how they are used together. |
| `interests.md` | What Dhruv is curious about beyond work; what he is exploring now. |

Add more files freely (`talks.md`, `writing.md`, …). Every `.md` file except this README is loaded.

## Test data

`_test_data/` holds obviously fictional material tagged `TEST_DATA_ONLY`, used only to exercise retrieval in tests. It is never loaded unless `INCLUDE_TEST_DATA=true`. Never put real information there, and never put fictional information anywhere else.

## After editing

The index rebuilds itself on the next app start (it fingerprints the folder). To rebuild explicitly or to see what would be indexed:

```bash
python ingest.py --check
python ingest.py
```
