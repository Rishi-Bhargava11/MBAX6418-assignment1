# Uploading this project to GitHub (and making it public)

You already have `git`, the `gh` (GitHub) CLI, and an **authenticated
GitHub account** (`Rishi-Bhargava11`). This walks through everything in order.

## 1. Check what will be committed

`.gitignore` already keeps out the things that shouldn't go up:
- `.env` — contains the course API key.
- `data/Gift_Cards.jsonl.gz` — large (12 MB) and re-downloadable,
  so it does **not** belong in the repo (re-download it from
  `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz`).
- Python caches (`.venv/`, `__pycache__/`).

The NRC-Emotion-Lexicon word list **is** committed so the word-list script
reproduces without extra steps.

## 2. Initialize and commit (run from inside this folder)

```bash
cd "C:/Users/rishb/Grad School/Business Solutions - 6418/Assignment 1"
git init
git add .
git commit -m "MBAX 6418 Assignment 1: sentiment & emotion classification"
```

## 3. Create a PUBLIC GitHub repo and push

The `gh` CLI is already logged in, so this is one command:

```bash
gh repo create MBAX6418-assignment1 --public --source . --remote origin --push
```

What it does: creates a **public** repo named `MBAX6418-assignment1` under your
account, wires it as the `origin` remote, and pushes your commit.

(Skip straight to push if the repo already exists:

```bash
git remote add origin https://github.com/Rishi-Bhargava11/MBAX6418-assignment1.git
git push -u origin main
```
)

## 4. Grab the shareable link

```bash
gh repo view --web          # opens it in your browser
```
The link looks like: `https://github.com/Rishi-Bhargava11/MBAX6418-assignment1`

Copy that URL into the **Canvas** submission field.

## 5. Double-check before you submit

Open the link (or a private/incognito window, since public) and confirm:
- The `README.md` renders with the report + screenshots.
- The code (`.py` files, `.txt` word list) is present.
- No `.env` or API key leaked into the repo (check the file list).
- Anyone can read it (it's public).

## If you ever made the repo private instead
Private works too, but you must add the instructor as a collaborator so they
can see it:
```bash
gh api -X PUT repos/Rishi-Bhargava11/MBAX6418-assignment1/collaborators/david.dobolyi@colorado.edu --silent
gh api repos/Rishi-Bhargava11/MBAX6418-assignment1/collaborators/david.dobolyi@colorado.edu
# second call with -f permission=push if you want maintainer rights
```
(A single line: the first adds, the second checks. Either way, after running,
the instructor can open the repo.)

---
**Every person submits their own repo** — this is your individual copy, even if
you team up with classmates.
