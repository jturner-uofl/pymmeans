# Repo-polish follow-ups — user-actionable items

Most of the "polish AF" work is now in the repo (README, CITATION.cff,
CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, issue/PR templates). A few
items can only be done from the GitHub UI or an external service and
are listed below.

## 1. GitHub UI settings (5 minutes)

On <https://github.com/jturner-uofl/pymmeans> click the gear icon
next to **About** in the right sidebar and set:

| Field | Suggested value |
|---|---|
| Description | `Estimated marginal means for Python. Native implementation of R emmeans + pbkrtest, integrated with conformal prediction, AIPW + DML, E-value sensitivity, and Rubin's-rules MI pooling.` |
| Website | `https://jturner-uofl.github.io/pymmeans/` (after Pages is enabled — see §3) |
| Topics | `statistics`, `python`, `emmeans`, `least-squares-means`, `marginal-means`, `pbkrtest`, `kenward-roger`, `satterthwaite`, `mixed-models`, `multiple-comparisons`, `contrasts`, `causal-inference`, `conformal-prediction`, `double-machine-learning`, `doubly-robust`, `aipw`, `multiple-imputation`, `e-value`, `sensitivity-analysis`, `r-emmeans-port` |

Also check:

- [ ] **Discussions** enabled (Settings → Features → Discussions)
- [ ] **Sponsorships** disabled (or enabled if you want it — your call)
- [ ] **Wiki** disabled (we have a docs site instead)
- [ ] **Issues** enabled (default)
- [ ] **Releases** publicly visible

## 2. Zenodo integration for citable software DOIs

This gives you a citable DOI for every GitHub release — required for
JSS submission and useful regardless.

1. Go to <https://zenodo.org/account/settings/github/>
2. Sign in with your GitHub account
3. Find `jturner-uofl/pymmeans` in the repository list
4. Flip the toggle to **on**
5. Go back to GitHub → **Releases** → **Draft a new release**
6. Tag = `v0.6.0` (it already exists, so Zenodo will pick it up
   immediately)
7. Title = `pymmeans 0.6.0`
8. Description = paste the v0.6.0 section from `CHANGELOG.md`
9. Click **Publish release**
10. Within ~10 minutes, Zenodo will email you a DOI badge like
    `10.5281/zenodo.NNNNNNN`

Then update `README.md` to include the Zenodo badge in the badges
block at the top:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.NNNNNNN.svg)](https://doi.org/10.5281/zenodo.NNNNNNN)
```

And update `CITATION.cff` with the resolved DOI so the GitHub
"Cite this repository" button surfaces it.

## 3. GitHub Pages for the documentation site

The mkdocs site is already built and committed under `site/`. To host
it at `jturner-uofl.github.io/pymmeans`:

### Option A (recommended) — GitHub Actions

Create `.github/workflows/docs.yml`:

```yaml
name: docs
on:
  push:
    branches: [main]
    paths: [docs/**, mkdocs.yml]
permissions:
  contents: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install mkdocs-material mkdocstrings[python]
      - run: mkdocs gh-deploy --force
```

Push this file; the workflow will deploy to a `gh-pages` branch
automatically. In Settings → Pages, set the source to **Deploy from
a branch** → `gh-pages` / `/ (root)`.

### Option B — manual deploy

```bash
mkdocs gh-deploy
```

(Requires `mkdocs` installed locally; pushes to a `gh-pages` branch.)

After either option, the site is live at
`https://jturner-uofl.github.io/pymmeans/`. Update the GitHub
repository **Website** field (§1) to point there.

## 4. Optional: conda-forge distribution

For broader reach (especially in the data-science / scientific-Python
community who prefer conda over pip), submit `pymmeans` to
conda-forge.

1. Fork <https://github.com/conda-forge/staged-recipes>
2. Add `recipes/pymmeans/meta.yaml` modeled on the staged-recipes
   examples
3. Open a PR to staged-recipes
4. After acceptance (~1–2 week review), `pymmeans` is installable via
   `conda install -c conda-forge pymmeans`

This is genuinely useful for the JSS-reviewer signal of "this package
is in distribution channels users actually pull from."

## 5. Optional: PyPI README rendering

PyPI sometimes renders Markdown READMEs poorly with HTML embedded.
Spot-check the v0.6.0 release at
<https://pypi.org/project/pymmeans/> after the next PyPI upload.
If the badges or any HTML look broken, consider stripping the
`<p align="center">` wrappers — pure Markdown renders most reliably.

## 6. Optional: Twitter / Bluesky / Mastodon announcement

A single short post once the Zenodo DOI is live:

> Released `pymmeans` 0.6.0 — a Python implementation of R `emmeans`
> + `pbkrtest` with conformal prediction, doubly-robust AIPW, and
> double-machine-learning ATE estimators. 1,037 tests, 87% coverage,
> 251 validation contracts. Faster than R `emmeans` on common paths.
> Repo: <github.com/jturner-uofl/pymmeans>
> Cite: <doi.org/10.5281/zenodo.NNNNNNN>

Helpful for adoption signal but entirely optional. Don't overthink it.

---

## Done summary (so far)

- [x] README rewritten
- [x] CITATION.cff refreshed for v0.6.0
- [x] CHANGELOG updated with v0.2.0 → v0.6.0 entries
- [x] CONTRIBUTING.md
- [x] CODE_OF_CONDUCT.md (Contributor Covenant 2.1)
- [x] Issue + PR templates
- [x] Makefile (replication wrapper, already shipped in v0.6.0)

## To do (user-actionable)

- [ ] GitHub UI: set Description / Website / Topics (§1)
- [ ] Enable Zenodo integration + publish v0.6.0 release on GitHub (§2)
- [ ] Add the Zenodo DOI to README + CITATION.cff once issued
- [ ] Deploy mkdocs to GitHub Pages (§3)
- [ ] (Optional) conda-forge submission (§4)
