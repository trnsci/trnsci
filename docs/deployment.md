# Deployment runbook

This document describes how the documentation site is built and served, and how to set up the custom `trnsci.dev` domain.

## How the site is built

- `main` branch push triggers `.github/workflows/docs.yml`.
- That workflow installs `mkdocs-material` + `mkdocs-monorepo-plugin` and runs `mkdocs gh-deploy --force`, which publishes the built site to the `gh-pages` branch.
- GitHub Pages serves the `gh-pages` branch at the Pages URL.

## Default URL

Until DNS is pointed at GitHub Pages, the site is live at:

```
https://trnsci.github.io/trnsci/
```

## Custom domain: trnsci.dev

The `docs/CNAME` file in the repository pins the custom domain. GitHub Pages reads this at build time and configures the Pages site to expect `trnsci.dev` as the canonical domain.

### DNS records (registrar side)

At the `trnsci.dev` registrar, add the following records. These point the apex and `www` at GitHub Pages.

Apex A records:

| Type | Name | Value |
|---|---|---|
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |

Apex AAAA records (IPv6):

| Type | Name | Value |
|---|---|---|
| AAAA | @ | 2606:50c0:8000::153 |
| AAAA | @ | 2606:50c0:8001::153 |
| AAAA | @ | 2606:50c0:8002::153 |
| AAAA | @ | 2606:50c0:8003::153 |

Subdomain CNAME:

| Type | Name | Value |
|---|---|---|
| CNAME | www | trnsci.github.io |

Propagation usually takes minutes to a couple of hours.

### After DNS propagates

1. Go to **Settings → Pages** on the umbrella repo.
2. Under "Custom domain", verify `trnsci.dev` is set.
3. Wait until the "DNS check successful" indicator appears.
4. Enable **Enforce HTTPS**. Certificate provisioning via Let's Encrypt takes a few minutes.

## Troubleshooting

- **Pages build fails** — check `Actions → Deploy docs`. Most failures are mkdocs nav references to missing files or broken monorepo-plugin includes.
- **Custom domain shows a 404** — confirm `docs/CNAME` exists on the `main` branch *and* on the `gh-pages` branch (the deploy copies it over). If missing, re-run the workflow.
- **HTTPS toggle greyed out** — DNS hasn't propagated yet. Wait and check again.
- **Sub-project docs missing** — the monorepo plugin requires each `trn*/mkdocs.yml` to be present and parseable at docs-build time. Verify each sub-project has a valid `mkdocs.yml`.

## Local preview

```bash
pip install mkdocs-material mkdocs-monorepo-plugin
mkdocs serve
```

Opens at `http://127.0.0.1:8000/`.
