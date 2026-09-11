# tinct dashboard

The visualization layer for [tinct](../README.md) — it reads signed certification
evidence bundles from disk and renders verdicts, safety gates, MoE routing, and
expert-offloading telemetry. No data leaves the machine: the Next.js server reads
`.tinct/evidence/*.json` locally and serves a view model to the browser.

```bash
npm install
npm run dev     # http://localhost:3000
```

## How data flows

```
.tinct/evidence/<run>_evidence.json
        │
        ▼
/api/evidence/latest     ← verifies Ed25519 signature + issuer trust, maps to view model
/api/evidence            ← downloads the raw bundle (only if verified AND trusted)
        │
        ▼
/dashboard               ← client fetch, skeleton, explicit empty/error states
```

Evidence discovery order: `TINCT_EVIDENCE_DIR`, then `.tinct/evidence` in the app
directory or any parent, then sibling project directories one level up. The newest
bundle (by mtime) wins; `?run=<name>` selects a specific one.

## Trust model

Verifying a signature against the key *embedded in the bundle* only proves the file
wasn't modified after signing — it says nothing about **who** signed it. An attacker
can tamper with a verdict and re-sign with their own key; the math verifies.

`trusted_keys.json` pins acceptable issuers. Bundles from unpinned keys render with an
amber **UNTRUSTED ISSUER** chip and their download is refused with `403`.

```json
{
  "my_project": "0f46d497357efa8c380d4806b0c104e33e184ddf19f57a612ab3e994ba56aa27"
}
```

Pin encodings accepted (all normalized to a SHA-256 fingerprint of the raw 32-byte key):

| Format | Example |
|---|---|
| raw hex | `0f46d497...56aa27` |
| base64 SPKI DER | `MCowBQYDK2VwAyEAD0bUlzV++ow4...` |
| ssh-ed25519 line | `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5...` |
| PEM block | `-----BEGIN PUBLIC KEY-----...` |

For production, prefer the `TRUSTED_PUBLIC_KEYS` environment variable
(comma-separated, set at deploy time) over the JSON file — a runtime-writable trust
file is itself an attack surface.

Status codes: `200` verified · `404` no bundle · `403` untrusted issuer · `409`
signature invalid (tampered).

## Demo walkthrough

Generate real signed bundles with tinct's own signer (requires the repo venv; writes
to `demo-cert/.tinct/`, which is git-ignored):

```bash
# The happy path: a signed SHIP verdict from a pinned issuer
python demo-cert/make_bundle.py

# Forensic view: DON'T SHIP with a toxicity spike and starved experts
python demo-cert/make_bundle.py --failing
# then open http://localhost:3000/dashboard?run=cert_20260904_160000_failing

# Security demo: a FORGED SHIP verdict signed by an unpinned key.
# The dashboard verifies the math, then flags UNTRUSTED ISSUER and
# refuses the download.
python demo-cert/make_bundle.py --rogue

# Make the trusted SHIP bundle the newest one again
python demo-cert/make_bundle.py --restore
```

Mock scenarios need no bundle on disk — they are always served behind a **MOCK DATA**
chip and never silently substitute for real evidence:

- `/dashboard?mock=1` — passing run
- `/dashboard?mock=fail` — failing run (red banner, failed gates first, red bars)

## Stack

Next.js (App Router) · TypeScript · Tailwind CSS v4 · shadcn/ui · Recharts ·
Node `crypto` for Ed25519 verification (server-side only).
