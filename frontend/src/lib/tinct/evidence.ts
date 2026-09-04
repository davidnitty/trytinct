// Server-side tinct evidence access: discovery, canonical-byte replication,
// and Ed25519 verification. This module must only ever run on the server —
// bundle verification never ships to the client.
//
// The canonical-byte logic mirrors src/tinct/security/evidence.py exactly:
//   json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
// over the 13 unsigned fields (everything except `signature`), with the same
// from_dict defaults for bundles written by older tinct versions.
import { createPublicKey, verify as verifySignature } from "node:crypto"
import { readFile, readdir, stat } from "node:fs/promises"
import path from "node:path"

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface EvidenceBundle {
  name: string // file stem, e.g. "cert_20260904_143022"
  path: string
  /** Plain parse — real JS numbers, for view-model mapping. */
  raw: Record<string, any>
  /** Literal-preserving parse — number tokens kept verbatim, for signing. */
  rawLit: Record<string, any>
}

/**
 * Parse JSON while keeping every number's original literal text (Node 22+
 * JSON.parse source-text access). Required because Python's json module
 * distinguishes 0.0 from 0 when serializing and JS does not — the canonical
 * bytes must agree byte-for-byte with what the signer hashed.
 */
function parseKeepingNumberLiterals(text: string): Record<string, any> {
  // The 3-arg reviver (with source text) is runtime-supported since Node 22
  // but not yet in TS's DOM lib types, hence the cast.
  const reviver = (_key: string, value: any, ctx: { source: string }) => {
    if (typeof value === "number") return { __num: ctx.source }
    return value
  }
  return JSON.parse(
    text,
    reviver as unknown as (this: any, key: string, value: any) => any,
  )
}

/**
 * Emit a number literal the way Python's json.dumps would:
 * int literals pass through; float literals get repr() semantics — integral
 * values keep a trailing ".0" and exponents are zero-padded to two digits.
 */
function pythonNumberLiteral(literal: string): string {
  if (!/[.eE]/.test(literal)) return literal // int literal
  const num = Number(literal)
  if (Number.isInteger(num)) return `${num}.0`
  return String(num).replace(/e([+-])(\d)$/, "e$10$2")
}

/** Deep key-sorted compact JSON — byte-compatible with Python's sort_keys dump. */
function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`
  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  if (keys.length === 1 && keys[0] === "__num" && typeof record.__num === "string") {
    return pythonNumberLiteral(record.__num)
  }
  const sorted = keys.sort()
  return `{${sorted
    .map((k) => `${JSON.stringify(k)}:${canonicalJson(record[k])}`)
    .join(",")}}`
}

function canonicalBytes(rawLit: Record<string, any>): Buffer {
  const payload = {
    project_name: rawLit.project_name,
    model: rawLit.model,
    family: rawLit.family,
    decision: rawLit.decision,
    created_at: rawLit.created_at ?? "",
    artifacts: rawLit.artifacts ?? {},
    data_report: rawLit.data_report ?? {},
    eval_report: rawLit.eval_report ?? {},
    metrics: rawLit.metrics ?? {},
    config: rawLit.config ?? {},
    safety_gates: rawLit.safety_gates ?? {},
    training_tool: rawLit.training_tool ?? "tinct",
    training_executed: rawLit.training_executed ?? true,
  }
  return Buffer.from(canonicalJson(payload), "utf-8")
}

/** Verify the bundle's embedded Ed25519 signature over its canonical bytes. */
export function verifyBundle(rawLit: Record<string, any>): boolean {
  const sig = rawLit.signature
  if (!sig?.public_key_pem || !sig?.value || sig?.alg !== "ed25519") return false
  try {
    const publicKey = createPublicKey(sig.public_key_pem)
    const signature = Buffer.from(sig.value, "hex")
    return verifySignature(null, canonicalBytes(rawLit), publicKey, signature)
  } catch {
    return false
  }
}

/**
 * Locate `.tinct/evidence` directories: the TINCT_EVIDENCE_DIR env var wins,
 * then we walk up from the app dir looking for a `.tinct` project, then check
 * sibling project dirs one level up (e.g. a demo project beside `frontend/`).
 */
async function findEvidenceDirs(): Promise<string[]> {
  const candidates: string[] = []
  if (process.env.TINCT_EVIDENCE_DIR) {
    candidates.push(path.resolve(process.env.TINCT_EVIDENCE_DIR))
  }

  let dir = process.cwd()
  for (let depth = 0; depth < 5 && dir !== path.dirname(dir); depth++) {
    candidates.push(path.join(dir, ".tinct", "evidence"))
    dir = path.dirname(dir)
  }

  const siblingsParent = path.dirname(process.cwd())
  try {
    const entries = await readdir(siblingsParent, { withFileTypes: true })
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith(".")) {
        candidates.push(path.join(siblingsParent, entry.name, ".tinct", "evidence"))
      }
    }
  } catch {
    // unreadable parent — env var and walk-up still apply
  }

  const checks = await Promise.all(
    candidates.map(async (candidate) => {
      const isDir = await stat(candidate)
        .then((s) => s.isDirectory())
        .catch(() => false)
      return isDir ? candidate : null
    }),
  )
  return [...new Set(checks.filter((c): c is string => c !== null))]
}

/** Most recently written `*_evidence.json` across all known evidence dirs. */
export async function findLatestBundle(): Promise<EvidenceBundle | null> {
  const dirs = await findEvidenceDirs()
  let latest: (EvidenceBundle & { mtimeMs: number }) | null = null

  for (const dir of dirs) {
    let names: string[] = []
    try {
      names = (await readdir(dir)).filter((n) => n.endsWith("_evidence.json"))
    } catch {
      continue
    }
    for (const name of names) {
      const fullPath = path.join(dir, name)
      const info = await stat(fullPath).catch(() => null)
      if (!info) continue
      if (latest && info.mtimeMs <= latest.mtimeMs) continue
      try {
        const text = await readFile(fullPath, "utf-8")
        latest = {
          name: name.replace(/_evidence\.json$/, ""),
          path: fullPath,
          raw: JSON.parse(text),
          rawLit: parseKeepingNumberLiterals(text),
          mtimeMs: info.mtimeMs,
        }
      } catch {
        // unreadable/corrupt JSON — skip it rather than crashing the API
      }
    }
  }

  return latest
}
