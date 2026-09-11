// Trust store for evidence issuers.
//
// Verifying a signature against the key embedded IN the bundle only proves
// the bundle wasn't modified after signing — it says nothing about WHO signed
// it. An attacker can tamper with the verdict and sign with their own key.
// The trust store pins which issuers are acceptable; anything else is
// untrusted even when the math checks out.
//
// Pins come from (union of):
//   1. trusted_keys.json next to the app (cwd):  { "<project>": "<key>" }
//   2. TRUSTED_PUBLIC_KEYS env var: comma-separated keys (hex or ssh-ed25519
//      lines — PEMs are multi-line, so put those in the JSON file).
//
// Keys are normalized to a SHA-256 fingerprint over the raw 32-byte Ed25519
// public key, so a pin can be written as raw hex, a PEM, or an
// "ssh-ed25519 AAAA..." line and still match.
import { createHash, createPublicKey } from "node:crypto"
import { readFile } from "node:fs/promises"
import path from "node:path"

export interface TrustEntry {
  label: string
  fingerprint: string
}

function sha256Hex(bytes: Buffer): string {
  return createHash("sha256").update(bytes).digest("hex")
}

/** SHA-256 over the raw 32-byte Ed25519 public key. */
export function keyFingerprint(publicKeyPem: string): string | null {
  try {
    const der = createPublicKey(publicKeyPem).export({ format: "der", type: "spki" })
    const raw = der.subarray(der.length - 32) // SPKI trailing 32 bytes = raw key
    return sha256Hex(raw)
  } catch {
    return null
  }
}

/**
 * Accepts every common Ed25519 public-key encoding:
 *   1. raw 64-char hex
 *   2. an "ssh-ed25519 AAAA..." line
 *   3. a PEM block
 *   4. base64-encoded SPKI DER  ("MCowBQYDK2VwAyEA..." — openssl -outform DER | base64)
 */
function pinToFingerprint(pin: string): string | null {
  const s = pin.trim()

  if (/^[0-9a-fA-F]{64}$/.test(s)) {
    return sha256Hex(Buffer.from(s, "hex"))
  }

  if (s.startsWith("ssh-ed25519 ")) {
    try {
      const blob = Buffer.from(s.slice("ssh-ed25519 ".length).trim(), "base64")
      if (blob.length < 32) return null
      return sha256Hex(blob.subarray(blob.length - 32))
    } catch {
      return null
    }
  }

  if (s.startsWith("-----BEGIN")) {
    return keyFingerprint(s)
  }

  // base64 SPKI DER: Ed25519 SPKI is exactly 44 bytes
  if (/^[A-Za-z0-9+/]+={0,2}$/.test(s) && s.length >= 44) {
    try {
      const der = Buffer.from(s, "base64")
      if (der.length >= 44) return sha256Hex(der.subarray(der.length - 32))
    } catch {
      return null
    }
  }

  return null
}

async function loadPins(): Promise<TrustEntry[]> {
  const entries: TrustEntry[] = []

  // 1. trusted_keys.json next to the app root.
  try {
    const file = path.join(process.cwd(), "trusted_keys.json")
    const parsed = JSON.parse(await readFile(file, "utf-8")) as Record<string, string>
    for (const [label, pin] of Object.entries(parsed)) {
      const fp = pinToFingerprint(pin)
      if (fp) entries.push({ label, fingerprint: fp })
    }
  } catch {
    // no trust file — env var may still provide pins
  }

  // 2. TRUSTED_PUBLIC_KEYS env var (comma-separated).
  const env = process.env.TRUSTED_PUBLIC_KEYS
  if (env) {
    for (const pin of env.split(",")) {
      if (!pin.trim()) continue
      const fp = pinToFingerprint(pin)
      if (fp) entries.push({ label: `env:${fp.slice(0, 12)}`, fingerprint: fp })
    }
  }

  return entries
}

export interface KeyTrust {
  trusted: boolean
  /** Which pin matched, e.g. the project name from trusted_keys.json. */
  label?: string
  /** SHA-256 fingerprint of the bundle's embedded public key. */
  fingerprint: string | null
}

/** Classify a bundle's embedded public key against the trust store. */
export async function classifyKey(publicKeyPem: string): Promise<KeyTrust> {
  const fingerprint = keyFingerprint(publicKeyPem)
  if (!fingerprint) return { trusted: false, fingerprint: null }
  const pins = await loadPins()
  const match = pins.find((p) => p.fingerprint === fingerprint)
  return { trusted: match !== undefined, label: match?.label, fingerprint }
}
