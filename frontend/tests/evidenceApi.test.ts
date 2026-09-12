// Evidence API behaviour, exercised through the real route handlers with the
// real Ed25519 verifier against temp evidence dirs (hermetic via
// TINCT_EVIDENCE_DIR). Fixtures are genuinely signed bundles, written to disk
// VERBATIM — re-serializing with plain JSON.stringify would collapse 0.0 to 0
// and corrupt the signatures under test.
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { afterAll, beforeEach, describe, expect, it } from "vitest"

import {
  parseKeepingNumberLiterals,
  serializeCanonicalJson,
} from "@/lib/tinct/evidence"
import { GET as downloadGET } from "@/app/api/evidence/route"
import { GET as latestGET } from "@/app/api/evidence/latest/route"
import { GET as listGET } from "@/app/api/evidence/list/route"

/* eslint-disable @typescript-eslint/no-explicit-any */

const FIXTURES = path.join(__dirname, "fixtures")
const OLDER_HEX = "6ed034042f2846a6ad6cd4c0b660fc67d94c36d0c8aabe9dc7bf6e7b180973c1"
const NEWER_HEX = "1721de20b79f978fa08fa846c4c0388960b980ddc4a761bdf61c83bb3c5c2d75"

let evidenceDir: string

async function fixtureText(name: string): Promise<string> {
  return readFile(path.join(FIXTURES, name), "utf-8")
}

async function writeBundle(runId: string, text: string): Promise<void> {
  await writeFile(path.join(evidenceDir, `${runId}_evidence.json`), text, "utf-8")
}

/** Tamper with a signed field while keeping the (now invalid) signature. */
function tamperedText(text: string): string {
  return text.replace('"decision": "SHIP"', '"decision": "DON\'T_SHIP"')
}

/** Strip the signature block entirely, preserving number literals. */
function unsignedText(text: string): string {
  const parsed = parseKeepingNumberLiterals(text)
  delete parsed.signature
  return serializeCanonicalJson(parsed)
}

async function listRuns(): Promise<any[]> {
  const response = await listGET()
  return (await response.json()).runs
}

beforeEach(async () => {
  evidenceDir = await mkdtemp(path.join(os.tmpdir(), "tinct-evidence-"))
  process.env.TINCT_EVIDENCE_DIR = evidenceDir
  delete process.env.TRUSTED_PUBLIC_KEYS
})

afterAll(async () => {
  delete process.env.TINCT_EVIDENCE_DIR
  delete process.env.TRUSTED_PUBLIC_KEYS
  if (evidenceDir) await rm(evidenceDir, { recursive: true, force: true })
})

describe("GET /api/evidence/list", () => {
  it("returns 200 with an empty array when the store is empty", async () => {
    const response = await listGET()
    expect(response.status).toBe(200)
    expect(await response.json()).toEqual({ runs: [] })
  })

  it("lists two bundles in descending timestamp order", async () => {
    await writeBundle("cert_older", await fixtureText("signed_bundle.json"))
    await writeBundle("cert_newer", await fixtureText("signed_bundle_newer.json"))

    const runs = await listRuns()
    expect(runs.map((r: any) => r.run_id)).toEqual(["cert_newer", "cert_older"])
    expect(runs.map((r: any) => r.integrity)).toEqual(["verified", "verified"])
  })

  it("lists a tampered bundle as tampered (not hidden) and still 200s", async () => {
    await writeBundle("cert_tampered", tamperedText(await fixtureText("signed_bundle.json")))

    const response = await listGET()
    expect(response.status).toBe(200)
    const runs = await response.json().then((body: any) => body.runs)
    expect(runs).toHaveLength(1)
    expect(runs[0].integrity).toBe("tampered")
  })

  it("lists an unsigned bundle as unsigned", async () => {
    await writeBundle("cert_unsigned", unsignedText(await fixtureText("signed_bundle.json")))

    const runs = await listRuns()
    expect(runs[0].integrity).toBe("unsigned")
    expect(runs[0].trusted).toBe(false)
  })

  it("reports an unpinned issuer as untrusted, and pinned as trusted", async () => {
    await writeBundle("cert_fixture", await fixtureText("signed_bundle.json"))

    const untrusted = await listRuns()
    expect(untrusted[0].trusted).toBe(false)

    process.env.TRUSTED_PUBLIC_KEYS = OLDER_HEX
    const trusted = await listRuns()
    expect(trusted[0].trusted).toBe(true)
  })
})

describe("GET /api/evidence/latest", () => {
  it("returns 404 when there is no bundle", async () => {
    const response = await latestGET(new Request("http://test/api/evidence/latest"))
    expect(response.status).toBe(404)
    expect((await response.json()).status).toBe("not_found")
  })

  it("returns 409 when the newest bundle is tampered", async () => {
    await writeBundle("cert_tampered", tamperedText(await fixtureText("signed_bundle.json")))

    const runs = await listRuns()
    expect(runs[0].integrity).toBe("tampered")

    const response = await latestGET(new Request("http://test/api/evidence/latest"))
    expect(response.status).toBe(409)
    expect((await response.json()).status).toBe("invalid")
  })

  it("returns 404 for an unknown ?run= without throwing", async () => {
    await writeBundle("cert_known", await fixtureText("signed_bundle.json"))

    const response = await latestGET(
      new Request("http://test/api/evidence/latest?run=cert_does_not_exist"),
    )
    expect(response.status).toBe(404)
    const body = await response.json()
    expect(body.status).toBe("not_found")
    expect(body.message).toContain("cert_does_not_exist")
  })

  it("serves the pinned, verified bundle with its trust state", async () => {
    process.env.TRUSTED_PUBLIC_KEYS = OLDER_HEX
    await writeBundle("cert_fixture", await fixtureText("signed_bundle.json"))

    const response = await latestGET(new Request("http://test/api/evidence/latest"))
    expect(response.status).toBe(200)
    const body = await response.json()
    expect(body.source).toBe("live")
    expect(body.data.verified).toBe(true)
    expect(body.data.trusted).toBe(true)
    expect(body.data.keyFingerprint).toHaveLength(64)
  })
})

describe("GET /api/evidence (download)", () => {
  it("403s a mathematically valid bundle from an unpinned issuer", async () => {
    await writeBundle("cert_fixture", await fixtureText("signed_bundle.json"))

    const response = await downloadGET()
    expect(response.status).toBe(403)
    expect((await response.json()).error).toContain("trust store")
  })

  it("serves the bundle as an attachment once the issuer is pinned", async () => {
    process.env.TRUSTED_PUBLIC_KEYS = OLDER_HEX
    await writeBundle("cert_fixture", await fixtureText("signed_bundle.json"))

    const response = await downloadGET()
    expect(response.status).toBe(200)
    expect(response.headers.get("content-disposition")).toBe(
      'attachment; filename="evidence_cert_fixture.json"',
    )
  })
})
