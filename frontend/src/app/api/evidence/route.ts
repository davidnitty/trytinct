// GET /api/evidence — download the latest signed evidence bundle.
// Fail-closed: the bundle's Ed25519 signature is verified server-side before
// it is served; a tampered bundle is never downloadable from the dashboard.
import { readFile } from "node:fs/promises"

import { findLatestBundle, verifyBundle } from "@/lib/tinct/evidence"

export const dynamic = "force-dynamic"

export async function GET() {
  const bundle = await findLatestBundle()
  if (!bundle) {
    return Response.json(
      { error: "no evidence bundle found" },
      { status: 404 },
    )
  }
  if (!verifyBundle(bundle.rawLit)) {
    return Response.json(
      { error: "signature verification failed — bundle may be tampered with" },
      { status: 409 },
    )
  }

  const bytes = await readFile(bundle.path)
  return new Response(bytes, {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="evidence_${bundle.name}.json"`,
    },
  })
}
