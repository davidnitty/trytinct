// GET /api/evidence — download the latest signed evidence bundle.
// Fail-closed twice over: the Ed25519 signature must verify, AND the issuing
// key must be pinned in the trust store. Tampered → 409; valid math but
// untrusted issuer → 403.
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

  const verification = await verifyBundle(bundle.rawLit)
  if (!verification.signatureValid) {
    return Response.json(
      { error: "signature verification failed — bundle may be tampered with" },
      { status: 409 },
    )
  }
  if (!verification.trusted) {
    return Response.json(
      {
        error:
          "issuer key is not in the trust store — refusing to serve (signature was mathematically valid, but the signer is unknown)",
      },
      { status: 403 },
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
