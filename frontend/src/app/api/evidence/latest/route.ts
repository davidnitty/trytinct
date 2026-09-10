// GET /api/evidence/latest — the dashboard's data endpoint.
// Returns the view model built server-side from a REAL evidence bundle
// (signature verified AND issuer pinned), or an explicit not_found — never a
// silent fallback. Mock data is served only when explicitly requested
// (?mock=1), which the dashboard surfaces behind a MOCK chip.
// A signature that verifies but was signed by an unpinned key still renders,
// behind a "MATH VALID, BUT UNTRUSTED ISSUER" warning.
import {
  bundleToDashboardData,
  mockToDashboardData,
} from "@/lib/tinct/dashboardData"
import {
  findBundleByName,
  findLatestBundle,
  verifyBundle,
} from "@/lib/tinct/evidence"

export const dynamic = "force-dynamic"

export async function GET(request: Request) {
  const url = new URL(request.url)
  if (url.searchParams.get("mock") === "1") {
    return Response.json({ status: "ok", source: "mock", data: mockToDashboardData() })
  }

  const run = url.searchParams.get("run")
  const bundle = run ? await findBundleByName(run) : await findLatestBundle()
  if (!bundle) {
    return Response.json(
      {
        status: "not_found",
        message: run
          ? `No evidence bundle named "${run}" found.`
          : "No evidence bundle found. Run `tinct ship` or `tinct certify` first — or view the demo data.",
      },
      { status: 404 },
    )
  }

  const verification = await verifyBundle(bundle.rawLit)
  if (!verification.signatureValid) {
    return Response.json(
      {
        status: "invalid",
        message:
          "An evidence bundle exists but its Ed25519 signature failed verification. It may have been tampered with — refusing to display it.",
      },
      { status: 409 },
    )
  }

  return Response.json({
    status: "ok",
    source: "live",
    data: bundleToDashboardData(bundle.name, bundle.raw, verification),
  })
}
