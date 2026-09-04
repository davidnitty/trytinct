// GET /api/evidence/latest — the dashboard's data endpoint.
// Returns the view model built server-side from the latest REAL evidence
// bundle (signature verified), or an explicit not_found — never a silent
// fallback. Mock data is served only when explicitly requested (?mock=1),
// which the dashboard surfaces behind a MOCK chip.
import {
  bundleToDashboardData,
  mockToDashboardData,
} from "@/lib/tinct/dashboardData"
import { findLatestBundle, verifyBundle } from "@/lib/tinct/evidence"

export const dynamic = "force-dynamic"

export async function GET(request: Request) {
  const url = new URL(request.url)
  if (url.searchParams.get("mock") === "1") {
    return Response.json({ status: "ok", source: "mock", data: mockToDashboardData() })
  }

  const bundle = await findLatestBundle()
  if (!bundle) {
    return Response.json(
      {
        status: "not_found",
        message:
          "No evidence bundle found. Run `tinct ship` or `tinct certify` first — or view the demo data.",
      },
      { status: 404 },
    )
  }

  const verified = verifyBundle(bundle.rawLit)
  if (!verified) {
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
    data: bundleToDashboardData(bundle.name, bundle.raw, verified),
  })
}
