// GET /api/evidence/list — metadata for every discovered evidence bundle,
// newest first, for the dashboard's run selector.
//
// Design notes:
//   - Bundles are VERIFIED during listing: a tampered bundle must not present
//     itself as ordinary in the dropdown. Tampered bundles are listed, not
//     hidden — a tampering attempt is an operational signal. Opening one still
//     returns 409 from /api/evidence/latest.
//   - Empty directory → 200 {"runs": []}. "Zero runs" is a valid answer to a
//     list question; "no latest" is an error for /latest. Different questions.
import { listBundles } from "@/lib/tinct/evidence"

export const dynamic = "force-dynamic"

export async function GET() {
  const runs = await listBundles()
  return Response.json({ runs })
}
