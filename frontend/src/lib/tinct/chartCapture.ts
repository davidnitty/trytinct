// Chart capture for the PDF export: snapshot the live Recharts DOM node to a
// PNG data URL. Client-only (dynamic import keeps html-to-image out of the
// server render path), and fail-safe: a failed capture returns null so the
// PDF falls back to tables instead of failing the whole export.
export async function captureChartPng(
  node: HTMLElement | null,
  pixelRatio = 2,
): Promise<string | null> {
  if (!node) return null
  try {
    const { toPng } = await import("html-to-image")
    // Dark background on purpose: the chart is styled for the dark dashboard
    // (gradient fills, light axis text), and the PDF page is white — capturing
    // the card's own background keeps the colors readable in the artifact.
    return await toPng(node, { backgroundColor: "#09090b", pixelRatio })
  } catch (exc) {
    console.warn("[tinct] chart capture failed; PDF omits the chart image", exc)
    return null
  }
}

/** Wait for Recharts to finish layout/animation before snapshotting. */
export async function nextPaint(frames = 2, extraMs = 400): Promise<void> {
  for (let i = 0; i < frames; i++) {
    await new Promise((resolve) => requestAnimationFrame(resolve))
  }
  await new Promise((resolve) => setTimeout(resolve, extraMs))
}
