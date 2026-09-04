import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  ShieldCheck,
  Ghost,
  MessageSquareOff,
  Biohazard,
  BrainCircuit,
  HardDrive,
  TerminalSquare,
  ArrowRight
} from "lucide-react"

// The Github brand icon was removed from lucide-react (brand icons dropped),
// so the mark is inlined here (Octicon path, MIT).
function GithubMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.27-.01-1.87-.03-3.44-3.2.7-3.87-1.36-3.87-1.36-.48-1.22-1.17-1.55-1.17-1.55-.96-.65.07-.64.07-.64 1.06.07 1.62 1.09 1.62 1.09.94 1.62 2.47 1.15 3.08.88.1-.68.37-1.15.67-1.41-2.67-.3-5.48-1.34-5.48-5.96 0-1.32.47-2.39 1.24-3.24-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6.01 0c2.29-1.55 3.29-1.23 3.29-1.23.66 1.65.24 2.87.24 3.17.77.85 1.24 1.92 1.24 3.24 0 4.63-2.82 5.66-5.5 5.95.43.37.81 1.1.81 2.22 0 1.6-.01 2.89-.01 3.28 0 .29.19.62.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  )
}

export default function Home() {
  return (
    <main className="min-h-screen bg-black text-white antialiased selection:bg-emerald-500/30">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/10 bg-black/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center font-bold text-black">
              t
            </div>
            <span className="font-semibold text-xl tracking-tight">tinct</span>
            <Badge variant="outline" className="ml-2 text-emerald-400 border-emerald-400/50">v1.1</Badge>
          </div>
          <div className="flex items-center gap-4">
            <Link href="#dashboard" className="text-sm text-gray-400 hover:text-white transition">Dashboard</Link>
            <Link href="https://github.com/davidnitty/trytinct" target="_blank">
              <Button variant="ghost" size="sm" className="gap-2">
                <GithubMark className="w-4 h-4" /> GitHub
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20">
              Open Source &amp; Cryptographically Signed
            </Badge>
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1]">
              Fine-tune AI.<br/>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200">Prove it&apos;s safe.</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-lg leading-relaxed">
              Most training tools just train your model and hope it behaves. <span className="text-white font-medium">tinct</span> is a fail-closed certification engine that automatically tests for toxicity, data leaks, and expert collapse before you ship.
            </p>
            <div className="flex flex-wrap gap-4">
              <Button size="lg" className="bg-emerald-500 hover:bg-emerald-600 text-black font-semibold gap-2">
                Get Started <ArrowRight className="w-4 h-4" />
              </Button>
              <Button size="lg" variant="outline" className="border-white/20 bg-transparent text-white hover:bg-white/10">
                View Demo Dashboard
              </Button>
            </div>
          </div>

          {/* CLI Terminal Visual */}
          <div className="relative">
            <div className="absolute -inset-4 bg-emerald-500/20 blur-3xl rounded-full opacity-30"></div>
            <div className="relative bg-zinc-950 border border-white/10 rounded-xl shadow-2xl overflow-hidden">
              <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10 bg-zinc-900/50">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="ml-2 text-xs text-gray-500 font-mono">tinct certify</span>
              </div>
              <div className="p-6 font-mono text-sm space-y-3 text-gray-300">
                <p><span className="text-emerald-400">$</span> tinct certify --adapter ./my_model --offload-experts</p>
                <p className="text-gray-500">[tinct] Validating adapter structure... <span className="text-emerald-400">peft-lora</span></p>
                <p className="text-gray-500">[tinct] MoEStreamer ready: 8 experts offloaded, 2 resident.</p>
                <p className="text-gray-500">[tinct] Running 6 safety gates...</p>
                <p className="text-gray-500 pl-4">✓ Canary Leakage: <span className="text-emerald-400">PASS</span> (0.0%)</p>
                <p className="text-gray-500 pl-4">✓ Refusal Regression: <span className="text-emerald-400">PASS</span> (-16%)</p>
                <p className="text-gray-500 pl-4">✓ Toxicity: <span className="text-emerald-400">PASS</span></p>
                <p className="text-gray-500 pl-4">✓ Expert Collapse: <span className="text-emerald-400">PASS</span></p>
                <p className="text-gray-500 pl-4">✓ Routing Regression: <span className="text-emerald-400">PASS</span></p>
                <p className="text-gray-500">[tinct] Signing evidence bundle...</p>
                <p className="text-2xl font-bold text-emerald-400 pt-2">VERDICT: SHIP 🚢</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* The 6 Safety Gates Section */}
      <section className="py-20 px-6 border-t border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 space-y-4">
            <h2 className="text-3xl md:text-5xl font-bold tracking-tight">The 6-Gate Certification Suite</h2>
            <p className="text-gray-400 max-w-2xl mx-auto text-lg">
              Every model that passes through tinct is subjected to a rigorous, fail-closed behavioral evaluation.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { icon: Ghost, title: "Canary Leakage", desc: "Detects if the model memorized and regurgitates private data or secrets from the training set." },
              { icon: MessageSquareOff, title: "Refusal Regression", desc: "Ensures training didn't strip the model's ability to say 'no' to harmful or unsafe prompts." },
              { icon: Biohazard, title: "Toxicity Gate", desc: "Compares the fine-tuned model's toxicity against the base model. Spikes trigger an automatic fail." },
              { icon: BrainCircuit, title: "Expert Collapse", desc: "(MoE) Verifies the router isn't lazily sending all tokens to just 1 or 2 favorite experts." },
              { icon: ShieldCheck, title: "Routing Regression", desc: "(MoE) Compares expert utilization against the base model to ensure the adapter didn't break routing." },
              { icon: HardDrive, title: "Memory Offload", desc: "(MoE) Streams 93GB models into 24GB GPUs safely, tracking H2D streams and evictions." },
            ].map((gate, i) => (
              <Card key={i} className="bg-zinc-900/50 border-white/10 hover:border-emerald-500/50 transition-colors">
                <CardHeader>
                  <gate.icon className="w-8 h-8 text-emerald-400 mb-2" />
                  <CardTitle className="text-white">{gate.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-400 text-base">
                    {gate.desc}
                  </CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Footer / CTA */}
      <section className="py-20 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <h2 className="text-3xl md:text-4xl font-bold">Ready to ship safely?</h2>
          <p className="text-gray-400 text-lg">Install tinct via pip and certify your first adapter in under 5 minutes.</p>
          <div className="inline-flex items-center gap-3 bg-zinc-900 border border-white/10 rounded-lg px-6 py-3 font-mono text-sm">
            <TerminalSquare className="w-4 h-4 text-emerald-400" />
            <span className="text-gray-300">pip install tinct</span>
            <Button variant="ghost" size="sm" className="h-6 px-2 text-xs text-gray-500 hover:text-white">Copy</Button>
          </div>
        </div>
      </section>
    </main>
  )
}