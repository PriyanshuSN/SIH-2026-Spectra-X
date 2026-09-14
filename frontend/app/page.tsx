import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ShieldCheck } from 'lucide-react'

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden bg-[#050508]">
      <div className="absolute inset-0 z-0 bg-grid"></div>
      
      <div className="z-10 flex flex-col items-center text-center max-w-4xl px-4 mt-[-50px]">
        <Badge variant="outline" className="mb-6 border-[#00ff41]/50 text-[#00ff41] bg-[#00ff41]/10 px-4 py-1 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4" />
          SIH26142 | NTRO Problem Statement
        </Badge>
        
        <h1 className="text-5xl md:text-7xl font-bold mb-4 tracking-tight text-white font-mono">
          SpectraX SRM
        </h1>
        
        <p className="text-xl md:text-2xl text-[#00ff41] mb-8 font-mono">
          &gt; Deep Learning Super-Resolution Mapping_
        </p>
        
        <p className="text-lg text-slate-300 mb-12 max-w-2xl leading-relaxed">
          Reconstructing genuine fine-scale geographic detail from 10m Sentinel-2 imagery using SwinIR neural networks. Verified against geographic fidelity metrics.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 mb-16">
          <Link href="/analyze">
            <Button size="lg" className="bg-[#1e3a2f] hover:bg-[#2a5242] border border-[#00ff41]/50 text-[#00ff41] px-8 hover:shadow-[0_0_15px_rgba(0,255,65,0.4)] transition-all font-mono">
              Launch Analysis →
            </Button>
          </Link>
          <Button size="lg" variant="outline" className="border-slate-700 text-slate-300 hover:bg-slate-800 font-mono">
            View on GitHub
          </Button>
        </div>
      </div>
      
      <div className="absolute bottom-0 left-0 w-full bg-[#0a0a0f] border-t border-[#1e3a2f] py-3 px-6 z-10">
        <div className="flex flex-wrap justify-center gap-x-8 gap-y-2 text-xs md:text-sm text-slate-400 font-mono">
          <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-[#00ff41]"></div> 600 Training Epochs</span>
          <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-[#00ff41]"></div> 0.90M Parameters</span>
          <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-[#00ff41]"></div> 353 Satellite Patches</span>
          <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-[#00ff41]"></div> Consistency Score: 0.92</span>
          <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-[#00ff41]"></div> 3× Resolution Enhancement</span>
        </div>
      </div>
    </main>
  )
}
