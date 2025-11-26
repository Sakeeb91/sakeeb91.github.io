window.PROJECTS = [
  {
    slug: "cacao-sommelier",
    title: "Cacao. | Modern Chocolatier",
    subtitle: "Premium chocolate e-commerce with AI-powered features",
    tags: ["Web Dev", "AI Integration"],
    highlights: [
      { label: "AI Features", text: "Gemini-powered pairing suggestions, text-to-speech tasting notes, and dynamic image generation." },
      { label: "Design", text: "Modern glassmorphism UI with smooth animations and responsive layouts." },
      { label: "Stack", text: "React + TypeScript + Vite deployed on Cloudflare Pages with global CDN." }
    ],
    links: [
      { label: "Live demo", url: "https://cacao-sommelier.pages.dev" },
      { label: "Source code", url: "https://github.com/Sakeeb91/cacao-sommelier" },
      { label: "View project", url: "https://sakeeb91.github.io/cacao-sommelier/" }
    ]
  },
  {
    slug: "gpinn",
    title: "Gradient-Enhanced Physics-Informed Neural Network (gPINN)",
    subtitle: "Inverse modeling for porous media with uncertainty bounds",
    tags: ["Geothermal", "Physics ML"],
    highlights: [
      { label: "Problem", text: "Recovers permeability and viscosity from sparse sensors with 70–90% accuracy." },
      { label: "Approach", text: "Gradient-enhanced PINNs converge in 2–5 minutes on CPU while quantifying uncertainty." },
      { label: "Impact", text: "Projected 80% reduction in geothermal exploration modeling costs." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/gpinn/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/Gradient-enhanced-Physics-Informed-Neural-Network-gPINN" }
    ]
  },
  {
    slug: "aerosurrogate-scikit",
    title: "AeroSurrogate-Scikit",
    subtitle: "Surrogate models that replace millisecond CFD solves",
    tags: ["Automotive CFD", "ML Surrogates"],
    highlights: [
      { label: "Data", text: "355 WMLES simulations expanded from 7 → 25+ engineered features." },
      { label: "Speed", text: "Achieves 1000× faster inference than full CFD without sacrificing accuracy." },
      { label: "Workflow", text: "Physics-informed validation keeps real-time design sweeps trustworthy." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/aerosurrogate-scikit/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/AeroSurrogate-Scikit" }
    ]
  },
  {
    slug: "drag-coefficient-prediction",
    title: "Drag Coefficient Prediction",
    subtitle: "Interpretable neural solvers for aerodynamic drag",
    tags: ["CFD", "Physics-Guided NN"],
    highlights: [
      { label: "Accuracy", text: "Delivers 99.54% drag prediction accuracy across multiple flow regimes." },
      { label: "Interpretability", text: "Log-Reynolds feature engineering keeps the model physically grounded." },
      { label: "Iteration", text: "~10 second CPU training loops support rapid experimentation." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/drag-coefficient-prediction/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/drag-coefficient-prediction" }
    ]
  },
  {
    slug: "fitzhugh-nagumo",
    title: "FitzHugh-Nagumo Lift & Learn",
    subtitle: "Deployable ROMs for nonlinear PDEs",
    tags: ["Biosignals", "Reduced Order"],
    highlights: [
      { label: "Pipeline", text: "Implements the full Lift → POD → Operator Inference workflow end to end." },
      { label: "Performance", text: "Hits <1e-3 reconstruction error with 100× speedups." },
      { label: "Robustness", text: "Noise analysis and interactive plots make the ROM production ready." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/FitzHugh-Nagumo-Lift-Learn/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/FitzHugh-Nagumo-Lift-Learn" }
    ]
  },
  {
    slug: "gardner-capacity-puzzle",
    title: "The Gardner Capacity Puzzle",
    subtitle: "Explaining why flawed derivations still work",
    tags: ["Statistical Physics", "Learning Theory"],
    highlights: [
      { label: "Visualization", text: "First 3D mapping of Gardner phase space for perceptron storage." },
      { label: "Validation", text: "Confirms α_c = 1/(κ² + 1) with R² = 0.9799 correlation." },
      { label: "Insight", text: "Reveals hidden symmetries linking statistical physics and generalization bounds." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/gardner-capacity-puzzle/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/gardner-capacity-puzzle" }
    ]
  },
  {
    slug: "topological-photonic-crystal-optimizer",
    title: "Topological Photonic Crystal Optimizer",
    subtitle: "Machine learning framework for disorder-robust resonators",
    tags: ["Photonics", "Bayesian Opt"],
    highlights: [
      { label: "Search", text: "NSGA-III + MEEP stack explores 400+ designs with 13+ physics features." },
      { label: "Result", text: "Jointly optimizes Q-factor, robustness, bandgap, and mode volume." },
      { label: "Automation", text: "Framework auto-extracts design rules for fabrication-ready devices." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/topological-photonic-crystal-optimizer/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/topological-photonic-crystal-optimizer" },
      { label: "Technical docs", url: "https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/blob/main/ADVANCED_FRAMEWORK_SUMMARY.md" }
    ]
  },
  {
    slug: "reinforcement-learning-grpo",
    title: "GRPO Healthcare AI",
    subtitle: "Constraint-aware ICU resource allocation",
    tags: ["Healthcare Ops", "Fair RL"],
    highlights: [
      { label: "Problem", text: "Balances ICU admissions across demographic cohorts under load." },
      { label: "Method", text: "Extends PPO with a group-robust constraint critic and fairness shaping." },
      { label: "Impact", text: "Delivers 25% fairness gains while holding operational throughput steady." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/reinforcement-learning-grpo/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/reinforcement-learning-grpo" }
    ]
  },
  {
    slug: "chaotic-systems-analysis",
    title: "Chaotic Systems Analysis",
    subtitle: "One-click diagnostics for chaotic attractors",
    tags: ["Nonlinear Dynamics", "Python Toolkit"],
    highlights: [
      { label: "Automation", text: "Parameter sweeps, Lyapunov spectra, and fractal dimensions generated automatically." },
      { label: "Visualization", text: "Outputs publication-ready attractor plots and bifurcation diagrams." },
      { label: "Ops", text: "CLI templates cut new-system setup from hours to minutes." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/chaotic-systems-analysis/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/chaotic-systems-analysis" }
    ]
  },
  {
    slug: "hopfield-market-regimes",
    title: "Hopfield Network Market Regimes",
    subtitle: "Neural physics for trader-ready regime maps",
    tags: ["Capital Markets", "Hopfield Nets"],
    highlights: [
      { label: "Framing", text: "Treats market regimes as stable attractors in a Hopfield energy landscape." },
      { label: "Signals", text: "Feeds the network with multi-factor technical indicator vectors." },
      { label: "Utility", text: "Produces regime tags that inform hedging and allocation playbooks." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/hopfield-market-regimes/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/hopfield-market-regimes" }
    ]
  },
  {
    slug: "EACTO",
    title: "EACTO",
    subtitle: "Entropy-adaptive thresholds for volatile markets",
    tags: ["Risk Management", "Cybernetics"],
    highlights: [
      { label: "Signals", text: "Uses entropy and complexity metrics to sense regime turbulence." },
      { label: "Engine", text: "Cybernetic controller blends information theory with statistical modeling." },
      { label: "Outcome", text: "Dynamically outperforms static VaR models during volatility spikes." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/EACTO/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/EACTO" }
    ]
  },
  {
    slug: "financial-asset-dynamics",
    title: "Financial Asset Dynamics",
    subtitle: "Institutional-grade analytics in one repo",
    tags: ["Quant Library", "Python"],
    highlights: [
      { label: "Coverage", text: "Portfolio optimization, VaR/CVaR, and Black–Scholes pricing modules." },
      { label: "Data", text: "Plugs into streaming feeds with built-in regime detection." },
      { label: "Usage", text: "Supports hedge-fund dashboards and research automation." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/financial-asset-dynamics/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/financial-asset-dynamics" }
    ]
  },
  {
    slug: "lqr-goodwin-oscillator-control",
    title: "LQR Goodwin Oscillator Control",
    subtitle: "Stabilizing pathological oscillations",
    tags: ["Control", "Macro Systems"],
    highlights: [
      { label: "Modeling", text: "Linearizes the Goodwin oscillator to design stabilizing LQR policies." },
      { label: "Integration", text: "Links macroeconomic drivers to biological and supply-chain oscillations." },
      { label: "Delivery", text: "Outputs dashboard-ready signals for therapy and operations teams." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/lqr-goodwin-oscillator-control/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/lqr-goodwin-oscillator-control" }
    ]
  },
  {
    slug: "robust-agent-navigation",
    title: "Robust Agent Navigation",
    subtitle: "Teaching agents to plan through uncertainty",
    tags: ["RL Demo", "Stochastic Nav"],
    highlights: [
      { label: "Environment", text: "Custom gridworld with stochastic motion and dynamic obstacles." },
      { label: "Learning", text: "Q-learning policy adapts to varying noise levels and penalties." },
      { label: "Insights", text: "Visualization suite reveals convergence, heatmaps, and agent resilience." }
    ],
    links: [
      { label: "View project", url: "https://sakeeb91.github.io/robust-agent-navigation/" },
      { label: "Source code", url: "https://github.com/Sakeeb91/robust-agent-navigation" }
    ]
  }
];
