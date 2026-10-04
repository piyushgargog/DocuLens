# Design Direction

Rewritten for v3.4.0. The owner provided Claude's own exported design tokens
("Claude Calm Light") as the reference and asked for DocuLens to adopt that
language, applied with the frontend-design method. This document records the
direction; the reference's words win wherever they are specific.

## Brief

- **Purpose:** ask a document questions and see exactly where each answer came
  from, calmly and with zero friction.
- **Audience:** students and reviewers who trust a tool that feels considered
  and intelligent, not marketing-heavy.
- **Tone:** quiet, editorial, warm, spacious — Claude's register. **Not:**
  loud, playful, gradient-y, or over-animated.
- **Constraints:** vanilla HTML/CSS/JS, no build step; WCAG AA; works at
  360px; untrusted text only ever inserted as text; nothing from other origins.

# Calm Light

A faithful application of **Claude Calm Light**: an airy, editorial surface
where ink and paper do the work and colour is spent only on identity.

**Colour — ink, paper, one warm accent.** Text and the primary call to action
are ink black (`#0B0B0B`). Surfaces are a warm near-white (`#FCFCFB`) and clean
white, separated by hairline *translucent-ink* borders (`rgba(11,11,11,.10)`)
rather than grey lines. The only chroma is a terracotta accent (`#C6613F`),
spent on brand moments and small emphasis — the logo, the citation chips, the
focus ring — never as a surface. Status greens/ambers are muted and functional.

**Type — a light serif voice over a clean sans.** Newsreader, a warm literary
text serif (Tiempos register), carries the big editorial lines — the hero
headline, the reading and drop prompts, the wordmark — at *light* weights so it
reads calm, not loud. Inter carries everything the interface needs to say:
body, labels, buttons, answers. No all-caps system labels, no mono small-data
labels, no one-word accenting in headlines — the headline is simply the serif,
whole, in sentence case.

**Layout — spacious and flat.** Generous negative space on a gentle rhythm
(8 / 16 / 28 / 40 / 64). Depth comes from the hairline borders, soft shadow and
whitespace, not elevation — a sheet of paper over a quiet ground. Controls and
inputs round at 10px, panels at 16px, cards at 24px; full pills are reserved
for chips.

**Components.** The primary button is a solid ink-black CTA with white text
(the strongest thing on screen); secondary actions are quiet and outlined;
links are underlined ink. The conversation keeps a subtle neutral question
bubble and a small monochrome "AI" mark; answers render as clean Markdown with
terracotta page-citation chips that open the exact source beside them.

**Motion.** Subtle and functional: things ease into place. The opening page
arrives in reading order with a quick soft stagger (0.55 s settle, 40-420 ms
delays), and the footer heart gives a small heartbeat. Under
`prefers-reduced-motion` almost everything is switched off, with two deliberate
exceptions that keep their normal timing: that page settle-in and the heartbeat.
Reason: on Windows with "Show animations" off the browser reports reduced motion,
and a page that pops in all at once with a dead heart read as broken to the
owner; a first attempt with slower opacity-only fades read as sluggish, so the
original quick motion is kept. The loading spinner always keeps turning. Lists
that load asynchronously (the "Try asking" starters) hold their place with quiet
placeholders and fill in once; nothing appears, vanishes and reappears.

**Is not:** a loud SaaS UI, a neon-on-black theme, a gradient landing page, or
a dense broadsheet. The single bold move is the light serif headline on warm
paper; everything else stays quiet and disciplined.

**Signature:** ink and warm paper with a single terracotta accent, and a calm
serif headline — the Claude register, applied honestly to a document tool.
