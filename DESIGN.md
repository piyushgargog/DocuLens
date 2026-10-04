# Design Direction — "Calm Light"

The design follows Claude's "Calm Light" design tokens, which the owner
provided as the reference.

## Brief

- **Purpose:** ask a document questions and see where each answer came from,
  calmly and with no friction.
- **Tone:** quiet, editorial, warm and spacious; not loud, playful or
  over-animated.
- **Constraints:** vanilla HTML/CSS/JS with no build step, WCAG AA, works at
  360px, untrusted text inserted only as text, nothing loaded from other
  origins.

## System

- **Colour:** ink `#0B0B0B` text and primary buttons; warm near-white
  `#FCFCFB` surfaces; translucent-ink hairline borders
  (`rgba(11,11,11,.10)`). The one accent, terracotta `#C6613F`, is used only
  for the logo, citation chips and focus ring. A warm dark theme mirrors this.
- **Type:** Newsreader (light weights) for the hero, prompts and wordmark;
  Inter for everything else. No all-caps labels.
- **Layout:** generous spacing (8/16/28/40/64), flat depth from borders and
  soft shadow, and radii of 10px (controls), 16px (panels) and 24px (cards).
- **Components:** solid ink primary button, quiet outlined secondary actions,
  a subtle neutral question bubble, a small monochrome "AI" mark, and Markdown
  answers with terracotta `p. N` chips. Source cards on a refused answer are
  dimmed.
- **Motion:** things ease into place. The landing page settles in with a quick
  stagger (0.55s, 40–420ms delays) and the footer heart beats. Under
  `prefers-reduced-motion` all other motion is off; those two keep their
  normal timing, because Windows' "animations off" setting otherwise made the
  page look broken. The spinner always turns. Async lists (the "Try asking"
  starters) show placeholders and fill in once.
- **Accessibility:** visible focus rings, keyboard-reachable controls,
  streamed answers announced through a live region, and AA contrast in both
  themes.

**Is not:** a loud SaaS UI, a neon dark theme, a gradient landing page, or a
dense broadsheet.
