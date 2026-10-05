/**
 * DocuLens -- frontend logic.
 *
 * Talks to the FastAPI backend (main.py) at /api/ingest, /api/ask/stream
 * (server-sent events), /api/summary, /api/suggestions, /api/remove,
 * /api/session and /api/status. All dynamic content
 * (questions, answers, document text, filenames) is inserted with
 * textContent / text nodes, never innerHTML, so nothing from a document or
 * the model can inject markup into the page.
 */

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

const uploadView = document.getElementById("upload-view");
const indexingView = document.getElementById("indexing-view");
const chatView = document.getElementById("chat-view");

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const uploadError = document.getElementById("upload-error");

const docList = document.getElementById("doc-list");
const addFileInput = document.getElementById("add-file-input");
const addBtn = document.getElementById("add-btn");
const addStatus = document.getElementById("add-status");
const removeBtn = document.getElementById("remove-btn");

const messagesEl = document.getElementById("messages");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const stopBtn = document.getElementById("stop-btn");
const attachBtn = document.getElementById("attach-btn");
const scopeLabel = document.getElementById("scope-label");
const exportBtn = document.getElementById("export-btn");

const docItemTemplate = document.getElementById("doc-item-template");
const sourceItemTemplate = document.getElementById("source-item-template");

function showView(view) {
  const changed = document.body.dataset.view !== view;
  document.body.dataset.view = view;
  const views = { upload: uploadView, indexing: indexingView, chat: chatView };
  for (const [name, el] of Object.entries(views)) {
    el.hidden = name !== view;
    // Fade the newly shown view in; restart the animation on every switch.
    el.classList.remove("view-enter");
    if (changed && name === view) {
      void el.offsetWidth;
      el.classList.add("view-enter");
    }
  }
  if (changed) view === "indexing" ? startReadingFacts() : stopReadingFacts();
}

// While a document is being read, rotate a quiet fact about documents so the
// wait feels alive and informative rather than blank.
const DOC_FACTS = [
  "“Document” comes from the Latin documentum — “a lesson, proof, or evidence.”",
  "The PDF format was created by Adobe in 1993 and became an open ISO standard in 2008.",
  "A single page of dense text holds roughly 500 words — about 3,000 characters.",
  "The world’s oldest surviving printed book, the Diamond Sutra, dates to 868 AD.",
  "Retrieval reads your whole document, then answers from only the most relevant passages.",
  "Every answer here cites the page it came from — so you can always check the source.",
  "Good chunking keeps sentences whole, so a passage never ends mid-thought.",
  "Scanned pages with no text layer are read with OCR before anything is indexed.",
  "A citation you can click is a citation you can trust — evidence beats eloquence.",
  "The first email attachment was sent in 1992; the file outlived the message.",
  "Embeddings turn each passage into a point in space, so similar ideas sit close together.",
  "If the document doesn’t say it, a good assistant says so — instead of guessing.",
];
let _factTimer = 0;

function startReadingFacts() {
  const el = document.getElementById("reading-fact");
  if (!el) return;
  const pool = DOC_FACTS.slice();
  const next = () => {
    if (pool.length === 0) pool.push(...DOC_FACTS);
    const fact = pool.splice(Math.floor(Math.random() * pool.length), 1)[0];
    el.classList.remove("show");
    // Let the fade-out run, then swap and fade in.
    setTimeout(() => {
      el.textContent = fact;
      el.classList.add("show");
    }, 180);
  };
  el.textContent = "";
  next();
  clearInterval(_factTimer);
  _factTimer = setInterval(next, 3600);
}

function stopReadingFacts() {
  clearInterval(_factTimer);
  _factTimer = 0;
}

/**
 * fetch + JSON that never throws. A proxy in front of the app (e.g. Nginx
 * returning an HTML 413/502 page) or a dropped connection becomes
 * {error: "..."} instead of an exception that would leave the UI stuck.
 */
async function api(url, body) {
  const options = { method: body === undefined ? "GET" : "POST" };
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    return { error: "Couldn't reach the server. Check your connection and try again." };
  }

  let data;
  try {
    data = await response.json();
  } catch (err) {
    if (response.status === 413) return { error: "That file is over the 25MB limit." };
    return { error: `The server sent an unexpected response (HTTP ${response.status}). Try again in a moment.` };
  }
  if (!response.ok && !data.error) {
    data.error = `The request failed (HTTP ${response.status}). Try again in a moment.`;
  }
  data.httpStatus = response.status;
  return data;
}

// ---------- Messages ----------

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/**
 * Scroll the conversation so `el` sits at the top of the pane. Used when an
 * answer arrives: jumping to the bottom would skip past the start of a long
 * answer, so the view stops at the question and the answer reads top-down.
 */
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

function scrollToStart(el, smooth = true) {
  const offset = el.getBoundingClientRect().top - messagesEl.getBoundingClientRect().top;
  messagesEl.scrollTo({
    top: messagesEl.scrollTop + offset - 12,
    behavior: smooth && !reducedMotion.matches ? "smooth" : "auto",
  });
}

function append(el) {
  el.classList.add("msg-enter");
  messagesEl.appendChild(el);
  scrollToBottom();
  return el;
}

function addUser(text) {
  const el = document.createElement("p");
  el.className = "msg msg-user";
  el.textContent = text;
  return append(el);
}

function addNote(text) {
  const el = document.createElement("div");
  el.className = "msg msg-assistant msg-note";
  const p = document.createElement("p");
  p.className = "msg-text";
  p.textContent = text;
  el.appendChild(p);
  return append(el);
}

function addPending(text) {
  const el = addNote(text);
  el.classList.add("msg-pending");
  return el;
}

function addError(text) {
  const el = document.createElement("p");
  el.className = "msg msg-error";
  el.setAttribute("role", "alert");
  el.removeAttribute("aria-busy");
  el.textContent = text;
  return append(el);
}

// Plain-text flattening of an answer, for Copy / Export (no DOM).
function plain(text) {
  return delatex(text)
    .replace(/```[^\n]*\n?/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/(^|[\s(])[*_](?=\S)([^*_\n]+?)[*_](?=[\s).,!?:;]|$)/g, "$1$2")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^([ \t]*)[*\-•][ \t]+/gm, "$1• ")
    .replace(/^([ \t]*)(\d+)\.[ \t]+/gm, "$1$2. ");
}

// The prompt asks for plain text, but models (especially smaller fallback
// ones) still write LaTeX for formulas. Turn the common pieces into readable
// text rather than showing backslashes.
function delatex(text) {
  const symbols = { times: "×", cdot: "·", dots: "…", ldots: "…", approx: "≈", leq: "≤", geq: "≥", infty: "∞", to: "→", in: "∈" };
  let out = text
    .replace(/\\\(|\\\)|\\\[|\\\]/g, "")
    .replace(/\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}/g, "$1")
    .replace(/\\sqrt\{([^{}]*)\}/g, "√$1")
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "($1)/($2)")
    .replace(/\\([a-zA-Z]+)/g, (m, name) => symbols[name] ?? m);
  // x_{i} -> x_i, x^{2} -> x^2 (braces only add noise in plain text);
  // repeated so nested braces unwrap from the inside out.
  for (let previous = ""; previous !== out; ) {
    previous = out;
    out = out.replace(/([_^])\{([^{}]*)\}/g, "$1$2");
  }
  return out;
}

// Citations the model writes, e.g. 【report.pdf, Page 3】, [Page 3],
// (Page 3), (see Pages 3 and 4) or (report.pdf, Page 3). The model isn't
// consistent about which form it uses, so all are turned into page tabs.
const PAGE_REF = String.raw`(?:[^()\[\]\n]*?,\s*)?Pages?\s*(?:[\d\s,–-]|and|&)+`;
const CITATION = new RegExp(
  String.raw`【([^】]*)】|\[(${PAGE_REF})\]|\((?:see\s+)?(${PAGE_REF})\)`,
  "gi",
);

function citedPages(inner) {
  const at = inner.search(/Pages?\s*\d/i);
  if (at < 0) return { doc: "", pages: [] };
  // Everything before the comma preceding "Page" is the document name; only
  // numbers after "Page" are pages (so "report2024.pdf" isn't page 2024).
  const comma = inner.lastIndexOf(",", at);
  const doc = comma > 0 ? inner.slice(0, comma).trim() : "";
  const pages = [...inner.slice(at).matchAll(/\d+/g)].map((m) => Number(m[0]));
  return { doc: /page/i.test(doc) ? "" : doc, pages: [...new Set(pages)] };
}

/** One citation (e.g. "report.pdf, Page 3") -> page chips appended to `el`.
 *  A chip whose page is among this answer's sources is a button that opens
 *  the sources and highlights the passage; otherwise it's a plain label. */
function appendCitation(el, inner, notes) {
  const { doc, pages } = citedPages(inner);
  if (pages.length === 0) return false;
  for (const page of pages) {
    const slip = findSlip(notes, page, doc);
    const tab = document.createElement(slip ? "button" : "span");
    tab.className = "cite";
    tab.textContent = `p. ${page}`;
    tab.title = doc ? `${doc}, page ${page}` : `Page ${page}`;
    if (slip) {
      tab.type = "button";
      tab.setAttribute("aria-label", `Show the passage from ${tab.title}`);
      tab.addEventListener("click", () => markSlip(notes, slip));
    }
    el.append(tab);
  }
  return true;
}

/** Inline **bold**, *italic* and `code` within one plain run -> text/strong/em/code nodes. */
function appendFormatted(el, text) {
  const INLINE = /\*\*(.+?)\*\*|`([^`]+)`|(?:^|(?<=[\s(]))[*_](\S(?:[^*_\n]*\S)?)[*_](?=[\s).,!?:;]|$)/g;
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > last) el.append(text.slice(last, m.index));
    if (m[1] !== undefined) {
      const b = document.createElement("strong");
      b.textContent = m[1];
      el.append(b);
    } else if (m[2] !== undefined) {
      const c = document.createElement("code");
      c.textContent = m[2];
      el.append(c);
    } else {
      const i = document.createElement("em");
      i.textContent = m[3];
      el.append(i);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) el.append(text.slice(last));
}

/** Inline content with citations: split on citations, format the rest. */
function appendInline(el, text, notes) {
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const inner = match[1] ?? match[2] ?? match[3];
    const before = text.slice(last, match.index);
    if (citedPages(inner).pages.length === 0) continue;
    appendFormatted(el, before.replace(/\s+$/, ""));
    appendCitation(el, inner, notes);
    last = match.index + match[0].length;
  }
  appendFormatted(el, text.slice(last));
}

/**
 * Render the answer as rich (but strictly safe) Markdown into `container`:
 * headings, bullet and numbered lists, code blocks, inline code, bold/italic,
 * blockquotes, tables and rules — with page citations turned into chips.
 * Every node is built with createElement + textContent, never innerHTML, so
 * nothing in the model's (or document's) text can inject markup.
 */
function renderAnswerText(container, text, notes) {
  const lines = delatex(text).replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  const isUL = (l) => /^\s*[-*•]\s+/.test(l);
  const isOL = (l) => /^\s*\d+[.)]\s+/.test(l);

  while (i < lines.length) {
    let line = lines[i];
    if (!line.trim()) { i++; continue; }

    // Fenced code block
    if (/^\s*```/.test(line)) {
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++;
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = body.join("\n");
      pre.append(code);
      container.append(pre);
      continue;
    }
    // Heading
    const h = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
    if (h) {
      const el = document.createElement(h[1].length <= 2 ? "h4" : "h5");
      el.className = "md-h";
      appendInline(el, h[2].trim(), notes);
      container.append(el);
      i++;
      continue;
    }
    // Horizontal rule
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      container.append(document.createElement("hr"));
      i++;
      continue;
    }
    // Table (header row with pipes, then a separator row of ---)
    if (line.includes("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const table = document.createElement("table");
      table.className = "md-table";
      const thead = document.createElement("thead");
      const htr = document.createElement("tr");
      for (const c of cells(line)) { const th = document.createElement("th"); appendInline(th, c, notes); htr.append(th); }
      thead.append(htr); table.append(thead);
      i += 2;
      const tbody = document.createElement("tbody");
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        const tr = document.createElement("tr");
        for (const c of cells(lines[i])) { const td = document.createElement("td"); appendInline(td, c, notes); tr.append(td); }
        tbody.append(tr); i++;
      }
      table.append(tbody);
      container.append(table);
      continue;
    }
    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const bq = document.createElement("blockquote");
      const parts = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) parts.push(lines[i++].replace(/^\s*>\s?/, ""));
      appendInline(bq, parts.join(" "), notes);
      container.append(bq);
      continue;
    }
    // Lists (unordered / ordered)
    if (isUL(line) || isOL(line)) {
      const ordered = isOL(line);
      const list = document.createElement(ordered ? "ol" : "ul");
      list.className = "md-list";
      while (i < lines.length && (ordered ? isOL(lines[i]) : isUL(lines[i]))) {
        const li = document.createElement("li");
        let item = lines[i].replace(ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*•]\s+/, "");
        i++;
        // Continuation lines (indented, not a new list item or blank).
        while (i < lines.length && lines[i].trim() && !isUL(lines[i]) && !isOL(lines[i]) && /^\s+/.test(lines[i])) {
          item += " " + lines[i].trim();
          i++;
        }
        appendInline(li, item, notes);
        list.append(li);
      }
      container.append(list);
      continue;
    }
    // Paragraph: gather consecutive plain lines.
    const para = [];
    while (i < lines.length && lines[i].trim() && !isUL(lines[i]) && !isOL(lines[i]) &&
           !/^\s*(```|>|#{1,6}\s)/.test(lines[i])) {
      para.push(lines[i++]);
    }
    const p = document.createElement("p");
    p.className = "md-p";
    appendInline(p, para.join(" "), notes);
    container.append(p);
  }
}

function findSlip(notes, page, doc) {
  if (!notes) return null;
  const slips = [...notes.list.querySelectorAll(".slip")];
  return (
    slips.find((s) => Number(s.dataset.page) === page && (!doc || s.dataset.doc === doc)) ||
    slips.find((s) => Number(s.dataset.page) === page) ||
    null
  );
}

function setNotesOpen(notes, open) {
  notes.panel.classList.toggle("open", open);
  notes.toggle.setAttribute("aria-expanded", String(open));
  notes.toggle.textContent = `${open ? "Hide" : "Show"} ${notes.label.toLowerCase()} (${notes.count})`;
}

function markSlip(notes, slip) {
  setNotesOpen(notes, true);
  for (const s of notes.list.querySelectorAll(".slip.marked")) s.classList.remove("marked");
  // Restart the highlighter animation even when the same tab is clicked twice.
  void slip.offsetWidth;
  slip.classList.add("marked");
  slip.scrollIntoView({ block: "nearest", behavior: reducedMotion.matches ? "auto" : "smooth" });
}

/**
 * Source passages as margin notes. On a wide conversation they sit beside
 * the answer (CSS container query); on a narrow one they fold under it
 * behind a Show/Hide button.
 */
function buildNotes(sources, label) {
  if (!sources || sources.length === 0) return null;

  // Only name the document when more than one is loaded -- otherwise it's noise.
  const multiDoc = new Set(sources.map((s) => s.doc)).size > 1 || docList.children.length > 1;

  const panel = document.createElement("aside");
  panel.className = "notes";
  panel.setAttribute("aria-label", label);
  const list = document.createElement("ol");
  list.className = "slips";
  panel.appendChild(list);

  for (const [i, src] of sources.entries()) {
    const node = sourceItemTemplate.content.cloneNode(true);
    const slip = node.querySelector(".slip");
    slip.style.setProperty("--i", i);
    slip.dataset.page = src.page;
    slip.dataset.doc = src.doc || "";
    node.querySelector(".slip-page").textContent =
      multiDoc && src.doc ? `${src.doc}, p. ${src.page}` : `Page ${src.page}`;
    node.querySelector(".slip-score").textContent =
      typeof src.score === "number" ? `similarity ${src.score.toFixed(2)}` : "";
    if (src.section) {
      const sectionEl = node.querySelector(".slip-section");
      sectionEl.textContent = src.section;
      sectionEl.title = src.section;
      sectionEl.hidden = false;
    }
    // PDF extraction keeps the page's hard line breaks; rejoin them so the
    // passage reads as prose (blank lines between paragraphs are kept).
    node.querySelector(".slip-text").textContent = src.text.replace(/(?<!\n)\n(?!\n)/g, " ");
    // In the margin, notes are clipped to a few lines; clicking one opens it.
    slip.tabIndex = 0;
    slip.setAttribute("role", "button");
    slip.setAttribute("aria-expanded", "false");
    const toggleSlip = () => slip.setAttribute("aria-expanded", String(slip.classList.toggle("expanded")));
    slip.addEventListener("click", toggleSlip);
    slip.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleSlip();
      }
    });
    list.appendChild(node);
  }

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "btn btn-text notes-toggle";
  const notes = { panel, list, toggle, label, count: sources.length };
  toggle.addEventListener("click", () => setNotesOpen(notes, !panel.classList.contains("open")));
  setNotesOpen(notes, false);
  return notes;
}

/** Turn a pending message into a finished answer (or add a new one). */
const REFUSAL = "I could not find the answer";
const FOLLOW_UPS = [
  ["Explain more simply", "Explain that more simply, in plain words for someone new to the topic, still using only what the passages say."],
  // Asking for "more" invites padding from outside knowledge (seen in testing
  // on a short document), so the instruction repeats the grounding rule.
  ["Go deeper", "Give more detail on that, using only details that appear in the passages. If the documents say nothing more, say so."],
];

function buildActions(answer, { followUps }) {
  const bar = document.createElement("div");
  bar.className = "answer-actions";
  if (followUps) {
    for (const [label, question] of FOLLOW_UPS) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn btn-text follow-up";
      button.textContent = label;
      button.addEventListener("click", () => askQuestion(question, label));
      bar.appendChild(button);
    }
  }
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "btn btn-text copy-answer";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(answerAsText(answer));
      copy.textContent = "Copied";
      showToast("Answer copied, with its sources");
    } catch (err) {
      copy.textContent = "Couldn't copy";
    }
    setTimeout(() => (copy.textContent = "Copy"), 1600);
  });
  bar.appendChild(copy);
  return bar;
}

/** "gpt-oss-120b" rather than "openai/gpt-oss-120b:free". */
function modelName(model) {
  return model.replace(/^[\w.-]+\//, "").replace(/:free$/, "");
}

/** Which provider and model wrote an answer, shown beside its actions. */
function routeTag(route) {
  const tag = document.createElement("span");
  tag.className = "answered-by";
  tag.title = `Answered by ${route.provider} (${route.model})`;
  const dot = document.createElement("span");
  dot.className = "answered-by-dot";
  dot.setAttribute("aria-hidden", "true");
  tag.append(dot, `${route.provider} · ${modelName(route.model)}`);
  return tag;
}

/** The answer as plain text, with its sources listed underneath. */
function answerAsText(answer) {
  const lines = [plain(answer.text).trim()];
  if (answer.sources.length) {
    lines.push("", "Sources:");
    for (const src of answer.sources) {
      const where = src.doc ? `${src.doc}, page ${src.page}` : `Page ${src.page}`;
      const excerpt = src.text.replace(/\s+/g, " ").trim();
      lines.push(`- ${where}: "${excerpt.length > 160 ? excerpt.slice(0, 160) + "…" : excerpt}"`);
    }
  }
  return lines.join("\n");
}

/** Lay out an answer (text + margin notes) inside `el`; the text is filled in
 *  by updateAnswer, repeatedly while streaming. */
function startAnswer(el, sources, { heading, sourcesLabel = "Sources" } = {}) {
  el.className = "msg msg-assistant answer msg-enter";
  el.removeAttribute("role");
  // The conversation is a polite live region; keep this answer quiet while
  // it is being written so screen readers announce it once, when complete.
  el.setAttribute("aria-busy", "true");
  el.replaceChildren();

  const main = document.createElement("div");
  main.className = "answer-main";
  if (heading) {
    const h = document.createElement("p");
    h.className = "msg-heading";
    h.textContent = heading;
    main.appendChild(h);
  }
  const thinking = buildThinking();
  main.appendChild(thinking.wrap);
  const notes = buildNotes(sources, sourcesLabel);
  const body = document.createElement("div");
  body.className = "msg-text";
  main.appendChild(body);
  el.appendChild(main);
  if (notes) {
    el.classList.add("has-notes");
    el.appendChild(notes.panel);
  }
  return { el, main, body, notes, thinking, sources: sources || [], text: "", route: null };
}

/** The model's own reasoning, shown like ChatGPT's "Thinking": a collapsible
 *  panel that streams while the model thinks, then folds to "Thought for Ns".
 *  Hidden entirely for models that expose no reasoning. */
function buildThinking() {
  const wrap = document.createElement("div");
  wrap.className = "thinking";
  wrap.hidden = true;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "thinking-toggle";
  toggle.setAttribute("aria-expanded", "true");
  const spark = document.createElement("span");
  spark.className = "thinking-spark";
  spark.setAttribute("aria-hidden", "true");
  const label = document.createElement("span");
  label.className = "thinking-label";
  label.textContent = "Thinking";
  const chevron = document.createElement("span");
  chevron.className = "thinking-chevron";
  chevron.setAttribute("aria-hidden", "true");
  toggle.append(spark, label, chevron);

  const body = document.createElement("div");
  body.className = "thinking-body";
  const text = document.createElement("p");
  text.className = "thinking-text";
  body.appendChild(text);

  const t = { wrap, toggle, label, body, text, raw: "", startedAt: 0, done: false, open: true };
  toggle.addEventListener("click", () => setThinkingOpen(t, !t.open));
  wrap.append(toggle, body);
  return t;
}

function setThinkingOpen(t, open) {
  t.open = open;
  t.wrap.classList.toggle("open", open);
  t.toggle.setAttribute("aria-expanded", String(open));
}

function appendReasoning(answer, text) {
  const t = answer.thinking;
  if (!t.startedAt) {
    t.startedAt = performance.now();
    t.wrap.hidden = false;
    t.wrap.classList.add("live");
    setThinkingOpen(t, true);
  }
  t.raw += text;
  t.text.textContent = t.raw;
  t.body.scrollTop = t.body.scrollHeight;
}

/** Called when the first answer text arrives: stop the thinking animation and
 *  collapse it to a one-line "Thought for Ns" the reader can reopen. */
function finishThinking(answer) {
  const t = answer.thinking;
  if (!t.startedAt || t.done) return;
  t.done = true;
  t.wrap.classList.remove("live");
  t.wrap.classList.add("thinking-done");
  const secs = Math.max(1, Math.round((performance.now() - t.startedAt) / 1000));
  t.label.textContent = `Thought for ${secs}s`;
  setThinkingOpen(t, false);
}

function updateAnswer(answer) {
  answer.body.replaceChildren();
  renderAnswerText(answer.body, answer.text, answer.notes);
}

function finishAnswer(answer, { followUps = false, stopped = false } = {}) {
  answer.body.classList.remove("streaming");
  if (answer.thinking) finishThinking(answer);
  updateAnswer(answer);
  if (stopped) {
    const note = document.createElement("p");
    note.className = "answer-stopped";
    note.textContent = "Stopped. This partial answer isn't used for follow-up questions.";
    answer.main.appendChild(note);
  }
  const refused = answer.text.trim().startsWith(REFUSAL);
  if (refused && answer.notes) markSearchedOnly(answer);
  if (!stopped && answer.text) {
    const bar = buildActions(answer, { followUps: followUps && !refused });
    if (answer.route) bar.appendChild(routeTag(answer.route));
    answer.main.appendChild(bar);
  }
  if (answer.notes) answer.main.appendChild(answer.notes.toggle);
  answer.el.removeAttribute("aria-busy");
}

/** A refusal's passages were searched, not used: fold them behind the toggle
 *  (even on wide screens) and style them as a quiet record, not as evidence. */
function markSearchedOnly(answer) {
  const notes = answer.notes;
  answer.el.classList.remove("has-notes");
  answer.el.classList.add("answer-refused");
  notes.panel.classList.add("notes-searched");
  notes.label = "Passages searched";
  notes.panel.setAttribute("aria-label", "Passages searched (none answered the question)");
  if (!notes.panel.querySelector(".notes-caption")) {
    const caption = document.createElement("p");
    caption.className = "notes-caption";
    caption.textContent = "These were searched, but none of them answers the question.";
    notes.panel.prepend(caption);
  }
  setNotesOpen(notes, false);
}

function showAnswer(el, text, sources, { heading, sourcesLabel = "Sources", followUps = false, route = null } = {}) {
  const answer = startAnswer(el, sources, { heading, sourcesLabel });
  answer.text = text;
  answer.route = route;
  finishAnswer(answer, { followUps });
  return answer;
}

function showFailure(el, text) {
  el.className = "msg msg-error";
  el.setAttribute("role", "alert");
  el.textContent = text;
}

// ---------- Documents ----------

let shownDocIds = new Set();

// Documents the reader has unticked; questions search all the others.
const excludedDocIds = new Set();
let loadedDocs = [];

function includedDocIds() {
  return loadedDocs.map((d) => d.id).filter((id) => !excludedDocIds.has(id));
}

function syncScope() {
  const total = loadedDocs.length;
  const included = includedDocIds().length;
  if (total === 0) scopeLabel.textContent = "";
  else if (included === 0) scopeLabel.textContent = "Tick a document to search";
  else if (total === 1) scopeLabel.textContent = "Searching 1 document";
  else if (included === total) scopeLabel.textContent = `Searching all ${total} documents`;
  else scopeLabel.textContent = `Searching ${included} of ${total} documents`;
  scopeLabel.classList.toggle("scope-empty", total > 0 && included === 0);
  syncQuestionState();
}

function renderDocuments(documents) {
  loadedDocs = documents;
  for (const id of [...excludedDocIds]) if (!documents.some((d) => d.id === id)) excludedDocIds.delete(id);
  docList.replaceChildren();
  for (const doc of documents) {
    const node = docItemTemplate.content.cloneNode(true);
    if (!shownDocIds.has(doc.id)) node.querySelector(".doc-item").classList.add("doc-enter");
    const include = node.querySelector(".doc-include");
    include.checked = !excludedDocIds.has(doc.id);
    include.setAttribute("aria-label", `Search ${doc.filename}`);
    include.addEventListener("change", () => {
      if (include.checked) excludedDocIds.delete(doc.id);
      else excludedDocIds.add(doc.id);
      syncScope();
    });
    node.querySelector(".doc-name").textContent = doc.filename;
    node.querySelector(".doc-name").title = doc.filename;
    node.querySelector(".doc-meta").textContent =
      `${doc.num_pages} ${doc.num_pages === 1 ? "page" : "pages"}`;
    node.querySelector(".doc-summary").addEventListener("click", (e) => summarizeDoc(doc, e.currentTarget));
    const remove = node.querySelector(".doc-remove");
    remove.setAttribute("aria-label", `Remove ${doc.filename}`);
    remove.title = "Remove";
    remove.addEventListener("click", () => removeDoc(doc.id));
    docList.appendChild(node);
  }
  shownDocIds = new Set(documents.map((d) => d.id));
  syncScope();
  if (documents.length === 0) {
    transcript.length = 0;
    syncExport();
    messagesEl.replaceChildren();
    fileInput.value = "";
    showView("upload");
  }
}

function validateFile(file) {
  if (!/\.(pdf|txt|md|markdown|docx)$/i.test(file.name)) return "Unsupported file. Use a PDF, Word (.docx), text or Markdown file.";
  if (file.size > MAX_UPLOAD_BYTES) return "That file is over the 25MB limit.";
  return null;
}

async function ingest(file) {
  const formData = new FormData();
  formData.append("file", file);
  return api("/api/ingest", formData);
}

// First document: full-screen upload -> indexing -> chat.
async function uploadFirst(file) {
  uploadError.hidden = true;
  const problem = validateFile(file);
  if (problem) {
    uploadError.textContent = problem;
    uploadError.hidden = false;
    return;
  }

  document.getElementById("reading-name").textContent = `Reading ${file.name}…`;
  showView("indexing");
  const data = await ingest(file);
  fileInput.value = "";

  if (data.error) {
    showView("upload");
    uploadError.textContent = data.error;
    uploadError.hidden = false;
    return;
  }

  messagesEl.replaceChildren();
  showView("chat");
  renderDocuments(data.documents);
  addNote(`${data.filename} is ready. Ask anything about it.`);
  for (const warning of data.warnings || []) addNote(warning);
  showSuggestions(data.id);
  questionInput.value = "";
  questionInput.focus();
}

// Further documents: read inline, conversation stays.
async function uploadAdditional(file) {
  const problem = validateFile(file);
  if (problem) {
    addError(problem);
    return;
  }

  addStatus.hidden = false;
  addBtn.hidden = true;
  const data = await ingest(file);
  addStatus.hidden = true;
  addBtn.hidden = false;
  addFileInput.value = "";

  if (data.error) {
    addError(`Couldn't add ${file.name}: ${data.error}`);
    return;
  }
  renderDocuments(data.documents);
  addNote(`Added ${data.filename}. Questions now search all ${data.documents.length} documents.`);
  for (const warning of data.warnings || []) addNote(warning);
}

// Generic starters: only a fallback, shown if the model's tailored questions
// can't be fetched. While they load, quiet placeholders hold the spot, so the
// list appears exactly once instead of showing, vanishing and reappearing.
const STARTER_QUESTIONS = [
  "What is this document about?",
  "Summarize the key points.",
  "What are the main topics covered?",
];

function makeSuggestion(question) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "suggestion";
  button.textContent = question;
  button.addEventListener("click", () => askQuestion(question));
  return button;
}

/** Hold the "Try asking" spot with placeholders, then fill it once with the
 *  model's tailored questions. Best-effort: on any error the generic starters
 *  are used instead. */
async function showSuggestions(docId) {
  const box = document.createElement("div");
  box.className = "msg suggestions";
  const label = document.createElement("p");
  label.className = "suggestions-label";
  label.textContent = "Try asking";
  box.appendChild(label);
  const list = document.createElement("div");
  list.className = "suggestion-list";
  list.setAttribute("aria-busy", "true");
  for (let i = 0; i < STARTER_QUESTIONS.length; i++) {
    const placeholder = document.createElement("div");
    placeholder.className = "suggestion-skeleton";
    placeholder.setAttribute("aria-hidden", "true");
    list.appendChild(placeholder);
  }
  box.appendChild(list);
  append(box);

  const data = await api("/api/suggestions", { id: docId });
  const tailored = !data.error && Array.isArray(data.questions) && data.questions.length > 0;
  list.replaceChildren(...(tailored ? data.questions : STARTER_QUESTIONS).map(makeSuggestion));
  list.removeAttribute("aria-busy");
  scrollToBottom();
}

/** The server dropped saved documents it could not restore: show what is left. */
async function syncDocuments() {
  const data = await api("/api/session");
  if (data.error) return;
  renderDocuments(data.documents);
  if (!data.documents.length) showView("upload");
}

async function summarizeDoc(doc, button) {
  button.disabled = true;
  const pending = addPending(`Summarizing ${doc.filename}…`);
  const data = await api("/api/summary", { id: doc.id });
  button.disabled = false;

  if (data.error) {
    showFailure(pending, data.error);
    if (data.httpStatus === 409) syncDocuments();
    return;
  }
  // The heading already says "Summary of …"; drop the model's own "Summary:" lead-in.
  const summary = data.summary.replace(/^\s*\**(?:document\s+)?summary\**:?\**\s*/i, "");
  const answer = showAnswer(pending, summary, data.sources, {
    heading: `Summary of ${data.filename}`,
    sourcesLabel: "Passages used",
    route: data.answered_by,
  });
  transcript.push({ question: `Summary of ${data.filename}`, answer: answer.text, sources: answer.sources, route: answer.route });
  const coverage = data.coverage;
  if (coverage && coverage.complete === false) {
    const parts = [];
    if (coverage.skipped_ranges?.length) parts.push(`${coverage.skipped_ranges.slice(0, 6).join(", ")} were not read (the document is long)`);
    if (coverage.failed_ranges?.length) parts.push(`${coverage.failed_ranges.slice(0, 6).join(", ")} could not be summarised`);
    addNote(`This summary is partial: ${parts.join("; ")}. Ask about those parts directly for details.`);
  }
  refreshProviders();
  syncExport();
  scrollToStart(pending);
}

async function removeDoc(id) {
  const name = loadedDocs.find((d) => d.id === id)?.filename;
  const data = await api("/api/remove", { id });
  if (data.error) {
    addError(data.error);
    return;
  }
  renderDocuments(data.documents);
  if (name) showToast(`Removed ${name}`);
}

// ---------- Events ----------

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadFirst(fileInput.files[0]);
});

addFileInput.addEventListener("change", () => {
  if (addFileInput.files.length > 0) uploadAdditional(addFileInput.files[0]);
});

// The upload labels are focusable; let Enter/Space open the file picker like a button.
function openPickerOnKey(label, input) {
  label.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
}
openPickerOnKey(addBtn, addFileInput);
openPickerOnKey(attachBtn, addFileInput);
openPickerOnKey(dropzone, fileInput);

// Drag-and-drop on the dropzone, with a visual hover state.
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) uploadFirst(file);
});

removeBtn.addEventListener("click", async () => {
  await api("/api/remove", {});
  renderDocuments([]);
  showToast("All documents removed from the server");
});

let busy = false;
let activeStream = null; // AbortController of the answer being streamed

// Question/answer pairs of this page view, for Export conversation.
const transcript = [];

// Saved chats (signed-in users): the list, and the one being shown. `currentChatId`
// is null for a new conversation; the server creates the chat from the first question.
let chats = [];
let currentChatId = null;
let resolveMe = () => {};
const meReady = new Promise((resolve) => {
  resolveMe = resolve;
});

function setBusy(on) {
  busy = on;
  document.body.classList.toggle("busy", on);
  stopBtn.hidden = !on;
  askBtn.hidden = on;
  questionInput.disabled = on;
  if (!on) {
    syncQuestionState();
    questionInput.focus({ preventScroll: true });
  }
}

/** Read a server-sent-events body, calling onEvent(name, data) per event. */
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let name = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) onEvent(name, JSON.parse(data));
    }
  }
}

/** `shown` is what appears in the conversation when it differs from the
 *  instruction sent (the follow-up buttons send a fuller instruction). */
async function askQuestion(question, shown = question) {
  question = question.trim();
  if (!question || busy) return;
  const docIds = includedDocIds();
  if (docIds.length === 0) {
    syncScope();
    return;
  }

  const asked = addUser(shown);
  questionInput.value = "";
  autoGrow();
  setBusy(true);
  const pending = addPending("Searching your documents…");

  const controller = new AbortController();
  activeStream = controller;
  const body = { question };
  if (docIds.length < loadedDocs.length) body.doc_ids = docIds;
  if (currentChatId) body.chat_id = currentChatId;

  let answer = null;
  let failed = null;
  let frame = 0;
  const render = () => {
    frame = 0;
    if (answer) updateAnswer(answer);
  };

  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok || !(response.headers.get("content-type") || "").startsWith("text/event-stream")) {
      let data = {};
      try {
        data = await response.json();
      } catch (err) {
        data = {};
      }
      failed = data.error || `The request failed (HTTP ${response.status}). Try again in a moment.`;
      if (response.status === 409) syncDocuments();
      if (response.status === 404 && currentChatId) loadChats();  // the chat is gone (deleted elsewhere)
    } else {
      await readEvents(response, (name, data) => {
        if (name === "sources") {
          answer = startAnswer(pending, data);
          answer.body.classList.add("streaming");
          scrollToStart(asked);
        } else if (name === "reasoning" && answer) {
          appendReasoning(answer, data.text);
          scrollToBottom();
        } else if (name === "chat") {
          upsertChat(data);
        } else if (name === "route" && answer) {
          answer.route = data;
        } else if (name === "token" && answer) {
          if (!answer.text) finishThinking(answer);
          answer.text += data.text;
          if (!frame) frame = requestAnimationFrame(render);
        } else if (name === "error") {
          failed = data.error;
        }
      });
    }
  } catch (err) {
    if (err.name !== "AbortError") failed = "Couldn't reach the server. Check your connection and try again.";
  }
  if (frame) cancelAnimationFrame(frame);

  const stopped = controller.signal.aborted;
  if (answer && (answer.text || stopped)) {
    finishAnswer(answer, { followUps: true, stopped });
    if (failed) addError(failed);
    else if (!stopped) {
      transcript.push({ question: shown, answer: answer.text, sources: answer.sources, route: answer.route });
      syncExport();
    }
  } else if (stopped) {
    pending.remove();
  } else {
    showFailure(pending, failed || "Something went wrong. Please try again.");
  }
  scrollToStart(asked);
  activeStream = null;
  setBusy(false);
  refreshProviders();
}

function stopAnswer() {
  if (activeStream) activeStream.abort();
}

askForm.addEventListener("submit", (e) => {
  e.preventDefault();
  askQuestion(questionInput.value);
});

// Enter submits, Shift+Enter inserts a newline.
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askForm.requestSubmit();
  }
});

// Minimal auto-grow so multi-line questions are easier to write on mobile.
function autoGrow() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${questionInput.scrollHeight}px`;
}
questionInput.addEventListener("input", autoGrow);

const charCount = document.getElementById("char-count");
const MAX_QUESTION_CHARS = Number(questionInput.maxLength) || 1000;

function syncQuestionState() {
  const length = questionInput.value.length;
  askBtn.disabled = questionInput.value.trim() === "" || (loadedDocs.length > 0 && includedDocIds().length === 0);
  charCount.hidden = length < MAX_QUESTION_CHARS * 0.8;
  charCount.textContent = `${length} / ${MAX_QUESTION_CHARS}`;
  charCount.classList.toggle("at-limit", length >= MAX_QUESTION_CHARS);
}
questionInput.addEventListener("input", syncQuestionState);

// Restore documents and conversation after a page reload (the session
// cookie outlives the page). Sources aren't kept server-side, so restored
// answers show their citations without the passage list.
/** Replace the conversation on screen with these turns (a saved chat, or a reload). */
function renderConversation(history) {
  messagesEl.replaceChildren();
  transcript.length = 0;
  let lastQuestion = null;
  for (const turn of history) {
    lastQuestion = addUser(turn.question);
    showAnswer(addNote(""), turn.answer, [], { route: turn.answered_by });
    transcript.push({ question: turn.question, answer: turn.answer, sources: [], route: turn.answered_by });
  }
  syncExport();
  return lastQuestion;
}

(async function restoreSession() {
  const [data, me] = await Promise.all([api("/api/session"), meReady]);
  const docs = !data.error && data.documents ? data.documents : [];
  if (docs.length) renderDocuments(docs);
  if (me && me.user) {
    // Signed in: the sidebar lists every saved chat; open the most recent one.
    await loadChats();
    if (chats.length && !busy) {
      await openChat(chats[0].id, { quiet: true });
      return;
    }
    if (docs.length) {
      showView("chat");
      addNote("Your documents are ready. Ask anything about them.");
    }
    return;
  }
  if (!docs.length) return;
  showView("chat");
  const lastQuestion = renderConversation(data.history || []);
  if (lastQuestion) scrollToStart(lastQuestion, false);
  else addNote("Your documents are still loaded. Ask anything about them.");
})();

// ---------- Saved chats ----------

const chatsPanel = document.getElementById("chats-panel");
const chatListEl = document.getElementById("chat-list");
const chatEmpty = document.getElementById("chat-empty");

async function loadChats() {
  const data = await api("/api/chats");
  if (data.error) return;
  chats = data.chats || [];
  renderChats();
}

function chatButton(label, title, className, svgPath) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = className;
  b.title = title;
  b.setAttribute("aria-label", title);
  if (svgPath) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", svgPath);
    svg.appendChild(path);
    b.appendChild(svg);
  } else {
    b.textContent = label;
  }
  return b;
}

function renderChats() {
  chatListEl.replaceChildren();
  chatEmpty.hidden = chats.length > 0;
  for (const chat of chats) {
    const li = document.createElement("li");
    li.className = "chat-item" + (chat.id === currentChatId ? " active" : "");
    const open = chatButton(chat.title, chat.title, "chat-open");
    open.textContent = chat.title;
    open.addEventListener("click", () => openChat(chat.id));
    const actions = document.createElement("span");
    actions.className = "chat-actions";
    const rename = chatButton("", "Rename chat", "chat-act", "M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4");
    const del = chatButton("", "Delete chat", "chat-act danger", "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3");
    rename.addEventListener("click", () => startRename(li, chat));
    del.addEventListener("click", () => confirmDelete(li, del, chat));
    actions.append(rename, del);
    li.append(open, actions);
    chatListEl.appendChild(li);
  }
}

function startRename(li, chat) {
  const input = document.createElement("input");
  input.className = "chat-rename";
  input.value = chat.title;
  input.maxLength = 80;
  input.setAttribute("aria-label", "Chat name");
  li.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (save && title && title !== chat.title) {
      const data = await api("/api/chats/rename", { id: chat.id, title });
      if (data.error) showToast(data.error);
      else chat.title = data.chat.title;
    }
    renderChats();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

function confirmDelete(li, button, chat) {
  if (!li.classList.contains("confirming")) {
    li.classList.add("confirming");
    button.textContent = "Delete?";
    button.title = "Click again to delete this chat";
    setTimeout(() => {
      if (li.isConnected && li.classList.contains("confirming")) renderChats();
    }, 3500);
    return;
  }
  deleteChat(chat);
}

async function deleteChat(chat) {
  const data = await api("/api/chats/delete", { id: chat.id });
  if (data.error) {
    showToast(data.error);
    return;
  }
  chats = chats.filter((c) => c.id !== chat.id);
  if (chat.id === currentChatId) startNewChat();
  else renderChats();
  showToast("Chat deleted");
}

async function openChat(id, { quiet = false } = {}) {
  if (busy) {
    showToast("Wait for the answer to finish first.");
    return;
  }
  const data = await api(`/api/chat?id=${encodeURIComponent(id)}`);
  if (data.error) {
    showToast(data.error);
    await loadChats();
    return;
  }
  currentChatId = data.chat.id;
  showView("chat");
  const lastQuestion = renderConversation(data.history);
  if (loadedDocs.length === 0) addNote("Add a document to keep asking in this chat.");
  if (lastQuestion) scrollToStart(lastQuestion, false);
  renderChats();
  if (!quiet) questionInput.focus();
}

/** A fresh conversation: nothing is saved until the first question is asked. */
function startNewChat() {
  if (busy) {
    showToast("Wait for the answer to finish first.");
    return;
  }
  currentChatId = null;
  messagesEl.replaceChildren();
  transcript.length = 0;
  syncExport();
  if (loadedDocs.length) {
    showView("chat");
    addNote("New chat. Ask anything about your documents.");
    questionInput.focus();
  } else {
    showView("upload");
  }
  renderChats();
}

document.getElementById("new-chat-btn").addEventListener("click", startNewChat);

/** A chat the server just created (or touched) goes to the top of the list. */
function upsertChat(chat) {
  currentChatId = chat.id;
  chats = [chat, ...chats.filter((c) => c.id !== chat.id)];
  renderChats();
}

// ---------- v2: stop, export, keyboard, drop anywhere ----------

stopBtn.addEventListener("click", stopAnswer);

function syncExport() {
  exportBtn.disabled = transcript.length === 0;
}

/** Download the conversation as Markdown, each answer with its sources. */
function exportConversation() {
  const date = new Date().toISOString().slice(0, 10);
  const lines = [
    "# Conversation — DocuLens",
    "",
    `Exported ${date}. Documents: ${loadedDocs.map((d) => d.filename).join(", ") || "none"}.`,
  ];
  for (const turn of transcript) {
    lines.push("", `## ${turn.question}`, "", plain(turn.answer).trim());
    if (turn.route) lines.push("", `_Answered by ${turn.route.provider} (${turn.route.model})_`);
    if (turn.sources.length) {
      lines.push("", "**Sources**", "");
      for (const src of turn.sources) {
        const where = src.doc ? `${src.doc}, page ${src.page}` : `Page ${src.page}`;
        const excerpt = src.text.replace(/\s+/g, " ").trim();
        lines.push(`- ${where}: "${excerpt.length > 200 ? excerpt.slice(0, 200) + "…" : excerpt}"`);
      }
    }
  }
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `conversation-${date}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast(`Exported conversation-${date}.md`);
}
exportBtn.addEventListener("click", exportConversation);

// "/" jumps to the question box (unless already typing); Esc stops an answer.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && busy) {
    stopAnswer();
    return;
  }
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
  if (e.key === "/" && !typing && document.body.dataset.view === "chat" && !questionInput.disabled) {
    e.preventDefault();
    questionInput.focus();
  }
});

// Drop a PDF anywhere on the page. On the start screen it becomes the first
// document; in the conversation it's added. Dropping elsewhere never makes the
// browser navigate away to the PDF.
let dragDepth = 0;
function hasFiles(e) {
  return [...(e.dataTransfer?.types || [])].includes("Files");
}
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  dragDepth += 1;
  if (document.body.dataset.view === "chat") document.body.classList.add("drop-target");
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) document.body.classList.remove("drop-target");
});
window.addEventListener("dragover", (e) => {
  if (hasFiles(e)) e.preventDefault();
});
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("drop-target");
  if (e.target.closest && e.target.closest("#dropzone")) return; // handled by the dropzone itself
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (document.body.dataset.view === "chat") uploadAdditional(file);
  else if (document.body.dataset.view === "upload") uploadFirst(file);
});

// Footer "Privacy" opens a short explanation of what happens to a PDF.
const privacyDialog = document.getElementById("privacy-dialog");
document.getElementById("privacy-btn").addEventListener("click", () => privacyDialog.showModal());
// A click on the backdrop (outside the dialog box) closes it too.
privacyDialog.addEventListener("click", (e) => {
  const box = privacyDialog.getBoundingClientRect();
  const outside = e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom;
  if (e.target === privacyDialog && outside) privacyDialog.close();
});

// ---------- v2.2: providers, theme, toasts, jump to latest ----------

/** Footer: which AI providers this server can use, and whether each is
 *  usable right now (green) or cooling down after a rate limit (amber). */
const providerList = document.getElementById("provider-list");
const footerProviders = document.getElementById("footer-providers");

async function refreshProviders() {
  const data = await api("/api/status");
  if (data.error || !Array.isArray(data.providers) || data.providers.length === 0) return;
  // Groq may appear twice (main + fallback model): one pill per provider,
  // ready if any of its models is.
  const byName = new Map();
  for (const p of data.providers) {
    const entry = byName.get(p.provider) || { ready: false, models: [] };
    entry.ready ||= p.state === "ready";
    entry.models.push(modelName(p.model));
    byName.set(p.provider, entry);
  }
  providerList.replaceChildren();
  for (const [name, { ready, models }] of byName) {
    const li = document.createElement("li");
    li.className = "provider";
    li.dataset.state = ready ? "ready" : "cooling";
    li.title = `${name}: ${models.join(", ")} — ${ready ? "available" : "busy, retrying soon"}`;
    const dot = document.createElement("span");
    dot.className = "provider-dot";
    dot.setAttribute("aria-hidden", "true");
    li.append(dot, name);
    li.setAttribute("aria-label", `${name}, ${ready ? "available" : "busy"}`);
    providerList.appendChild(li);
  }
  footerProviders.hidden = false;

  // The top bar mirrors overall provider health in the conversation view,
  // where the footer is hidden. Ready if any provider is; down only if the
  // status call returned providers but none is usable.
  const readyCount = [...byName.values()].filter((e) => e.ready).length;
  const topStatus = document.getElementById("topbar-status");
  if (topStatus) {
    topStatus.dataset.state = readyCount > 0 ? "ready" : "down";
    const names = [...byName].map(([n, e]) => `${n} ${e.ready ? "✓" : "busy"}`).join(", ");
    topStatus.title = `AI providers: ${names} · release notes`;
    topStatus.hidden = false;
  }
}
refreshProviders();
setInterval(() => {
  if (!document.hidden) refreshProviders();
}, 60000);

/** Light / dark / follow the system, remembered in this browser. */
const themeBtn = document.getElementById("theme-btn");
const THEME_LABELS = { system: "match system", light: "light", dark: "dark" };

function applyTheme(choice) {
  if (choice === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
  themeBtn.dataset.choice = choice;
  const label = `Theme: ${THEME_LABELS[choice]}`;
  themeBtn.setAttribute("aria-label", label);
  themeBtn.title = label;
}

applyTheme(document.documentElement.dataset.theme || "system");
themeBtn.addEventListener("click", () => {
  const order = ["system", "light", "dark"];
  const next = order[(order.indexOf(themeBtn.dataset.choice) + 1) % order.length];
  applyTheme(next);
  try {
    if (next === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", next);
  } catch (err) {
    /* not remembered, still applied */
  }
  showToast(`Theme: ${THEME_LABELS[next]}`);
});

/** Short confirmations that don't belong in the conversation. */
const toasts = document.getElementById("toasts");

function showToast(text) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = text;
  toasts.appendChild(toast);
  while (toasts.children.length > 3) toasts.firstElementChild.remove();
  setTimeout(() => {
    toast.classList.add("leaving");
    setTimeout(() => toast.remove(), 250);
  }, 2600);
}

// ---------- v3.8: Google sign-in ----------
// Sign-in is optional to look at but required for full use: guests get a small
// daily allowance (the server enforces it; this only explains it).

const signinLink = document.getElementById("signin-link");
const userChip = document.getElementById("user-chip");
const guestNote = document.getElementById("guest-note");

async function refreshMe() {
  const me = await api("/api/me");
  resolveMe(me.error ? null : me);
  if (me.error || !me.auth_enabled) return;
  document.body.dataset.auth = me.user ? "user" : "guest";
  const shelfNote = document.getElementById("shelf-note");
  shelfNote.hidden = false;
  shelfNote.textContent = me.user ? "Saved to your account for up to 30 days." : "Guest session: not saved. Sign in to keep your documents.";
  if (me.user) {
    signinLink.hidden = true;
    guestNote.hidden = true;
    document.getElementById("user-name").textContent = me.user.name;
    document.getElementById("user-avatar").textContent = (me.user.name.trim()[0] || "?").toUpperCase();
    userChip.title = me.user.email;
    userChip.hidden = false;
    chatsPanel.hidden = false;
    // The opening screen of a signed-in user is a start-a-chat screen, not the pitch.
    const first = me.user.name.trim().split(/\s+/)[0] || "there";
    document.querySelector("#upload-view .intro-title").replaceChildren(
      document.createTextNode(`Welcome back, ${first}.`),
      document.createElement("br"),
      document.createTextNode("What shall we read today?"),
    );
    document.querySelector("#upload-view .intro-body").textContent =
      "Drop a PDF, Word, text or Markdown file to start a new chat. Your chats and documents are saved in the sidebar.";
  } else {
    userChip.hidden = true;
    signinLink.hidden = false;
    firebaseConfig = me.firebase;
    loadFirebaseSdk().catch(() => {});
    const limits = me.limits || {};
    const docs = limits.max_docs === 1 ? "1 document" : `${limits.max_docs} documents`;
    document.getElementById("guest-limits").textContent =
      `${docs} at a time and ${limits.questions_per_day} questions a day.`;
    guestNote.hidden = false;
  }
}

document.getElementById("signout-btn").addEventListener("click", async () => {
  await api("/api/logout", {});
  location.reload();
});

// Firebase's browser SDK is our own bundle (/static/firebase-auth.js), loaded
// once a guest is on the page so the sign-in popup opens inside the click.
let firebaseConfig = null;
let firebaseLoading = null;

function loadFirebaseSdk() {
  if (window.DocuLensFirebase) return Promise.resolve();
  firebaseLoading ||= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/static/firebase-auth.js";
    script.onload = resolve;
    script.onerror = () => {
      firebaseLoading = null;
      reject(new Error("sdk"));
    };
    document.head.appendChild(script);
  });
  return firebaseLoading;
}

const signinDialog = document.getElementById("signin-dialog");
const googleBtn = document.getElementById("google-btn");
const signinError = document.getElementById("signin-error");

function openSignin() {
  signinError.hidden = true;
  if (!signinDialog.open) signinDialog.showModal();
  googleBtn.focus();
}

let signingIn = false;

async function signIn() {
  if (signingIn || !firebaseConfig) return;
  signingIn = true;
  googleBtn.disabled = true;
  signinError.hidden = true;
  const fail = (text) => {
    signinError.textContent = text;
    signinError.hidden = false;
  };
  try {
    await loadFirebaseSdk();
    const idToken = await window.DocuLensFirebase.signIn(firebaseConfig);
    const result = await api("/api/login", { id_token: idToken });
    if (result.error) {
      fail(result.error);
      return;
    }
    location.reload();
  } catch (err) {
    const code = err && err.code;
    if (code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request") return;
    fail(
      code === "auth/popup-blocked"
        ? "Your browser blocked the sign-in window. Allow pop-ups for this site and try again."
        : code === "auth/unauthorized-domain"
          ? "This address is not authorised for sign-in yet."
          : "Couldn't sign you in. Please try again."
    );
  } finally {
    signingIn = false;
    googleBtn.disabled = false;
  }
}

signinLink.addEventListener("click", openSignin);
document.getElementById("guest-signin").addEventListener("click", openSignin);
googleBtn.addEventListener("click", signIn);

refreshMe();

/** A round button to jump back down after scrolling up in a long conversation. */
const jumpBtn = document.getElementById("jump-btn");

function syncJump() {
  const fromBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
  jumpBtn.hidden = fromBottom < 240;
  if (!jumpBtn.hidden) jumpBtn.style.bottom = `${askForm.offsetHeight + 12}px`;
}
messagesEl.addEventListener("scroll", syncJump, { passive: true });
new ResizeObserver(syncJump).observe(messagesEl);
jumpBtn.addEventListener("click", () => {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: reducedMotion.matches ? "auto" : "smooth" });
});
