/**
 * AI Document Assistant -- frontend logic.
 *
 * Talks to the FastAPI backend (main.py) at /api/ingest, /api/ask,
 * /api/summary, /api/remove and /api/session. All dynamic content
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

const docItemTemplate = document.getElementById("doc-item-template");
const sourceItemTemplate = document.getElementById("source-item-template");

function showView(view) {
  document.body.dataset.view = view;
  uploadView.hidden = view !== "upload";
  indexingView.hidden = view !== "indexing";
  chatView.hidden = view !== "chat";
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
  return data;
}

// ---------- Messages ----------

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function append(el) {
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
  el.textContent = text;
  return append(el);
}

// Models often wrap key phrases in **bold**. Text is shown as plain text (no
// markdown rendering, by design), so drop the markers instead of showing
// literal asterisks.
function plain(text) {
  return text.replace(/\*\*(.+?)\*\*/g, "$1");
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

/**
 * Fill `container` with the answer text, replacing citations with page tabs.
 * A tab whose page is among this answer's sources becomes a button that
 * opens the sources and highlights that passage.
 */
function renderAnswerText(container, text, sources, details) {
  text = plain(text);
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const inner = match[1] ?? match[2] ?? match[3];
    const { doc, pages } = citedPages(inner);
    if (pages.length === 0) continue;

    container.append(text.slice(last, match.index).replace(/\s+$/, ""));
    for (const page of pages) {
      const slip = findSlip(details, page, doc);
      const tab = document.createElement(slip ? "button" : "span");
      tab.className = "cite";
      tab.textContent = `p. ${page}`;
      tab.title = doc ? `${doc}, page ${page}` : `Page ${page}`;
      if (slip) {
        tab.type = "button";
        tab.setAttribute("aria-label", `Show the passage from ${tab.title}`);
        tab.addEventListener("click", () => markSlip(details, slip));
      }
      container.append(tab);
    }
    last = match.index + match[0].length;
  }
  container.append(text.slice(last));
}

function findSlip(details, page, doc) {
  if (!details) return null;
  const slips = [...details.querySelectorAll(".slip")];
  return (
    slips.find((s) => Number(s.dataset.page) === page && (!doc || s.dataset.doc === doc)) ||
    slips.find((s) => Number(s.dataset.page) === page) ||
    null
  );
}

function markSlip(details, slip) {
  details.open = true;
  for (const s of details.querySelectorAll(".slip.marked")) s.classList.remove("marked");
  // Restart the highlighter animation even when the same tab is clicked twice.
  void slip.offsetWidth;
  slip.classList.add("marked");
  slip.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

function buildSources(sources, label) {
  if (!sources || sources.length === 0) return null;

  // Only name the document when more than one is loaded -- otherwise it's noise.
  const multiDoc = new Set(sources.map((s) => s.doc)).size > 1 || docList.children.length > 1;

  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `${label} (${sources.length})`;
  details.appendChild(summary);

  const list = document.createElement("ol");
  list.className = "slips";
  for (const src of sources) {
    const node = sourceItemTemplate.content.cloneNode(true);
    const slip = node.querySelector(".slip");
    slip.dataset.page = src.page;
    slip.dataset.doc = src.doc || "";
    node.querySelector(".slip-page").textContent =
      multiDoc && src.doc ? `${src.doc}, page ${src.page}` : `Page ${src.page}`;
    node.querySelector(".slip-score").textContent =
      typeof src.score === "number" ? `similarity ${src.score.toFixed(2)}` : "";
    // PDF extraction keeps the page's hard line breaks; rejoin them so the
    // passage reads as prose (blank lines between paragraphs are kept).
    node.querySelector(".slip-text").textContent = src.text.replace(/(?<!\n)\n(?!\n)/g, " ");
    list.appendChild(node);
  }
  details.appendChild(list);
  return details;
}

/** Turn a pending message into a finished answer (or add a new one). */
function showAnswer(el, text, sources, { heading, sourcesLabel = "Sources" } = {}) {
  el.className = "msg msg-assistant";
  el.replaceChildren();

  if (heading) {
    const h = document.createElement("p");
    h.className = "msg-heading";
    h.textContent = heading;
    el.appendChild(h);
  }

  const details = buildSources(sources, sourcesLabel);
  const body = document.createElement("p");
  body.className = "msg-text";
  renderAnswerText(body, text, sources, details);
  el.appendChild(body);
  if (details) el.appendChild(details);
  scrollToBottom();
}

function showFailure(el, text) {
  el.className = "msg msg-error";
  el.setAttribute("role", "alert");
  el.textContent = text;
}

// ---------- Documents ----------

function renderDocuments(documents) {
  docList.replaceChildren();
  for (const doc of documents) {
    const node = docItemTemplate.content.cloneNode(true);
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
  if (documents.length === 0) {
    messagesEl.replaceChildren();
    fileInput.value = "";
    showView("upload");
  }
}

function validateFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) return "That isn't a PDF. Choose a file ending in .pdf.";
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
}

async function summarizeDoc(doc, button) {
  button.disabled = true;
  const pending = addPending(`Summarizing ${doc.filename}…`);
  const data = await api("/api/summary", { id: doc.id });
  button.disabled = false;

  if (data.error) {
    showFailure(pending, data.error);
    return;
  }
  // The heading already says "Summary of …"; drop the model's own "Summary:" lead-in.
  const summary = data.summary.replace(/^\s*\**summary\**:?\**\s*/i, "");
  showAnswer(pending, summary, data.sources, {
    heading: `Summary of ${data.filename}`,
    sourcesLabel: "Passages used",
  });
}

async function removeDoc(id) {
  const data = await api("/api/remove", { id });
  if (data.error) {
    addError(data.error);
    return;
  }
  renderDocuments(data.documents);
}

// ---------- Events ----------

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadFirst(fileInput.files[0]);
});

addFileInput.addEventListener("change", () => {
  if (addFileInput.files.length > 0) uploadAdditional(addFileInput.files[0]);
});

// The "Add another PDF" label is focusable; let Enter/Space open the picker like a button.
addBtn.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    addFileInput.click();
  }
});

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
});

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addUser(question);
  questionInput.value = "";
  autoGrow();
  askBtn.disabled = true;
  questionInput.disabled = true;

  const pending = addPending("Searching your documents…");
  const data = await api("/api/ask", { question });

  if (data.error) {
    showFailure(pending, data.error);
  } else {
    showAnswer(pending, data.answer, data.sources);
  }

  askBtn.disabled = false;
  questionInput.disabled = false;
  questionInput.focus();
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

// Restore documents and conversation after a page reload (the session
// cookie outlives the page). Sources aren't kept server-side, so restored
// answers show their citations without the passage list.
(async function restoreSession() {
  const data = await api("/api/session");
  if (data.error || !data.documents || data.documents.length === 0) return;
  showView("chat");
  renderDocuments(data.documents);
  for (const turn of data.history) {
    addUser(turn.question);
    showAnswer(addNote(""), turn.answer, []);
  }
  if (data.history.length === 0) addNote("Your documents are still loaded. Ask anything about them.");
})();
