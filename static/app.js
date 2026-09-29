/**
 * AI Document Assistant -- frontend logic.
 *
 * Talks to the FastAPI backend (main.py) at /api/ingest, /api/ask,
 * /api/summary, /api/remove and /api/session. All dynamic content
 * (questions, answers, document text, filenames) is inserted with
 * textContent, never innerHTML, so nothing from a document or the model can
 * inject markup into the page.
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
    return { error: "Network error. Please check your connection and try again." };
  }

  let data;
  try {
    data = await response.json();
  } catch (err) {
    if (response.status === 413) return { error: "File is too large. The limit is 25MB." };
    return { error: `The server returned an unexpected response (HTTP ${response.status}). Please try again.` };
  }
  if (!response.ok && !data.error) {
    data.error = `Request failed (HTTP ${response.status}). Please try again.`;
  }
  return data;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Models often wrap key phrases in **bold**. Text is shown with textContent
// (no markdown rendering, by design), so drop the markers instead of showing
// literal asterisks.
function plain(text) {
  return text.replace(/\*\*(.+?)\*\*/g, "$1");
}

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = plain(text);
  messagesEl.appendChild(el);
  scrollToBottom();
  return el;
}

function addSources(sources, label = "Sources") {
  if (!sources || sources.length === 0) return;

  // Only name the document when more than one is loaded -- otherwise it's noise.
  const multiDoc = new Set(sources.map((s) => s.doc)).size > 1 || docList.children.length > 1;

  const details = document.createElement("details");
  details.className = "message-sources";

  const summary = document.createElement("summary");
  summary.textContent = `${label} (${sources.length})`;
  details.appendChild(summary);

  for (const src of sources) {
    const node = sourceItemTemplate.content.cloneNode(true);
    node.querySelector(".source-page").textContent =
      multiDoc && src.doc ? `${src.doc} · Page ${src.page}` : `Page ${src.page}`;
    node.querySelector(".source-score").textContent =
      typeof src.score === "number" ? `similarity: ${src.score.toFixed(3)}` : "";
    node.querySelector(".source-text").textContent = src.text;
    details.appendChild(node);
  }

  messagesEl.appendChild(details);
  scrollToBottom();
}

function renderDocuments(documents) {
  docList.replaceChildren();
  for (const doc of documents) {
    const node = docItemTemplate.content.cloneNode(true);
    node.querySelector(".doc-name").textContent = doc.filename;
    node.querySelector(".doc-name").title = doc.filename;
    node.querySelector(".doc-meta").textContent = `${doc.num_pages} pages · ${doc.num_chunks} chunks`;
    node.querySelector(".doc-summary").addEventListener("click", (e) => summarizeDoc(doc, e.currentTarget));
    node.querySelector(".doc-remove").addEventListener("click", () => removeDoc(doc.id));
    docList.appendChild(node);
  }
  if (documents.length === 0) {
    messagesEl.replaceChildren();
    fileInput.value = "";
    showView("upload");
  }
}

function validateFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) return "Please upload a PDF file.";
  if (file.size > MAX_UPLOAD_BYTES) return "File is too large. The limit is 25MB.";
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
  renderDocuments(data.documents);
  addMessage("assistant", `"${data.filename}" is ready — ask me anything about it, or add more PDFs to search across them.`);
  questionInput.value = "";
  showView("chat");
  questionInput.focus();
}

// Further documents: indexed inline, conversation stays.
async function uploadAdditional(file) {
  const problem = validateFile(file);
  if (problem) {
    addMessage("error", problem);
    return;
  }

  addStatus.hidden = false;
  addBtn.hidden = true;
  const data = await ingest(file);
  addStatus.hidden = true;
  addBtn.hidden = false;
  addFileInput.value = "";

  if (data.error) {
    addMessage("error", `Couldn't add "${file.name}": ${data.error}`);
    return;
  }
  renderDocuments(data.documents);
  addMessage("assistant", `Added "${data.filename}". Questions now search all ${data.documents.length} documents.`);
}

async function summarizeDoc(doc, button) {
  button.disabled = true;
  const pending = addMessage("assistant", `Summarizing "${doc.filename}"…`);
  const data = await api("/api/summary", { id: doc.id });
  button.disabled = false;

  if (data.error) {
    pending.textContent = data.error;
    pending.className = "message error";
    return;
  }
  pending.textContent = plain(`Summary of "${data.filename}":\n\n${data.summary}`);
  addSources(data.sources, "Excerpts used");
}

async function removeDoc(id) {
  const data = await api("/api/remove", { id });
  if (data.error) {
    addMessage("error", data.error);
    return;
  }
  renderDocuments(data.documents);
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadFirst(fileInput.files[0]);
});

addFileInput.addEventListener("change", () => {
  if (addFileInput.files.length > 0) uploadAdditional(addFileInput.files[0]);
});

// The "+ Add PDF" label is focusable; let Enter/Space open the picker like a button.
addBtn.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    addFileInput.click();
  }
});

// Drag-and-drop support on the dropzone label, with a visual hover state.
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

  addMessage("user", question);
  questionInput.value = "";
  autoGrow();
  askBtn.disabled = true;
  questionInput.disabled = true;

  const thinking = addMessage("assistant", "Thinking…");
  const data = await api("/api/ask", { question });

  if (data.error) {
    thinking.textContent = data.error;
    thinking.className = "message error";
  } else {
    thinking.textContent = plain(data.answer);
    addSources(data.sources);
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
// cookie outlives the page).
(async function restoreSession() {
  const data = await api("/api/session");
  if (data.error || !data.documents || data.documents.length === 0) return;
  renderDocuments(data.documents);
  for (const turn of data.history) {
    addMessage("user", turn.question);
    addMessage("assistant", turn.answer);
  }
  if (data.history.length === 0) addMessage("assistant", "Documents restored — ask me anything about them.");
  showView("chat");
})();
