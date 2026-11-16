// launchcontainers.core QC page (from QC-registration). /api/index lists the items
// (item × sub × ses × run); /api/item gives one item's layers and slice
// positions (the server renders it first when it can and it is missing or
// stale). Each tile stacks the item's layer PNGs; a layer toggle only flips a
// class. Ratings go to /api/qc, one TSV per rater (qc/qc_rater-<name>.tsv).

const $ = (id) => document.getElementById(id);
const VIEWS = ["coronal", "sagittal", "axial"];
const ORIENT = { coronal: ["L", "R", "S", "I"], sagittal: ["P", "A", "S", "I"], axial: ["L", "R", "A", "P"] };
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

const S = {
  idx: null,          // /api/index
  item: null,
  list: [],           // entries of the current item, in sub / ses / run order
  pos: 0,
  rater: "",
  ratings: new Map(), // key -> row
  layers: {},         // layer key -> on (remembered across items)
  filter: "all",
  meta: new Map(),    // key -> Promise of /api/item
  token: 0,
};
const key = (it) => [it.item, it.sub, it.ses, it.run || ""].join("|");
const cur = () => S.list[S.pos];
const runLabel = (r) => (!r ? "" : /^\d+$/.test(r) ? `run-${r}` : r);
const label = (it) => `sub-${it.sub} ses-${it.ses}${it.run ? " " + runLabel(it.run) : ""}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function status(msg, kind = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = kind;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch {}
  if (!res.ok) throw new Error((body && body.error) || `${res.status} ${res.statusText}`);
  return body;
}

function getMeta(it) {
  const k = key(it);
  if (!S.meta.has(k)) {
    const q = new URLSearchParams({ item: it.item, sub: it.sub, ses: it.ses, run: it.run || "" });
    const p = api(`api/item?${q}`);
    p.catch(() => S.meta.delete(k)); // retry next time
    S.meta.set(k, p);
  }
  return S.meta.get(k);
}

const src = (m, view, i, layer) => `png/${m.dir}/${view}-${i}_${layer}.png?v=${m.fingerprint}`;

function tilesHTML(m) {
  return VIEWS.map((view) => {
    const [l, r, t, b] = ORIENT[view];
    const tiles = m.cuts_mm[view].map((mm, i) =>
      `<div class="tile">` +
      m.layers.map((ly, n) => `<img class="layer${n === 0 ? " first" : ""}" data-k="${esc(ly.key)}" src="${src(m, view, i, ly.key)}">`).join("") +
      `<span class="o l">${l}</span><span class="o r">${r}</span><span class="o t">${t}</span>` +
      `<span class="o b">${b}</span><span class="mm">${mm.toFixed(1)}</span></div>`).join("");
    return `<div class="trow"><div class="lab">${view}</div>${tiles}</div>`;
  }).join("");
}

function preload(it) {
  if (!it) return;
  getMeta(it).then((m) => {
    for (const view of VIEWS) m.cuts_mm[view].forEach((_, i) => {
      for (const ly of m.layers) new Image().src = src(m, view, i, ly.key);
    });
  }).catch(() => {});
}

// ------------------------------------------------------------ header selects
function fill(el, values, current, fmt) {
  el.innerHTML = values.map((v) => `<option value="${esc(v)}"${v === current ? " selected" : ""}>${esc(fmt(v))}</option>`).join("");
}
const uniq = (xs) => [...new Set(xs)];

function fillSelects() {
  const it = cur();
  fill($("item"), uniq(S.idx.items.map((x) => x.item)), S.item, (v) => v);
  fill($("sub"), uniq(S.list.map((x) => x.sub)), it.sub, (v) => `sub-${v}`);
  fill($("ses"), uniq(S.list.filter((x) => x.sub === it.sub).map((x) => x.ses)), it.ses, (v) => `ses-${v}`);
  const runs = S.list.filter((x) => x.sub === it.sub && x.ses === it.ses).map((x) => x.run || "");
  fill($("run"), runs, it.run || "", (v) => runLabel(v) || "—");
  $("run").disabled = runs.length < 2 && !runs[0];
}

// ------------------------------------------------------------ sidebar
function showVerdict() {
  const r = S.ratings.get(key(cur()));
  $("ok-true").classList.toggle("on", r?.ok === "True");
  $("ok-false").classList.toggle("on", r?.ok === "False");
  $("comment").value = r?.comment || "";
  const done = S.list.filter((x) => S.ratings.has(key(x))).length;
  $("progress").textContent = `${done} / ${S.list.length} rated`;
}

function renderList() {
  const rows = S.list.map((it, i) => ({ it, i, r: S.ratings.get(key(it)) })).filter(({ r }) =>
    S.filter === "all" || (S.filter === "todo" ? !r : r?.ok === S.filter));
  $("runs").innerHTML = rows.map(({ it, i, r }) =>
    `<li data-i="${i}"${i === S.pos ? ' class="active"' : ""}>` +
    `<span class="mark ${r ? r.ok : ""}">${r ? (r.ok === "True" ? "✓" : "✗") : "·"}</span>` +
    `<span class="name">${esc(label(it))}</span>` +
    (r?.comment ? `<span class="note" title="${esc(r.comment)}">💬</span>` : "") + "</li>").join("")
    || '<li class="dim">none</li>';
  $("runs").querySelector("li.active")?.scrollIntoView({ block: "nearest" });
}

function renderLayers(m) {
  for (const ly of m.layers) if (!(ly.key in S.layers)) S.layers[ly.key] = ly.show !== false;
  $("layers").innerHTML = m.layers.map((ly, n) =>
    `<label class="row"><input type="checkbox" data-k="${esc(ly.key)}"${S.layers[ly.key] ? " checked" : ""}> ${esc(ly.label)} <kbd>${n + 1}</kbd></label>`).join("");
  applyLayers();
}
function applyLayers() {
  document.querySelectorAll("#tiles img[data-k]").forEach((el) => el.classList.toggle("off", !S.layers[el.dataset.k]));
  $("layers").querySelectorAll("input").forEach((el) => { el.checked = !!S.layers[el.dataset.k]; });
}
function toggleLayer(n) {
  const ly = S.shown?.layers[n];
  if (ly) { S.layers[ly.key] = !S.layers[ly.key]; applyLayers(); }
}

// ------------------------------------------------------------ main view
async function show() {
  const it = cur();
  const token = ++S.token;
  fillSelects();
  showVerdict();
  renderList();
  location.hash = new URLSearchParams({ i: it.item, s: it.sub, e: it.ses, r: it.run || "" }).toString();
  $("where").textContent = `${S.pos + 1} / ${S.list.length} · ${S.idx.analysis}`;
  $("title").textContent = `${label(it)} · ${it.item}`;
  if (!S.meta.has(key(it))) $("tiles").innerHTML = '<div class="busy">loading (rendering if this one has no PNGs yet)…</div>';
  let m;
  try {
    m = await getMeta(it);
  } catch (e) {
    if (token === S.token) { $("tiles").innerHTML = ""; status(`ERROR ${e.message}`, "err"); }
    return;
  }
  if (token !== S.token) return;
  if (m.rendered_now) status("rendered now", "ok");
  S.shown = m;
  $("question").textContent = m.question || S.idx.question || `Does ${m.item} look right?`;
  $("title").innerHTML = esc(`${label(m)} · ${m.item}`) +
    (m.resampled?.length ? ` <span class="warn">(${esc(m.resampled.join(", "))} resampled to the base grid for display)</span>` : "");
  const inp = m.inputs || {};
  $("title").title = Object.entries(inp).map(([k, v]) => `${k}: ${v.path}`).join("\n");
  $("tiles").innerHTML = tilesHTML(m);
  renderLayers(m);
  preload(S.list[S.pos + 1]);
  preload(S.list[S.pos - 1]);
}

function go(pos) {
  if (pos < 0 || pos >= S.list.length) return;
  S.pos = pos;
  show();
}
function stepSubject(d) {
  const subs = uniq(S.list.map((x) => x.sub));
  const next = subs[subs.indexOf(cur().sub) + d];
  if (next) go(S.list.findIndex((x) => x.sub === next));
}
function nextUnrated() {
  for (let n = 1; n <= S.list.length; n++) {
    const i = (S.pos + n) % S.list.length;
    if (!S.ratings.has(key(S.list[i]))) return go(i);
  }
  status("everything of this item is rated", "ok");
}
function setItem(item, keep) {
  S.item = item;
  S.list = S.idx.items.filter((x) => x.item === item);
  const i = keep ? S.list.findIndex((x) => x.sub === keep.sub && x.ses === keep.ses && (x.run || "") === (keep.run || "")) : -1;
  S.pos = Math.max(i, 0);
}

// ------------------------------------------------------------ ratings
async function loadRatings() {
  S.ratings = new Map();
  $("rater-file").textContent = S.rater ? `saved to ${S.idx.analysis_dir}/qc/qc_rater-${S.rater}.tsv` : "type your name to start rating";
  if (S.rater) {
    try {
      const { rows } = await api(`api/qc?rater=${encodeURIComponent(S.rater)}`);
      for (const r of rows) S.ratings.set([r.item, r.sub, r.ses, r.run || ""].join("|"), r);
    } catch (e) { status(`ERROR ${e.message}`, "err"); }
  }
  showVerdict();
  renderList();
}

async function rate(ok, advance = true) {
  if (!S.rater) {
    status("type your name in Rater first", "err");
    $("rater").classList.add("missing");
    $("rater").focus();
    return;
  }
  const it = cur();
  try {
    const row = await api("api/qc", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rater: S.rater, item: it.item, sub: it.sub, ses: it.ses, run: it.run || "", ok, comment: $("comment").value }),
    });
    S.ratings.set(key(it), row);
    status(`saved ${row.ok} · ${label(it)} · ${it.item}`, "ok");
  } catch (e) {
    status(`ERROR not saved: ${e.message}`, "err");
    return;
  }
  if (advance && $("advance").checked && S.pos + 1 < S.list.length && it === cur()) go(S.pos + 1);
  else { showVerdict(); renderList(); }
}

// ------------------------------------------------------------ wiring
function wire() {
  $("item").onchange = (e) => { setItem(e.target.value, cur()); show(); };
  $("sub").onchange = (e) => go(S.list.findIndex((x) => x.sub === e.target.value));
  $("ses").onchange = (e) => go(S.list.findIndex((x) => x.sub === cur().sub && x.ses === e.target.value));
  $("run").onchange = (e) => go(S.list.findIndex((x) => x.sub === cur().sub && x.ses === cur().ses && (x.run || "") === e.target.value));
  $("prev").onclick = () => go(S.pos - 1);
  $("next").onclick = () => go(S.pos + 1);
  $("ok-true").onclick = () => rate(true);
  $("ok-false").onclick = () => rate(false);
  $("next-unrated").onclick = nextUnrated;
  $("runs").onclick = (e) => { const li = e.target.closest("li[data-i]"); if (li) go(+li.dataset.i); };
  $("filter").onclick = (e) => {
    const b = e.target.closest("button[data-f]");
    if (!b) return;
    S.filter = b.dataset.f;
    $("filter").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    renderList();
  };
  $("layers").onchange = (e) => { S.layers[e.target.dataset.k] = e.target.checked; applyLayers(); };
  $("rater").onchange = () => {
    S.rater = $("rater").value.trim();
    $("rater").classList.remove("missing");
    store.set("qcr.rater", S.rater);
    loadRatings();
  };
  $("comment").onchange = () => {
    const r = S.ratings.get(key(cur()));
    if (r) rate(r.ok === "True", false);
    else status("comment kept until you press T or F", "");
  };
  $("sidebar-toggle").onclick = () => document.body.classList.toggle("no-aside");
  $("theme").onclick = () => {
    const t = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = t;
    store.set("qcr.theme", t);
  };

  document.addEventListener("keydown", (e) => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (k === "ArrowRight") go(S.pos + 1);
    else if (k === "ArrowLeft") go(S.pos - 1);
    else if (k === "ArrowDown") stepSubject(1);
    else if (k === "ArrowUp") stepSubject(-1);
    else if (k === "t" || k === "T") rate(true);
    else if (k === "f" || k === "F") rate(false);
    else if (k === "n" || k === "N") nextUnrated();
    else if (k === "c" || k === "C") $("comment").focus();
    else if (k >= "1" && k <= "9") toggleLayer(+k - 1);
    else if (k === "\\") document.body.classList.toggle("no-aside");
    else return;
    e.preventDefault();
  });
}

async function init() {
  try {
    S.idx = await api("api/index");
  } catch (e) {
    status(`ERROR ${e.message}`, "err");
    return;
  }
  document.title = `QC · ${S.idx.analysis}`;
  $("analysis").textContent = S.idx.analysis;
  wire();
  if (!S.idx.items.length) { $("where").textContent = "nothing rendered yet: run `lc qc -w <analysis dir>` first"; return; }
  const h = new URLSearchParams(location.hash.slice(1));
  const items = uniq(S.idx.items.map((x) => x.item));
  setItem(items.includes(h.get("i")) ? h.get("i") : items[0], { sub: h.get("s"), ses: h.get("e"), run: h.get("r") });
  S.rater = store.get("qcr.rater") || "";
  $("rater").value = S.rater;
  await loadRatings();
  show();
}

init();
