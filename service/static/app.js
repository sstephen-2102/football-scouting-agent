const ARTIFACT_FOR_PAGE = {
  archetypes: "archetypes",
  forecast: "forecast",
  experiments: "scores",
  scores: "scores",
  search: "search",
  squad: "squad",
  policy: "policy",
};

const cache = {};

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options && options.headers) },
    ...options,
  });
  let body = {};
  try {
    body = await response.json();
  } catch (_error) {
    body = {};
  }
  return { ok: response.ok, status: response.status, body };
}

function detail(body) {
  return body.detail || body;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function notReadyCard(info) {
  const row = info.detail || info;
  return (
    `<div class="empty"><h3>Not ready</h3>` +
    `<p>Missing artifact <code>${escapeHtml(row.artifact || "unknown")}</code> ` +
    `(owner: ${escapeHtml(row.owner || "unassigned")}).</p>` +
    `<p>Run <code>${escapeHtml(row.producer || "python -m src.pipeline status")}</code>, then retry.</p></div>`
  );
}

function resultCard(title, payload) {
  return (
    `<div class="result"><h3>${escapeHtml(title)}</h3>` +
    `<pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre></div>`
  );
}

async function loadCore() {
  if (!cache.meta) {
    cache.meta = (await api("/v1/meta")).body;
    cache.agent = (await api("/v1/agent")).body;
    cache.status = (await api("/v1/status")).body;
  }
  return cache;
}

function currentPage() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return hash || "dashboard";
}

function showPage(name) {
  document.querySelectorAll(".page").forEach((node) => {
    const active = node.id === `page-${name}`;
    node.classList.toggle("active", active);
    node.hidden = !active;
  });
  document.querySelectorAll(".nav-group a").forEach((link) => {
    link.classList.toggle("active", link.dataset.page === name);
  });
}

function renderDashboard(core) {
  const artifacts = core.status.artifacts || {};
  const ready = Object.values(artifacts).filter((row) => row.ready).length;
  const total = Object.keys(artifacts).length;
  document.getElementById("dash-metrics").innerHTML = [
    ["Snapshot", core.meta.snapshot_date],
    ["Artifacts ready", `${ready} / ${total}`],
    ["Min appearances", core.meta.min_appearances],
    ["Leagues", core.meta.big5.join(" · ")],
  ]
    .map(
      ([label, value]) =>
        `<div class="metric"><div class="label">${escapeHtml(label)}</div>` +
        `<div class="value">${escapeHtml(value)}</div></div>`,
    )
    .join("");

  document.getElementById("dash-artifacts").innerHTML = Object.entries(artifacts)
    .map(
      ([name, row]) =>
        `<tr><td>${escapeHtml(name)}</td>` +
        `<td class="${row.ready ? "ready" : "missing"}">${row.ready ? "ready" : "missing"}</td>` +
        `<td>${escapeHtml(row.owner)}</td></tr>`,
    )
    .join("");

  document.getElementById("dash-disclaimer").innerHTML =
    `<h3>Methodology notes</h3><p>${escapeHtml(core.agent.disclaimer)}</p>` +
    `<p>Negative results are in scope. This lab does not recommend buying or selling players.</p>`;
}

function renderAgent(core) {
  const peas = core.agent.peas || {};
  document.getElementById("peas-cards").innerHTML = Object.entries(peas)
    .map(
      ([key, value]) =>
        `<div class="card"><h2>${escapeHtml(key)}</h2><p>${escapeHtml(value)}</p></div>`,
    )
    .join("");
  document.getElementById("dimension-rows").innerHTML = Object.entries(
    core.agent.dimensions || {},
  )
    .map(
      ([axis, text]) =>
        `<tr><td>${escapeHtml(axis)}</td><td>${escapeHtml(text)}</td></tr>`,
    )
    .join("");
}

function renderData(core) {
  document.getElementById("data-metrics").innerHTML = [
    ["Dataset", core.meta.dataset],
    ["Train seasons", core.meta.train_seasons.join(", ")],
    ["Validation", core.meta.val_seasons.join(", ")],
    ["Test", core.meta.test_seasons.join(", ")],
  ]
    .map(
      ([label, value]) =>
        `<div class="metric"><div class="label">${escapeHtml(label)}</div>` +
        `<div class="value">${escapeHtml(value)}</div></div>`,
    )
    .join("");
}

function artifactBanner(page, core) {
  const name = ARTIFACT_FOR_PAGE[page];
  if (!name) return;
  const row = core.status.artifacts[name];
  const host = document.getElementById(`${page}-status`);
  if (!host || !row) return;
  if (row.ready) {
    host.innerHTML = `<div class="result"><h3>Artifact ready</h3><p><code>${escapeHtml(row.path)}</code></p></div>`;
    return;
  }
  host.innerHTML = notReadyCard({ ...row, artifact: name });
}

function parseIds(raw) {
  return raw
    .split(/[,\s]+/)
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isInteger(item) && item > 0);
}

function wireForms() {
  document.getElementById("scores-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const playerId = event.target.player_id.value;
    const response = await api(`/v1/scores/${playerId}`);
    const host = document.getElementById("scores-result");
    host.innerHTML = response.ok
      ? resultCard("Score", response.body)
      : response.status === 503
        ? notReadyCard(detail(response.body))
        : resultCard("Error", response.body);
  });

  document.getElementById("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const ids = parseIds(event.target.shortlist.value);
    const response = await api("/v1/search/xi", {
      method: "POST",
      body: JSON.stringify({
        club_id: Number(event.target.club_id.value),
        season: Number(event.target.season.value),
        shortlist_player_ids: ids,
      }),
    });
    const host = document.getElementById("search-result");
    host.innerHTML = response.ok
      ? resultCard("XI search", response.body)
      : response.status === 503
        ? notReadyCard(detail(response.body))
        : resultCard("Error", response.body);
  });

  document.getElementById("squad-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const response = await api("/v1/squad/allocate", {
      method: "POST",
      body: JSON.stringify({
        club_id: Number(event.target.club_id.value),
        season: Number(event.target.season.value),
        budget_eur: Number(event.target.budget_eur.value),
      }),
    });
    const host = document.getElementById("squad-result");
    host.innerHTML = response.ok
      ? resultCard("Allocation", response.body)
      : response.status === 503
        ? notReadyCard(detail(response.body))
        : resultCard("Error", response.body);
  });

  document.getElementById("policy-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const response = await api("/v1/policy/bid", {
      method: "POST",
      body: JSON.stringify({
        player_id: Number(event.target.player_id.value),
        remaining_budget_eur: Number(event.target.remaining_budget_eur.value),
        position_hole: event.target.position_hole.value,
      }),
    });
    const host = document.getElementById("policy-result");
    host.innerHTML = response.ok
      ? resultCard("Policy", response.body)
      : response.status === 503
        ? notReadyCard(detail(response.body))
        : resultCard("Error", response.body);
  });
}

async function render() {
  const page = currentPage();
  showPage(page);
  const core = await loadCore();
  renderDashboard(core);
  renderAgent(core);
  renderData(core);
  Object.keys(ARTIFACT_FOR_PAGE).forEach((name) => artifactBanner(name, core));
}

window.addEventListener("hashchange", render);
wireForms();
render();
