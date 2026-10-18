/* V12 Engines dashboard: shared helpers.
 *
 * Every call goes to a RELATIVE path (/api/...). nginx forwards it to the scoring API and adds the
 * team key server-side, so this code never holds a key and needs no host or port.
 * When the API cannot be reached, pages fall back to files in ./samples/ and say so on screen.
 */
(function () {
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (x, d = 1) => (x == null || isNaN(x) ? '–' : (x * 100).toFixed(d) + '%');
  const int = (n) => (n == null ? '–' : Number(n).toLocaleString('en-US'));
  const num = (x, d = 2) => (x == null || isNaN(x) ? '–' : Number(x).toFixed(d));

  async function api(path) {
    const r = await fetch('/api' + path, { headers: { Accept: 'application/json' }, cache: 'no-store' });
    if (!r.ok) {
      let detail = '';
      try { const j = await r.json(); detail = j.error ? `${j.error.code}: ${j.error.message}` : ''; } catch (e) { /* not JSON */ }
      throw new Error(`/api${path} returned HTTP ${r.status}${detail ? ' (' + detail + ')' : ''}`);
    }
    return r.json();
  }

  /** Live data when possible, otherwise the sample file; returns {data, live, error}. */
  async function apiOrSample(path, sample) {
    try {
      return { data: await api(path), live: true, error: null };
    } catch (e) {
      try {
        const r = await fetch(sample, { cache: 'no-store' });
        return { data: await r.json(), live: false, error: e.message };
      } catch (e2) {
        return { data: null, live: false, error: e.message };
      }
    }
  }

  async function post(path) {
    const r = await fetch('/api' + path, { method: 'POST', headers: { Accept: 'application/json' } });
    if (!r.ok) throw new Error(`/api${path} returned HTTP ${r.status}`);
    return r.json();
  }

  /** Static file next to the pages (e.g. evidence.json written on the server); null if absent. */
  async function staticJson(path) {
    try {
      const r = await fetch(path, { cache: 'no-store' });
      if (!r.ok) return null;
      const t = r.headers.get('content-type') || '';
      return t.includes('json') ? r.json() : JSON.parse(await r.text());
    } catch (e) { return null; }
  }

  const FEATURE_LABELS = {
    prior_excess_vendor: 'Vendor: recent late rate vs overall',
    prior_excess_country: 'Country: recent late rate vs overall',
    prior_excess_fulfillment_route: 'Route: recent late rate vs overall',
    planned_lead_time_days: 'Planned lead time',
    quantity: 'Order quantity',
    lead_time_missing: 'Lead time not recorded',
    prior_known_vendor: 'Vendor has recent history',
    prior_known_country: 'Country has recent history',
    prior_known_fulfillment_route: 'Route has recent history',
  };
  const featureLabel = (f) => FEATURE_LABELS[f] || f;

  /** Header with tabs and two status pills. kind: ok | warn | bad. */
  function header(active, pills) {
    const tabs = [['index.html', 'Live'], ['model.html', 'Model'], ['platform.html', 'Platform']];
    const el = document.getElementById('top');
    el.innerHTML = `
      <div class="top-in">
        <div class="top-row">
          <div><div class="brand">Delivery-Delay Early Warning</div>
            <div class="brand-sub">Health commodity shipments scored at order time</div></div>
          <div class="pills" id="pills"></div>
        </div>
        <nav class="tabs" aria-label="Dashboard sections">
          ${tabs.map(([href, name]) => `<a href="${href}"${name === active ? ' aria-current="page"' : ''}>${name}</a>`).join('')}
        </nav>
      </div>`;
    setPills(pills || []);
  }

  function setPills(pills) {
    const box = document.getElementById('pills');
    if (!box) return;
    box.innerHTML = pills.map((p) =>
      `<div class="pill${p.mono ? ' mono' : ''}" title="${esc(p.title || '')}">` +
      (p.kind ? `<span class="dot ${esc(p.kind)}"></span>` : '') + `<span>${esc(p.text)}</span></div>`).join('');
  }

  /** Common pills from /api/health. */
  function healthPills(h, live) {
    if (!live || !h || h.status !== 'ok') {
      return [{ kind: 'bad', text: 'API unreachable · sample data' }];
    }
    const arch = h.machine ? `${h.machine}` : 'architecture ?';
    return [
      { kind: 'ok', text: `Live API · ${h.backend === 'snapml' ? 'Snap ML' : h.backend} · ${h.model_version}` },
      { mono: true, text: `${arch === 's390x' ? 'LinuxONE · s390x' : arch}` },
    ];
  }

  window.V12 = { esc, pct, int, num, api, apiOrSample, post, staticJson, featureLabel, header, setPills, healthPills };
})();
