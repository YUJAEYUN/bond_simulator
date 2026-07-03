const CONFIG = {
  pairs: {
    sp500: { equityFile: 'data/sp500.csv', bondFile: 'data/us10y_yield.csv', label: 'S&P500 / 미국채 10Y', color: '#5b9cf6' },
    kospi: { equityFile: 'data/kospi.csv', bondFile: 'data/kr10y_yield.csv', label: 'KOSPI / 한국채 10Y', color: '#5b9cf6' },
  },
  weightGrid: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
  gridColors: { 1.0: '#1f4e8c', 0.8: '#3d7ab5', 0.6: '#6aa6d8', 0.4: '#f2b53c', 0.2: '#e07b39', 0.0: '#c0392b' },
  maturityYears: 10,
};

let state = {
  pair: 'sp500',
  weight: 0.6,
  rebalanceFreq: 'annual',
  threshold: -0.10,
  crisisPct: 0.05,
  selectedEpisode: 0,
};

let dataByPair = {};   // pair -> { aligned }
let charts = {};

// ─── CSV / parsing ──────────────────────────────────────────────────────────
function fetchCSV(path) {
  return fetch(path).then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${path}`);
    return r.text();
  }).then(text => new Promise((res, rej) => {
    Papa.parse(text, { header: true, skipEmptyLines: true, complete: r => res(r.data), error: rej });
  }));
}

function parseSeries(rows) {
  return rows.map(r => ({
    date: new Date(r.date + 'T00:00:00Z'),
    close: parseFloat(r.close),
    changePct: parseFloat(r.change_pct),
  })).filter(d => !isNaN(d.date.getTime()) && !isNaN(d.close) && !isNaN(d.changePct))
     .sort((a, b) => a.date - b.date);
}

// ─── Equity TR ──────────────────────────────────────────────────────────────
function buildEquityIndex(data) {
  const n = data.length;
  const tr = new Float64Array(n);
  tr[0] = 100;
  for (let t = 1; t < n; t++) tr[t] = tr[t - 1] * (1 + data[t].changePct / 100);
  return data.map((d, i) => ({ date: d.date, tr: tr[i] }));
}

// ─── Bond total return (constant-maturity par-bond repricing) ─────────────
function parBondPrice(y, c, T) {
  const annuity = Math.abs(y) > 1e-8 ? (1 - Math.pow(1 + y, -T)) / y : T;
  return c * 100 * annuity + 100 * Math.pow(1 + y, -T);
}

function buildBondTR(data, maturityYears) {
  const n = data.length;
  const tr = new Float64Array(n);
  const dailyReturn = new Float64Array(n);
  tr[0] = 100;
  for (let t = 1; t < n; t++) {
    const yPrev = data[t - 1].close / 100;
    const yToday = data[t].close / 100;
    const priceToday = parBondPrice(yToday, yPrev, maturityYears);
    const dayGap = (data[t].date - data[t - 1].date) / 86400000;
    const accrued = yPrev * 100 * (dayGap / 365);
    dailyReturn[t] = (priceToday + accrued) / 100 - 1;
    tr[t] = tr[t - 1] * (1 + dailyReturn[t]);
  }
  return data.map((d, i) => ({ date: d.date, tr: tr[i] }));
}

// ─── Align + portfolio ──────────────────────────────────────────────────────
function alignSeries(equityTR, bondTR) {
  const bondMap = new Map(bondTR.map(d => [d.date.getTime(), d.tr]));
  const rows = [];
  for (const e of equityTR) {
    const bTr = bondMap.get(e.date.getTime());
    if (bTr !== undefined) rows.push({ date: e.date, equityTr: e.tr, bondTr: bTr });
  }
  for (let i = 1; i < rows.length; i++) {
    rows[i].equityRet = rows[i].equityTr / rows[i - 1].equityTr - 1;
    rows[i].bondRet = rows[i].bondTr / rows[i - 1].bondTr - 1;
  }
  rows[0].equityRet = NaN;
  rows[0].bondRet = NaN;
  return rows;
}

function periodKey(date, freq) {
  if (freq === 'annual') return date.getUTCFullYear();
  if (freq === 'quarterly') return date.getUTCFullYear() * 10 + Math.floor(date.getUTCMonth() / 3);
  return null;
}

function portfolioTR(aligned, equityWeight, rebalanceFreq) {
  const n = aligned.length;
  const wEq = equityWeight, wBd = 1 - equityWeight;
  const series = new Float64Array(n);
  series[0] = 100;
  let eqVal = 100 * wEq, bdVal = 100 * wBd;
  let prevKey = rebalanceFreq === 'none' ? null : periodKey(aligned[0].date, rebalanceFreq);
  for (let t = 1; t < n; t++) {
    eqVal *= (1 + (aligned[t].equityRet || 0));
    bdVal *= (1 + (aligned[t].bondRet || 0));
    if (rebalanceFreq !== 'none') {
      const key = periodKey(aligned[t].date, rebalanceFreq);
      if (key !== prevKey) {
        const total = eqVal + bdVal;
        eqVal = total * wEq;
        bdVal = total * wBd;
        prevKey = key;
      }
    }
    series[t] = eqVal + bdVal;
  }
  return series;
}

// ─── Drawdown / episodes ────────────────────────────────────────────────────
function rollingDrawdown(values) {
  const n = values.length;
  const dd = new Float64Array(n);
  let runMax = -Infinity;
  for (let i = 0; i < n; i++) {
    runMax = Math.max(runMax, values[i]);
    dd[i] = values[i] / runMax - 1;
  }
  return dd;
}

function identifyEpisodes(dates, values, threshold) {
  const n = values.length;
  const dd = rollingDrawdown(values);
  const episodes = [];
  let inEpisode = false, peakIdx = 0, troughIdx = 0;
  let runMax = values[0], runMaxIdx = 0;

  let i = 1;
  while (i < n) {
    if (values[i] > runMax) { runMax = values[i]; runMaxIdx = i; }
    if (!inEpisode) {
      if (dd[i] < 0) {
        peakIdx = runMaxIdx;
        inEpisode = true;
        troughIdx = i;
      }
      i++;
    } else {
      if (values[i] < values[troughIdx]) troughIdx = i;
      if (dd[i] >= 0) {
        if (dd[troughIdx] <= threshold) {
          episodes.push({
            peakIdx, peakDate: dates[peakIdx], peakValue: values[peakIdx],
            troughIdx, troughDate: dates[troughIdx], troughValue: values[troughIdx],
            recoveryIdx: i, recoveryDate: dates[i],
            mdd: dd[troughIdx],
            recoveryDays: (dates[i] - dates[troughIdx]) / 86400000,
          });
        }
        inEpisode = false;
      }
      i++;
    }
  }
  if (inEpisode && dd[troughIdx] <= threshold) {
    episodes.push({
      peakIdx, peakDate: dates[peakIdx], peakValue: values[peakIdx],
      troughIdx, troughDate: dates[troughIdx], troughValue: values[troughIdx],
      recoveryIdx: null, recoveryDate: null,
      mdd: dd[troughIdx], recoveryDays: null,
    });
  }
  return episodes;
}

// ─── H1: exit-loss simulation ───────────────────────────────────────────────
function simulateExitLosses(aligned, weightSeries, episodes) {
  const dates = aligned.map(d => d.date);
  const n = dates.length;
  const results = [];
  for (const ep of episodes) {
    const searchEnd = ep.recoveryIdx !== null ? ep.recoveryIdx : n - 1;
    for (const w of CONFIG.weightGrid) {
      const series = weightSeries[w];
      const peakVal = series[ep.peakIdx];
      let troughLocal = 0, minRel = Infinity;
      for (let t = ep.peakIdx; t <= searchEnd; t++) {
        const rel = series[t] / peakVal - 1;
        if (rel < minRel) { minRel = rel; troughLocal = t - ep.peakIdx; }
      }
      const troughIdx = ep.peakIdx + troughLocal;
      let recoveryIdx = null;
      for (let t = troughIdx; t < n; t++) {
        if (series[t] >= peakVal) { recoveryIdx = t; break; }
      }
      const recoveryDays = recoveryIdx !== null ? (dates[recoveryIdx] - dates[troughIdx]) / 86400000 : null;
      results.push({ episode: ep, weight: w, mdd: minRel, troughIdx, recoveryDays, recovered: recoveryIdx !== null });
    }
  }
  return results;
}

function aggregateByWeight(results) {
  const agg = {};
  for (const w of CONFIG.weightGrid) {
    const rows = results.filter(r => r.weight === w);
    const mdds = rows.map(r => r.mdd);
    const recs = rows.filter(r => r.recovered).map(r => r.recoveryDays);
    agg[w] = {
      meanMdd: mean(mdds), worstMdd: Math.min(...mdds),
      meanRecovery: recs.length ? mean(recs) : null,
      nRecovered: recs.length, nEpisodes: rows.length,
    };
  }
  return agg;
}

const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

// ─── H3: 위험 대비 수익(변동성, 샤프비율) ───────────────────────────────────
function annualizedVol(returns) {
  const m = mean(returns);
  const variance = mean(returns.map(r => (r - m) ** 2));
  return Math.sqrt(variance) * Math.sqrt(252);
}

function computeSharpeGrid() {
  const { weightSeries, dates } = cache;
  const years = (dates[dates.length - 1] - dates[0]) / 86400000 / 365.25;
  const grid = {};
  for (const w of CONFIG.weightGrid) {
    const s = weightSeries[w];
    const rets = [];
    for (let t = 1; t < s.length; t++) rets.push(s[t] / s[t - 1] - 1);
    const cagrW = Math.pow(s[s.length - 1] / 100, 1 / years) - 1;
    const volW = annualizedVol(rets);
    grid[w] = { cagr: cagrW, vol: volW, sharpe: volW > 0 ? cagrW / volW : 0 };
  }
  return grid;
}

// ─── H2: correlation ─────────────────────────────────────────────────────────
function pearson(xs, ys) {
  const n = xs.length;
  const mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  const r = sxy / Math.sqrt(sxx * syy);
  return { r, n };
}

// two-tailed p-value for Pearson r via Student-t approximation (regularized incomplete beta)
function pValueForR(r, n) {
  if (n < 3) return NaN;
  const df = n - 2;
  const t = Math.abs(r) * Math.sqrt(df / Math.max(1e-12, 1 - r * r));
  return betaInc(df / 2, 0.5, df / (df + t * t));
}
function betaInc(a, b, x) {
  if (x <= 0) return 1;
  if (x >= 1) return 0;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return bt * betaCF(a, b, x) / a;
  return 1 - bt * betaCF(b, a, 1 - x) / b;
}
function betaCF(a, b, x) {
  const MAXIT = 200, EPS = 3e-9, FPMIN = 1e-30;
  let qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}
function logGamma(x) {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function quantile(sortedArr, q) {
  const pos = (sortedArr.length - 1) * q;
  const base = Math.floor(pos), rest = pos - base;
  if (sortedArr[base + 1] !== undefined) return sortedArr[base] + rest * (sortedArr[base + 1] - sortedArr[base]);
  return sortedArr[base];
}

function rollingCorrelation(equityRet, bondRet, window) {
  const n = equityRet.length;
  const out = new Float64Array(n).fill(NaN);
  for (let end = window; end < n; end++) {
    const xs = equityRet.slice(end - window, end);
    const ys = bondRet.slice(end - window, end);
    out[end] = pearson(xs, ys).r;
  }
  return out;
}

function yearlyCorrelation(aligned) {
  const byYear = {};
  for (let i = 1; i < aligned.length; i++) {
    const y = aligned[i].date.getUTCFullYear();
    (byYear[y] = byYear[y] || { eq: [], bd: [] });
    byYear[y].eq.push(aligned[i].equityRet);
    byYear[y].bd.push(aligned[i].bondRet);
  }
  return Object.entries(byYear).filter(([, v]) => v.eq.length >= 20)
    .map(([y, v]) => ({ year: +y, r: pearson(v.eq, v.bd).r }));
}

// ─── Formatting ─────────────────────────────────────────────────────────────
const fmtPct = (v, d = 2) => (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
const fmtDate = d => `${d.getUTCFullYear()}.${String(d.getUTCMonth() + 1).padStart(2, '0')}.${String(d.getUTCDate()).padStart(2, '0')}`;
const fmtLabel = d => `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}`;

// ─── Load ───────────────────────────────────────────────────────────────────
async function loadPair(key) {
  const cfg = CONFIG.pairs[key];
  const [eqRaw, bdRaw] = await Promise.all([fetchCSV(cfg.equityFile), fetchCSV(cfg.bondFile)]);
  const eqData = parseSeries(eqRaw);
  const bdData = parseSeries(bdRaw);
  const equityTR = buildEquityIndex(eqData);
  const bondTR = buildBondTR(bdData, CONFIG.maturityYears);
  const aligned = alignSeries(equityTR, bondTR);
  return { aligned };
}

async function init() {
  try {
    const [sp500, kospi] = await Promise.all([loadPair('sp500'), loadPair('kospi')]);
    dataByPair.sp500 = sp500;
    dataByPair.kospi = kospi;

    document.getElementById('loading').style.display = 'none';
    document.getElementById('app').style.display = 'block';

    setupControls();
    update();
  } catch (err) {
    document.getElementById('loading').innerHTML = `
      <div class="error-box">
        <h3>데이터 로딩 실패</h3>
        <p>파일을 직접 열면 CSV를 불러올 수 없습니다. 로컬 서버로 실행해주세요:</p>
        <p style="margin-top:8px;color:#7a8aaa">python3 -m http.server 8080</p>
        <p style="margin-top:12px;font-size:11px;color:#444">오류: ${err.message}</p>
      </div>`;
  }
}

function setupControls() {
  document.querySelectorAll('.pair-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.pair-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.pair = btn.dataset.pair;
      state.selectedEpisode = 0;
      update();
    });
  });
  document.querySelector(`.pair-tab[data-pair="${state.pair}"]`).classList.add('active');

  const slider = document.getElementById('weightSlider');
  slider.addEventListener('input', () => {
    state.weight = (+slider.value) / 100;
    document.getElementById('weightVal').textContent = slider.value;
    document.getElementById('bondVal').textContent = (100 - slider.value) + '%';
    updateWeightDependent();
    renderSummary();
  });

  document.querySelectorAll('#rebalanceGroup .radio-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#rebalanceGroup .radio-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.rebalanceFreq = btn.dataset.freq;
      state.selectedEpisode = 0;
      update();
    });
  });
  document.querySelectorAll('#thresholdGroup .radio-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#thresholdGroup .radio-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.threshold = +btn.dataset.th;
      state.selectedEpisode = 0;
      update();
    });
  });
  document.querySelectorAll('#crisisGroup .radio-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#crisisGroup .radio-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.crisisPct = +btn.dataset.pct;
      updateH2();
    });
  });
}

// ─── Main render pipeline ───────────────────────────────────────────────────
let cache = {}; // per pair+freq+threshold: { weightSeries, episodes, exitResults, agg }

function computeGridCache() {
  const { aligned } = dataByPair[state.pair];
  const key = `${state.pair}|${state.rebalanceFreq}|${state.threshold}`;
  if (cache.key === key) return cache;

  const weightSeries = {};
  for (const w of CONFIG.weightGrid) weightSeries[w] = portfolioTR(aligned, w, state.rebalanceFreq);
  const dates = aligned.map(d => d.date);
  const episodes = identifyEpisodes(dates, weightSeries[1.0], state.threshold);
  const exitResults = simulateExitLosses(aligned, weightSeries, episodes);
  const agg = aggregateByWeight(exitResults);

  // 기본 선택: 연대순 첫 구간이 아니라 낙폭이 가장 컸던(가장 대표적인) 구간
  let worstIdx = 0, worstMdd = Infinity;
  episodes.forEach((e, i) => { if (e.mdd < worstMdd) { worstMdd = e.mdd; worstIdx = i; } });
  state.selectedEpisode = worstIdx;

  cache = { key, aligned, dates, weightSeries, episodes, exitResults, agg };
  return cache;
}

function update() {
  computeGridCache();
  renderSummary();
  updateWeightDependent();
  renderH1();
  renderEpisodeTable();
  renderEpisodeChart();
  updateH2();
  renderH3();
}

// ─── 핵심 요약 (100% 채권 vs 100% 주식 vs 혼합) ─────────────────────────────
function renderSummary() {
  const { aligned, dates, weightSeries } = cache;
  const equityOnly = weightSeries[1.0];
  const bondOnly = weightSeries[0.0];
  const portfolio = currentPortfolioSeries(state.weight);

  const years = (dates[dates.length - 1] - dates[0]) / 86400000 / 365.25;
  const cagr = s => Math.pow(s[s.length - 1] / 100, 1 / years) - 1;
  const mdd = s => Math.min(...rollingDrawdown(s));

  const cagrEq = cagr(equityOnly), cagrBd = cagr(bondOnly), cagrPort = cagr(portfolio);
  const mddEq = mdd(equityOnly), mddBd = mdd(bondOnly), mddPort = mdd(portfolio);

  const valid = aligned.slice(1);
  const corr = pearson(valid.map(d => d.equityRet), valid.map(d => d.bondRet)).r;

  document.getElementById('summaryGrid').innerHTML = `
    <div class="metric-tile">
      <div class="lbl tip" data-tip="투자 기간 전체를 1년 단위로 환산했을 때 평균적으로 매년 번 수익률입니다.">연평균 수익률</div>
      <div class="val c-orange" style="font-size:16px">주식 ${fmtPct(cagrEq)}</div>
      <div class="val c-green" style="font-size:16px;margin-top:2px">채권 ${fmtPct(cagrBd)}</div>
      <div class="sub">지금 설정(주식${Math.round(state.weight * 100)}%): ${fmtPct(cagrPort)}</div>
    </div>
    <div class="metric-tile">
      <div class="lbl tip" data-tip="가장 비쌀 때(고점) 대비 가장 많이 떨어졌던 순간의 낙폭입니다. 그때 팔았다면 이만큼 손해를 본다는 뜻입니다.">최대로 떨어진 폭</div>
      <div class="val c-orange" style="font-size:16px">주식 ${fmtPct(mddEq)}</div>
      <div class="val c-green" style="font-size:16px;margin-top:2px">채권 ${fmtPct(mddBd)}</div>
      <div class="sub">지금 설정(주식${Math.round(state.weight * 100)}%): ${fmtPct(mddPort)}</div>
    </div>
    <div class="metric-tile">
      <div class="lbl tip" data-tip="주식과 채권이 같은 날 같은 방향으로 움직였는지를 나타내는 숫자입니다. 0보다 작을수록(음수일수록) 주식이 떨어질 때 채권이 오르는 경향이 강하다는 뜻입니다.">주식과 반대로 움직인 정도</div>
      <div class="val ${corr < 0 ? 'c-green' : 'c-red'}">${corr.toFixed(2)}</div>
      <div class="sub">${corr < -0.15 ? '어느 정도 반대로 움직임' : corr < 0 ? '약하게 반대로 움직임' : '오히려 같이 움직임 (방어 효과 약함)'}</div>
    </div>
    <div class="metric-tile">
      <div class="lbl">분석 기간</div>
      <div class="val">${years.toFixed(1)}년</div>
      <div class="sub">${fmtDate(dates[0])} ~ ${fmtDate(dates[dates.length - 1])}</div>
    </div>
  `;

  document.getElementById('summaryVerdict').innerHTML = `
    이 기간 동안 <b>주식</b>은 매년 평균 ${fmtPct(cagrEq)}, <b>채권</b>은 매년 평균 ${fmtPct(cagrBd)} 벌었습니다.
    대신 가장 심하게 떨어졌을 때 주식은 ${fmtPct(mddEq)}, 채권은 ${fmtPct(mddBd)}까지 떨어졌습니다 — <b>채권은 덜 벌지만 덜 떨어집니다.</b>
    지금처럼 채권을 섞으면(주식 ${Math.round(state.weight * 100)}%) 수익은 ${fmtPct(cagrPort)}로 낮아지는 대신, 최대 낙폭은 ${fmtPct(mddPort)}로 줄어듭니다.
    또한 둘의 상관계수가 ${corr.toFixed(2)}로 ${corr < 0 ? '주식이 떨어질 때 채권이 반대로 움직이는 경향이 있어서' : '뚜렷하게 반대로 움직이지는 않아서'}, 함께 담았을 때 ${corr < 0 ? '어느 정도 방어 효과를 기대할 수 있습니다.' : '기대만큼 방어 효과가 크지 않을 수 있습니다.'}
    아래에서 실제 하락장 사례로 더 자세히 확인해보세요.
  `;
}

function currentPortfolioSeries(weight) {
  const { aligned, rebalanceFreqUsed } = cache;
  if (Math.abs(weight - Math.round(weight * 5) / 5) < 1e-9 && cache.weightSeries[weight]) return cache.weightSeries[weight];
  return portfolioTR(cache.aligned, weight, state.rebalanceFreq);
}

function updateWeightDependent() {
  const { aligned, dates } = cache;
  const portfolio = currentPortfolioSeries(state.weight);
  const equityOnly = cache.weightSeries[1.0];
  const dd = rollingDrawdown(portfolio);
  const ddEquity = rollingDrawdown(equityOnly);

  // metrics
  const years = (dates[dates.length - 1] - dates[0]) / 86400000 / 365.25;
  const cagr = Math.pow(portfolio[portfolio.length - 1] / 100, 1 / years) - 1;
  const cagrEquity = Math.pow(equityOnly[equityOnly.length - 1] / 100, 1 / years) - 1;
  const mdd = Math.min(...dd);
  const mddEquity = Math.min(...ddEquity);

  document.getElementById('metricsGrid').innerHTML = `
    <div class="metric-tile"><div class="lbl tip" data-tip="1년 단위로 환산했을 때 평균적으로 매년 번 수익률입니다.">연평균 수익률</div><div class="val c-blue">${fmtPct(cagr)}</div><div class="sub">주식만 100%면: ${fmtPct(cagrEquity)}</div></div>
    <div class="metric-tile"><div class="lbl tip" data-tip="가장 비쌀 때 대비 가장 많이 떨어졌던 순간의 낙폭입니다.">최대로 떨어진 폭</div><div class="val c-red">${fmtPct(mdd)}</div><div class="sub">주식만 100%면: ${fmtPct(mddEquity)}</div></div>
    <div class="metric-tile"><div class="lbl tip" data-tip="채권을 섞어서 주식 100%일 때보다 낙폭이 몇%p 줄었는지입니다. 클수록 하락장에서 덜 떨어졌다는 뜻입니다.">낙폭이 줄어든 정도</div><div class="val c-green">${fmtPct(mdd - mddEquity)}p</div><div class="sub">주식 100%와 비교</div></div>
    <div class="metric-tile"><div class="lbl">분석 기간</div><div class="val">${years.toFixed(1)}년</div><div class="sub">${fmtDate(dates[0])} ~ ${fmtDate(dates[dates.length - 1])}</div></div>
  `;

  // TR chart (sampled every 5 points for perf)
  const step = Math.max(1, Math.floor(dates.length / 1500));
  const labels = [], trPort = [], trEq = [], trBond = [];
  const bondOnly = cache.weightSeries[0.0];
  for (let i = 0; i < dates.length; i += step) {
    labels.push(fmtLabel(dates[i]));
    trPort.push(+portfolio[i].toFixed(2));
    trEq.push(+equityOnly[i].toFixed(2));
    trBond.push(+bondOnly[i].toFixed(2));
  }
  charts.tr = renderLineChart('trChart', charts.tr, labels, [
    { label: `혼합 포트폴리오 (주식 ${Math.round(state.weight * 100)}%)`, data: trPort, color: '#5b9cf6', dash: [] },
    { label: '100% 주식', data: trEq, color: '#f2b53c', dash: [5, 3] },
    { label: '100% 채권', data: trBond, color: '#4cce82', dash: [2, 2] },
  ], { logScale: true });

  const ddPort = [], ddEq = [];
  for (let i = 0; i < dates.length; i += step) {
    ddPort.push(+(dd[i] * 100).toFixed(2));
    ddEq.push(+(ddEquity[i] * 100).toFixed(2));
  }
  charts.dd = renderLineChart('ddChart', charts.dd, labels, [
    { label: `혼합 포트폴리오`, data: ddPort, color: '#5b9cf6', dash: [] },
    { label: '100% 주식', data: ddEq, color: '#f2b53c', dash: [5, 3] },
  ], { fill: true });
}

function renderLineChart(canvasId, existing, labels, datasets, opts = {}) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (existing) existing.destroy();
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: datasets.map(d => ({
        label: d.label, data: d.data, borderColor: d.color, borderWidth: 1.4,
        pointRadius: 0, tension: 0.15, fill: opts.fill ? 'origin' : false,
        backgroundColor: opts.fill ? d.color + '18' : undefined, borderDash: d.dash || [],
      })),
    },
    options: {
      responsive: true, animation: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#9aa3b5', font: { size: 11 }, boxWidth: 18 } } },
      scales: {
        x: { ticks: { color: '#555', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#1a1d27' } },
        y: {
          type: opts.logScale ? 'logarithmic' : 'linear',
          ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1d27' },
        },
      },
    },
  });
}

// ─── H1 rendering ───────────────────────────────────────────────────────────
function renderH1() {
  const { agg, episodes } = cache;
  const weights = CONFIG.weightGrid;
  const labels = weights.map(w => `${Math.round(w * 100)}%`);
  const colors = weights.map(w => CONFIG.gridColors[w]);

  charts.mddByWeight = renderBarChart('mddByWeightChart', charts.mddByWeight, labels,
    weights.map(w => +(agg[w].meanMdd * 100).toFixed(2)), colors, '평균 낙폭 (%)');
  charts.recoveryByWeight = renderBarChart('recoveryByWeightChart', charts.recoveryByWeight, labels,
    weights.map(w => agg[w].meanRecovery !== null ? +agg[w].meanRecovery.toFixed(0) : 0), colors, '원금 회복까지 평균 며칠 걸렸나');

  const base = agg[1.0].meanMdd;
  const improved = weights.filter(w => w !== 1.0).every(w => agg[w].meanMdd > base);
  document.getElementById('h1Verdict').innerHTML = `
    <div class="verdict ${improved ? 'pass' : 'fail'}">
      지난 ${episodes.length}번의 큰 하락(고점 대비 ${Math.abs(state.threshold * 100).toFixed(0)}% 이상 하락) 기준 —
      ${improved ? '채권을 많이 섞을수록 모든 하락장에서 손실이 예외 없이 더 적었습니다.' : '채권을 섞는다고 항상 손실이 줄어들지는 않았습니다 (구간마다 결과가 달랐습니다).'}
    </div>`;
}

function renderBarChart(canvasId, existing, labels, data, colors, yLabel) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (existing) existing.destroy();
  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 4 }] },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { display: false }, title: { display: true, text: yLabel, color: '#9aa3b5', font: { size: 11 } } },
      scales: {
        x: { ticks: { color: '#9aa3b5', font: { size: 11 } }, grid: { display: false } },
        y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1d27' } },
      },
    },
  });
}

// ─── Episode explorer ───────────────────────────────────────────────────────
function renderEpisodeTable() {
  const { episodes } = cache;
  const ranked = episodes.map((e, i) => ({ ...e, idx: i })).sort((a, b) => a.mdd - b.mdd);
  let html = '<table><thead><tr><th>하락 시작일</th><th>주식 낙폭</th></tr></thead><tbody>';
  ranked.forEach(e => {
    const hl = e.idx === state.selectedEpisode ? ' class="hl clickable"' : ' class="clickable"';
    html += `<tr${hl} data-idx="${e.idx}"><td>${fmtDate(e.peakDate)}</td><td class="c-red">${fmtPct(e.mdd, 1)}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('episodeTable').innerHTML = html;
  document.querySelectorAll('#episodeTable tr[data-idx]').forEach(tr => {
    tr.addEventListener('click', () => {
      state.selectedEpisode = +tr.dataset.idx;
      renderEpisodeTable();
      renderEpisodeChart();
    });
  });
}

function renderEpisodeChart() {
  const { episodes, dates, weightSeries } = cache;
  if (!episodes.length) return;
  const ep = episodes[state.selectedEpisode] || episodes[0];
  const endIdx = Math.min((ep.recoveryIdx !== null ? ep.recoveryIdx : dates.length - 1) + 10, dates.length - 1);
  const labels = [];
  for (let i = ep.peakIdx; i <= endIdx; i++) labels.push(fmtDate(dates[i]));

  const datasets = CONFIG.weightGrid.map(w => {
    const series = weightSeries[w];
    const peakVal = series[ep.peakIdx];
    const data = [];
    for (let i = ep.peakIdx; i <= endIdx; i++) data.push(+((series[i] / peakVal - 1) * 100).toFixed(2));
    return { label: `주식 ${Math.round(w * 100)}%`, data, color: CONFIG.gridColors[w], dash: [] };
  });
  charts.episode = renderLineChart('episodeChart', charts.episode, labels, datasets);
}

// ─── H2 rendering ───────────────────────────────────────────────────────────
function updateH2() {
  const { aligned } = cache;
  const valid = aligned.slice(1);
  const eqRet = valid.map(d => d.equityRet);
  const bdRet = valid.map(d => d.bondRet);

  const full = pearson(eqRet, bdRet);
  const fullP = pValueForR(full.r, full.n);

  const sorted = [...eqRet].sort((a, b) => a - b);
  const cutoff = quantile(sorted, state.crisisPct);
  const crisisEq = [], crisisBd = [];
  for (let i = 0; i < eqRet.length; i++) if (eqRet[i] <= cutoff) { crisisEq.push(eqRet[i]); crisisBd.push(bdRet[i]); }
  const crisis = pearson(crisisEq, crisisBd);
  const crisisP = pValueForR(crisis.r, crisis.n);

  document.getElementById('corrSummary').innerHTML = `
    <div class="corr-box">
      <div class="lbl tip" data-tip="투자 기간 전체를 놓고 봤을 때, 주식과 채권이 같은 날 같은 방향으로 움직인 정도입니다. 음수일수록 주식이 떨어질 때 채권이 오르는 경향이 강합니다.">평소 (전체 기간)</div>
      <div class="val ${full.r < 0 ? 'c-green' : 'c-red'}">${full.r.toFixed(2)}</div>
      <div class="p tip" data-tip="표본 ${full.n}개, 유의확률 p=${fullP < 0.001 ? '<0.001' : fullP.toFixed(3)}">${fullP < 0.05 ? '우연이 아닐 가능성이 높음' : '표본이 적어 확실치 않음'}</div>
    </div>
    <div class="corr-box">
      <div class="lbl tip" data-tip="주가가 가장 많이 떨어진 날(하위 ${Math.round(state.crisisPct * 100)}%)만 뽑아서 봤을 때의 상관관계입니다. 평소보다 더 음수라면, 정작 필요한 위기 때 채권이 더 강하게 방어해준다는 뜻입니다.">위기 때 (주가 급락일 하위${Math.round(state.crisisPct * 100)}%)</div>
      <div class="val ${crisis.r < 0 ? 'c-green' : 'c-red'}">${crisis.r.toFixed(2)}</div>
      <div class="p tip" data-tip="표본 ${crisis.n}개, 유의확률 p=${crisisP < 0.001 ? '<0.001' : crisisP.toFixed(3)}">${crisisP < 0.05 ? '우연이 아닐 가능성이 높음' : '표본이 적어 확실치 않음'}</div>
    </div>
  `;

  const roll = rollingCorrelation(eqRet, bdRet, 60);
  const labels = valid.map(d => fmtLabel(d.date));
  const step = Math.max(1, Math.floor(labels.length / 1500));
  const sampledLabels = [], sampledRoll = [];
  for (let i = 0; i < labels.length; i += step) { sampledLabels.push(labels[i]); sampledRoll.push(isNaN(roll[i]) ? null : +roll[i].toFixed(3)); }
  charts.rollingCorr = renderLineChart('rollingCorrChart', charts.rollingCorr, sampledLabels,
    [{ label: '최근 60일 기준 함께 움직인 정도', data: sampledRoll, color: '#5b9cf6', dash: [] }]);

  const scatterCtx = document.getElementById('scatterChart').getContext('2d');
  if (charts.scatter) charts.scatter.destroy();
  const normalPts = [], crisisPts = [];
  for (let i = 0; i < eqRet.length; i++) {
    const pt = { x: +(eqRet[i] * 100).toFixed(3), y: +(bdRet[i] * 100).toFixed(3) };
    (eqRet[i] <= cutoff ? crisisPts : normalPts).push(pt);
  }
  charts.scatter = new Chart(scatterCtx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: '평상시', data: normalPts, backgroundColor: 'rgba(106,166,216,.35)', pointRadius: 2.5 },
        { label: `주식 하락 하위${Math.round(state.crisisPct * 100)}%`, data: crisisPts, backgroundColor: 'rgba(192,57,43,.75)', pointRadius: 3.2 },
      ],
    },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { labels: { color: '#9aa3b5', font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: '주식 일별수익률(%)', color: '#9aa3b5' }, ticks: { color: '#555' }, grid: { color: '#1a1d27' } },
        y: { title: { display: true, text: '채권 일별수익률(%)', color: '#9aa3b5' }, ticks: { color: '#555' }, grid: { color: '#1a1d27' } },
      },
    },
  });

  const yearly = yearlyCorrelation(aligned);
  const posYears = yearly.filter(y => y.r > 0).map(y => y.year);
  const h2Pass = full.r < 0 && fullP < 0.05 && crisis.r < 0 && crisisP < 0.05;
  document.getElementById('h2Verdict').innerHTML = `
    <div class="verdict ${h2Pass ? 'pass' : 'fail'}">
      ${h2Pass
        ? '평소에도, 특히 위기 때도 채권이 주식과 반대로 움직이는 경향이 뚜렷했습니다 — 방어 효과가 있다고 볼 수 있습니다.'
        : (full.r < 0 && fullP < 0.05
            ? '평소에는 채권이 주식과 반대로 움직였지만, 정작 위기 때는 그 경향이 뚜렷하지 않았습니다 — 하락장 방어 효과를 단정하기는 어렵습니다.'
            : '평소에도 위기 때도 채권이 주식과 뚜렷하게 반대로 움직인다고 보기는 어려웠습니다.')}
      ${posYears.length ? ` · 채권이 오히려 주식과 같이 움직인 해: ${posYears.join(', ')}` : ''}
    </div>`;
}

// ─── H3 rendering ───────────────────────────────────────────────────────────
const fmtPlainPct = (v, d = 1) => (v * 100).toFixed(d) + '%';

function renderH3() {
  const sharpeGrid = computeSharpeGrid();
  const weights = CONFIG.weightGrid;
  const labels = weights.map(w => `${Math.round(w * 100)}%`);
  const colors = weights.map(w => CONFIG.gridColors[w]);

  charts.sharpeByWeight = renderBarChart('sharpeByWeightChart', charts.sharpeByWeight, labels,
    weights.map(w => +sharpeGrid[w].sharpe.toFixed(2)), colors, '수익 ÷ 변동성 (무위험금리 0% 가정)');

  let bestW = weights[0];
  for (const w of weights) if (sharpeGrid[w].sharpe > sharpeGrid[bestW].sharpe) bestW = w;

  const eq = sharpeGrid[1.0];
  const best = sharpeGrid[bestW];
  const lev = best.vol > 0 ? eq.vol / best.vol : 1;
  const leveredReturn = best.cagr * lev;

  document.getElementById('leverageBox').innerHTML = `
    <div class="leverage-stack">
      <div class="corr-box">
        <div class="lbl">100% 주식 (그대로)</div>
        <div class="val c-orange">${fmtPct(eq.cagr)}</div>
        <div class="p">변동성 ${fmtPlainPct(eq.vol)} · 수익÷변동성 ${eq.sharpe.toFixed(2)}</div>
      </div>
      <div class="corr-box">
        <div class="lbl tip" data-tip="지금 그리드(주식 100%~0%) 중 수익÷변동성이 가장 높은 조합입니다.">가장 효율적인 조합 (주식 ${Math.round(bestW * 100)}%)</div>
        <div class="val c-blue">${fmtPct(best.cagr)}</div>
        <div class="p">변동성 ${fmtPlainPct(best.vol)} · 수익÷변동성 ${best.sharpe.toFixed(2)}</div>
      </div>
      <div class="corr-box">
        <div class="lbl tip" data-tip="위 조합에 레버리지 ${lev.toFixed(2)}배를 걸어서 100% 주식과 같은 변동성(${fmtPlainPct(eq.vol)})까지 맞췄다고 가정했을 때의 이론상 기대수익입니다. 레버리지 조달비용, 변동성 드래그, 마진콜 위험은 반영하지 않은 단순 계산치입니다.">레버리지로 주식과 같은 위험까지 올리면</div>
        <div class="val ${leveredReturn > eq.cagr ? 'c-green' : 'c-red'}">${fmtPct(leveredReturn)}</div>
        <div class="p">${lev.toFixed(2)}배 레버리지 가정 (이론상 근사치)</div>
      </div>
    </div>
  `;

  document.getElementById('h3Verdict').innerHTML = `
    <div class="verdict ${leveredReturn > eq.cagr ? 'pass' : 'fail'}">
      ${leveredReturn > eq.cagr
        ? `가장 효율적인 조합(주식 ${Math.round(bestW * 100)}%)은 100% 주식보다 덜 벌지만(${fmtPct(best.cagr)}), 같은 위험 수준까지 레버리지를 걸면 이론상 100% 주식(${fmtPct(eq.cagr)})보다 높은 ${fmtPct(leveredReturn)}을 기대할 수 있습니다 — "어떤 자산이 이기나"보다 "위험 대비 효율이 가장 좋은 조합을 찾고, 위험 수준은 레버리지로 조절하는" 접근이 이론상 더 유리하다는 뜻입니다.`
        : `이 그리드 안에서는 레버리지를 감안해도 100% 주식이 가장 효율적이었습니다. 이 구간에서는 채권을 섞는 분산 효과보다 주식 자체의 수익력이 더 크게 작용했습니다.`}
      실제로는 레버리지 조달비용·변동성 드래그·상관관계 붕괴 위험이 있어 이 수치보다 낮게 나올 가능성이 높습니다.
    </div>`;
}

init();
