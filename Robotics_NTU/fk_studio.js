import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {Line2} from 'three/addons/lines/Line2.js';
import {LineGeometry} from 'three/addons/lines/LineGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';

const $ = (id) => document.getElementById(id);
const vp = $('vp');
const ui = {
  robots: $('robots'), presets: $('presets'), sliders: $('sliders'),
  blurb: $('blurb'), ht: $('ht'), hs: $('hs'), note: $('note'),
  xv: $('xv'), yv: $('yv'), zv: $('zv'), ov: $('ov'), rv: $('rv'), pv: $('pv'), olab: $('olab'),
  frames: $('tg-frames'), trail: $('tg-trail'), work: $('tg-work'), auto: $('tg-auto'),
  home: $('go-home'), fit: $('go-fit'), clear: $('go-clear'),
  cams: [...document.querySelectorAll('.cam')]
};

window.addEventListener('error', (event) => {
  const message = event?.error?.message || event.message || 'Unknown runtime error';
  if (ui.note) ui.note.textContent = `前端运行报错: ${message}`;
  console.error(event.error || event.message);
});

const ren = new T.WebGLRenderer({antialias: true, powerPreference: 'high-performance'});
ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
ren.outputColorSpace = T.SRGBColorSpace;
ren.shadowMap.enabled = true;
ren.shadowMap.type = T.VSMShadowMap;
ren.toneMapping = T.ACESFilmicToneMapping;
ren.toneMappingExposure = 1.05;
vp.appendChild(ren.domElement);

const scene = new T.Scene();
scene.background = new T.Color(0x050c13);
scene.fog = new T.FogExp2(0x071019, 0.045);

const cam = new T.PerspectiveCamera(42, 1, 0.02, 160);
cam.position.set(3.8, 2.6, 3.8);
const ctrl = new OrbitControls(cam, ren.domElement);
ctrl.enableDamping = true;
ctrl.dampingFactor = 0.06;
ctrl.minDistance = 0.7;
ctrl.maxDistance = 28;

const composer = new EffectComposer(ren);
composer.addPass(new RenderPass(scene, cam));
composer.addPass(new UnrealBloomPass(new T.Vector2(1, 1), 0.42, 0.8, 0.68));
scene.environment = new T.PMREMGenerator(ren).fromScene(new RoomEnvironment(), 0.06).texture;

scene.add(new T.HemisphereLight(0xa8dbff, 0x081018, 1.1));
const key = new T.DirectionalLight(0xffffff, 3.5);
key.position.set(4.2, 6.4, 2.6);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, {left: -6, right: 6, top: 6, bottom: -6, near: 0.3, far: 22});
scene.add(key);
const rim = new T.DirectionalLight(0x89b7ff, 1.0);
rim.position.set(-5.2, 2.5, -3.4);
scene.add(rim);
const fill = new T.PointLight(0xffcf92, 0.9, 18, 2);
fill.position.set(0, 3.2, 0);
scene.add(fill);

const floor = new T.Mesh(
  new T.CircleGeometry(7.2, 120),
  new T.MeshPhysicalMaterial({color: 0x0a1823, roughness: 0.86, metalness: 0.08, clearcoat: 0.24, clearcoatRoughness: 0.38})
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);
const ring = new T.Mesh(
  new T.TorusGeometry(2.85, 0.018, 10, 180),
  new T.MeshBasicMaterial({color: 0x7ce4ff, transparent: true, opacity: 0.2})
);
ring.rotation.x = -Math.PI / 2;
ring.position.y = 0.002;
scene.add(ring);
const grid = new T.GridHelper(10, 40, 0x2d6b84, 0x143243);
grid.material.transparent = true;
grid.material.opacity = 0.32;
scene.add(grid);

const root = new T.Group();
const workspace = new T.Group();
scene.add(root, workspace);

const trailMat = new LineMaterial({color: 0x7ce4ff, linewidth: 0.0042, transparent: true, opacity: 0.88});
const trailGeo = new LineGeometry();
const trail = new Line2(trailGeo, trailMat);
scene.add(trail);

const foot = new T.Mesh(
  new T.RingGeometry(0.055, 0.078, 48),
  new T.MeshBasicMaterial({color: 0xffb867, transparent: true, opacity: 0.56, side: T.DoubleSide})
);
foot.rotation.x = -Math.PI / 2;
foot.position.y = 0.004;
scene.add(foot);

const stem = new T.Line(
  new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3(0, 1, 0)]),
  new T.LineDashedMaterial({color: 0xffc987, dashSize: 0.08, gapSize: 0.05, transparent: true, opacity: 0.55})
);
scene.add(stem);
const glow = new T.Mesh(
  new T.TorusGeometry(0.078, 0.008, 12, 64),
  new T.MeshBasicMaterial({color: 0xffb867, transparent: true, opacity: 0.92})
);
scene.add(glow);

const mats = {
  base: new T.MeshPhysicalMaterial({color: 0x102334, roughness: 0.36, metalness: 0.78, clearcoat: 0.42}),
  joint: new T.MeshPhysicalMaterial({color: 0xeaf7ff, roughness: 0.18, metalness: 1, clearcoat: 0.72}),
  ee: new T.MeshPhysicalMaterial({color: 0xffc56f, emissive: 0x6f2b00, emissiveIntensity: 1.1, roughness: 0.16, metalness: 0.96}),
  links: [0x27b7ff, 0x64da7d, 0xffb347, 0xf870ff, 0x79f7ff, 0xff7888].map((c) => new T.MeshPhysicalMaterial({color: c, roughness: 0.28, metalness: 0.64, clearcoat: 0.34}))
};

const mapV = (v) => new T.Vector3(v.x, v.z, -v.y);
const pos3 = (m) => new T.Vector3(m.elements[12], m.elements[13], m.elements[14]);
const viewP = (p, c) => mapV(p).add(new T.Vector3(0, c.baseHeight, 0));
const deg = (r) => r * 180 / Math.PI;

function mdh(a, ad, d, td) {
  const alpha = ad * Math.PI / 180, theta = td * Math.PI / 180;
  const ca = Math.cos(alpha), sa = Math.sin(alpha), ct = Math.cos(theta), st = Math.sin(theta);
  const m = new T.Matrix4();
  m.set(ct, -st, 0, a, st * ca, ct * ca, -sa, -d * sa, st * sa, ct * sa, ca, d * ca, 0, 0, 0, 1);
  return m;
}
function sdh(a, ad, d, td) {
  const alpha = ad * Math.PI / 180, theta = td * Math.PI / 180;
  const ca = Math.cos(alpha), sa = Math.sin(alpha), ct = Math.cos(theta), st = Math.sin(theta);
  const m = new T.Matrix4();
  m.set(ct, -st * ca, st * sa, a * ct, st, ct * ca, -ct * sa, a * st, 0, sa, ca, d, 0, 0, 0, 1);
  return m;
}
function chain(rows, gen = mdh) {
  let t = new T.Matrix4();
  const out = [t.clone()];
  rows.forEach((row) => { t = t.clone().multiply(gen(...row)); out.push(t.clone()); });
  return out;
}

const ROBOTS = {
  rrr: {
    label: 'RRR', title: 'RRR Planar Chain', sub: '最适合讲解基础三连杆 FK、phi 累加与末端轨迹。',
    blurb: '三连杆平面机构，最适合观察末端位置和姿态角 phi 的几何含义。',
    j: [{l: 'theta1', mn: -180, mx: 180, v: 30, u: 'deg'}, {l: 'theta2', mn: -180, mx: 180, v: 60, u: 'deg'}, {l: 'theta3', mn: -180, mx: 180, v: -45, u: 'deg'}],
    p: {Home: [0, 0, 0], Sweep: [30, 60, -45], Folded: [95, -120, 55], Reach: [15, 35, 20]},
    f: (a) => chain([[0, 0, 0, a[0]], [0.5, 0, 0, a[1]], [0.4, 0, 0, a[2]], [0.3, 0, 0, 0]]),
    r: [0.06, 0.053, 0.044, 0.034], baseHeight: 0.14, phi: [0, 1, 2]
  },
  puma: {
    label: 'PUMA', title: 'PUMA 560', sub: '6-DOF 空间链条，更适合看整体空间 reach 和腕部构型。',
    blurb: 'PUMA 560 更适合看空间链条、腕部构型与整体 reach，而不是单一平面姿态角。',
    j: [{l: 'theta1', mn: -180, mx: 180, v: 0, u: 'deg'}, {l: 'theta2', mn: -135, mx: 135, v: 90, u: 'deg'}, {l: 'theta3', mn: -135, mx: 135, v: -90, u: 'deg'}, {l: 'theta4', mn: -180, mx: 180, v: 0, u: 'deg'}, {l: 'theta5', mn: -100, mx: 100, v: 0, u: 'deg'}, {l: 'theta6', mn: -266, mx: 266, v: 0, u: 'deg'}],
    p: {Home: [0, 90, -90, 0, 0, 0], Reach: [10, 40, -25, 0, 25, 0], Elbow: [-35, 65, -55, 30, -20, 20], Wrist: [50, 25, -35, 80, 55, -60]},
    f: (a) => chain([[0, 90, 26.45 * 0.0254, a[0]], [0.4318, 0, 0, a[1]], [0.0203, -90, 0.15005, a[2]], [0, 90, 0.4318, a[3]], [0, -90, 0, a[4]], [0, 0, 0, a[5]]], sdh),
    r: [0.062, 0.057, 0.05, 0.043, 0.036, 0.028, 0.022], baseHeight: 0.1, tip: 0.18, phi: null
  },
  rrrp: {
    label: 'RRRP', title: 'RRRP Chain', sub: '前三轴旋转，最后一轴伸缩，适合观察 reach 被直接拉远。',
    blurb: 'RRRP 清楚地展示了旋转关节与伸缩关节在末端可达域上的不同作用。',
    j: [{l: 'theta1', mn: -180, mx: 180, v: 40, u: 'deg'}, {l: 'theta2', mn: -180, mx: 180, v: 60, u: 'deg'}, {l: 'theta3', mn: -180, mx: 180, v: -30, u: 'deg'}, {l: 'd4', mn: 0, mx: 0.6, v: 0.2, st: 0.01, u: 'm'}],
    p: {Home: [0, 0, 0, 0.18], Compact: [60, -55, 40, 0.08], Reach: [25, 35, -20, 0.55], Sweep: [-40, 90, -65, 0.32]},
    f: (a) => chain([[0, 0, 0, a[0]], [0.45, 0, 0, a[1]], [0.35, 0, 0, a[2]], [0.25, 0, 0, 0], [a[3], 0, 0, 0]]),
    r: [0.058, 0.053, 0.046, 0.033, 0.028], baseHeight: 0.14, phi: [0, 1, 2]
  },
  prrr: {
    label: 'PRRR', title: 'PRRR Chain', sub: '先抬升 base 再进入三连杆旋转链，适合对比 RRRP。',
    blurb: 'PRRR 适合对比 RRRP：同样四自由度，但 prismatic joint 位于链前端时，工作空间会整体抬升。',
    j: [{l: 'd1', mn: 0, mx: 0.8, v: 0.3, st: 0.01, u: 'm'}, {l: 'theta1', mn: -180, mx: 180, v: 40, u: 'deg'}, {l: 'theta2', mn: -180, mx: 180, v: 60, u: 'deg'}, {l: 'theta3', mn: -180, mx: 180, v: -30, u: 'deg'}],
    p: {Home: [0.22, 0, 0, 0], Tower: [0.65, 30, 55, -40], Reach: [0.35, 20, 35, 10], Folded: [0.18, -55, 110, -70]},
    f: (a) => chain([[0, 0, a[0], 0], [0, 0, 0, a[1]], [0.45, 0, 0, a[2]], [0.35, 0, 0, a[3]], [0.25, 0, 0, 0]]),
    r: [0.052, 0.05, 0.044, 0.038, 0.03], baseHeight: 0.14, phi: [1, 2, 3]
  }
};

const urlRobot = new URLSearchParams(location.search).get('robot');
const initialRobot = ROBOTS[urlRobot] ? urlRobot : 'rrr';
const S = {robot: initialRobot, view: 'iso', frames: true, showTrail: true, work: false, auto: false, current: [], target: [], pts: [], trail: [], cache: new Map(), busy: false};
const O = {links: [], joints: [], axes: [], ee: null, light: null};

function mkLink(mat) {
  const g = new T.Group();
  const body = new T.Mesh(new T.CylinderGeometry(1, 1, 1, 28, 1, true), mat);
  const a = new T.Mesh(new T.SphereGeometry(1, 26, 26), mat);
  const b = new T.Mesh(new T.SphereGeometry(1, 26, 26), mat);
  [body, a, b].forEach((m) => { m.castShadow = true; m.receiveShadow = true; g.add(m); });
  g.userData = {body, a, b};
  return g;
}
function setLink(link, p1, p2, r) {
  const dir = new T.Vector3().subVectors(p2, p1), L = dir.length();
  if (L < 1e-4) { link.visible = false; return; }
  link.visible = true;
  link.position.copy(p1).add(p2).multiplyScalar(0.5);
  link.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir.clone().normalize());
  const h = Math.max(L - 2 * r, 0.001), u = link.userData;
  u.body.scale.set(r, h, r);
  u.a.scale.set(r, r, r);
  u.b.scale.set(r, r, r);
  u.a.position.set(0, h * 0.5, 0);
  u.b.position.set(0, -h * 0.5, 0);
}
function clearRoot() { while (root.children.length) root.remove(root.children[0]); }
function rawPts(frames, c) {
  const pts = frames.map(pos3);
  if (c.tip) {
    const tip = pos3(frames.at(-1));
    const x = new T.Vector3().setFromMatrixColumn(frames.at(-1), 0).normalize();
    pts.push(tip.clone().add(x.multiplyScalar(c.tip)));
  }
  return pts;
}
function setAxes(ax, frame, c) {
  ax.position.copy(viewP(pos3(frame), c));
  const x = mapV(new T.Vector3().setFromMatrixColumn(frame, 0)).normalize();
  const y = mapV(new T.Vector3().setFromMatrixColumn(frame, 1)).normalize();
  const z = mapV(new T.Vector3().setFromMatrixColumn(frame, 2)).normalize();
  ax.setRotationFromMatrix(new T.Matrix4().makeBasis(x, y, z));
  ax.visible = S.frames;
}
function rebuild() {
  clearRoot();
  O.links = []; O.joints = []; O.axes = [];
  const c = ROBOTS[S.robot];
  const ped = new T.Mesh(new T.CylinderGeometry(0.17, 0.21, c.baseHeight, 40), mats.base);
  ped.position.y = c.baseHeight * 0.5;
  ped.castShadow = ped.receiveShadow = true;
  root.add(ped);
  const cap = new T.Mesh(new T.CylinderGeometry(0.14, 0.14, 0.035, 40), mats.joint);
  cap.position.y = c.baseHeight + 0.018;
  cap.castShadow = true;
  root.add(cap);
  c.r.forEach((_, i) => { const l = mkLink(mats.links[i % mats.links.length].clone()); root.add(l); O.links.push(l); });
  c.j.forEach((_, i) => {
    const r = (c.r[Math.min(i + 1, c.r.length - 1)] || 0.028) * 1.55;
    const j = new T.Mesh(new T.SphereGeometry(r, 28, 28), mats.joint.clone());
    const a = new T.AxesHelper(0.2);
    j.castShadow = j.receiveShadow = true;
    root.add(j, a);
    O.joints.push(j);
    O.axes.push(a);
  });
  O.ee = new T.Mesh(new T.SphereGeometry(0.055, 30, 30), mats.ee.clone());
  O.ee.castShadow = O.ee.receiveShadow = true;
  root.add(O.ee);
  O.light = new T.PointLight(0xffaf4e, 1.2, 3.5, 2);
  root.add(O.light);
}
function fit(points, preset = S.view) {
  const b = new T.Box3().setFromPoints(points), c = b.getCenter(new T.Vector3()), s = b.getSize(new T.Vector3());
  const r = Math.max(s.length() * 0.42, 1.15);
  const map = {iso: new T.Vector3(1.85, 1.18, 1.72), front: new T.Vector3(0.01, 0.72, 2.7), side: new T.Vector3(2.7, 0.74, 0.01), top: new T.Vector3(0.01, 3.15, 0.01)};
  cam.position.copy(c.clone().add(map[preset].multiplyScalar(r)));
  ctrl.target.copy(c);
  ctrl.update();
}
function stat(raw, c, a) {
  ui.xv.textContent = raw.x.toFixed(4);
  ui.yv.textContent = raw.y.toFixed(4);
  ui.zv.textContent = raw.z.toFixed(4);
  ui.rv.textContent = raw.length().toFixed(4);
  ui.pv.textContent = c.label;
  if (c.phi) {
    const phi = c.phi.reduce((acc, i) => acc + (a[i] || 0), 0);
    ui.olab.textContent = 'phi';
    ui.ov.textContent = `${phi.toFixed(1)} deg`;
  } else {
    const p = S.pts.at(-1), yaw = deg(Math.atan2(p.z, p.x));
    ui.olab.textContent = 'tool yaw';
    ui.ov.textContent = `${yaw.toFixed(1)} deg`;
  }
}
function setNote(text) { ui.note.textContent = text; }
function updateTrail(p) {
  if (!S.showTrail) { trail.visible = false; return; }
  const last = S.trail.at(-1);
  if (!last || last.distanceTo(p) > 0.016) {
    S.trail.push(p.clone());
    if (S.trail.length > 240) S.trail.shift();
    if (S.trail.length < 2) {
      trail.visible = false;
      trailGeo.setPositions([p.x, p.y, p.z, p.x, p.y, p.z]);
      return;
    }
    trail.visible = true;
    const out = [];
    S.trail.forEach((v) => out.push(v.x, v.y, v.z));
    trailGeo.setPositions(out);
    trail.computeLineDistances();
  }
}
function clearTrail() {
  S.trail = [];
  trail.visible = false;
  trailGeo.setPositions([0, 0, 0, 0, 0, 0]);
}
function updatePose(a, refit = false) {
  const c = ROBOTS[S.robot], frames = c.f(a), rp = rawPts(frames, c), pts = rp.map((p) => viewP(p, c));
  S.pts = pts;
  for (let i = 0; i < O.links.length; i += 1) setLink(O.links[i], pts[i], pts[i + 1], c.r[i] || 0.028);
  for (let i = 0; i < O.joints.length; i += 1) {
    const p = pts[Math.min(i + 1, pts.length - 2)];
    O.joints[i].position.copy(p);
    setAxes(O.axes[i], frames[Math.min(i + 1, frames.length - 1)], c);
  }
  const ee = pts.at(-1);
  O.ee.position.copy(ee);
  O.light.position.copy(ee);
  glow.position.copy(ee);
  foot.position.set(ee.x, 0.004, ee.z);
  stem.geometry.setFromPoints([new T.Vector3(ee.x, 0.01, ee.z), ee.clone()]);
  stem.computeLineDistances();
  updateTrail(ee);
  stat(rp.at(-1), c, a);
  if (refit) fit(pts);
}

function refreshButtons() {
  ui.frames.classList.toggle('active', S.frames);
  ui.trail.classList.toggle('active', S.showTrail);
  ui.work.classList.toggle('active', S.work);
  ui.auto.classList.toggle('active', S.auto);
}
function sync(vals) {
  vals.forEach((v, i) => {
    const c = ROBOTS[S.robot].j[i], el = $(`j${i}`), tx = $(`v${i}`);
    if (el) el.value = String(v);
    if (tx) tx.textContent = `${v.toFixed(c.u === 'm' ? 2 : 0)} ${c.u}`;
  });
}
function setTarget(vals, doSync = true) {
  S.target = vals.slice();
  if (doSync) sync(vals);
}

function buildRobots() {
  ui.robots.innerHTML = '';
  Object.entries(ROBOTS).forEach(([k, c]) => {
    const b = document.createElement('button');
    b.className = `card${S.robot === k ? ' active' : ''}`;
    b.innerHTML = `<strong>${c.label}</strong><span>${c.blurb}</span>`;
    b.onclick = () => setRobot(k);
    ui.robots.appendChild(b);
  });
}
function buildPresets() {
  ui.presets.innerHTML = '';
  const c = ROBOTS[S.robot];
  Object.entries(c.p).forEach(([name, vals], i) => {
    const b = document.createElement('button');
    b.className = `btn${i === 0 ? ' active' : ''}`;
    b.innerHTML = `${name}<small>${vals.map((v, j) => `${c.j[j].l}:${v.toFixed(c.j[j].u === 'm' ? 2 : 0)}`).join(' | ')}</small>`;
    b.onclick = () => {
      ui.presets.querySelectorAll('.btn').forEach((n) => n.classList.remove('active'));
      b.classList.add('active');
      S.auto = false;
      refreshButtons();
      setTarget(vals);
      setNote(`${c.label} 已切到 ${name} 预设，可以观察这组关节变量对应的末端位姿。`);
    };
    ui.presets.appendChild(b);
  });
}
function buildSliders() {
  ui.sliders.innerHTML = '';
  const c = ROBOTS[S.robot];
  c.j.forEach((j, i) => {
    const d = document.createElement('div');
    d.className = 'row';
    d.innerHTML = `<div class="head"><span class="name">${j.l}</span><span class="val" id="v${i}">${j.v.toFixed(j.u === 'm' ? 2 : 0)} ${j.u}</span></div><input id="j${i}" type="range" min="${j.mn}" max="${j.mx}" step="${j.st || 1}" value="${j.v}">`;
    d.querySelector('input').oninput = () => {
      S.auto = false;
      refreshButtons();
      const vals = c.j.map((_, k) => parseFloat($(`j${k}`).value));
      setTarget(vals, true);
      setNote('手动关节控制中。场景会以平滑过渡的方式逼近目标姿态。');
    };
    ui.sliders.appendChild(d);
  });
}
function tryUpdateUrl(name) {
  try {
    if (location.protocol === 'http:' || location.protocol === 'https:') {
      history.replaceState(null, '', `?robot=${name}`);
    }
  } catch (error) {
    console.warn('Could not update URL state for FK Studio:', error);
  }
}

function setRobot(name) {
  S.robot = name;
  tryUpdateUrl(name);
  const c = ROBOTS[name];
  S.current = c.j.map((j) => j.v);
  S.target = c.j.map((j) => j.v);
  clearTrail();
  buildRobots();
  buildPresets();
  buildSliders();
  rebuild();
  ui.blurb.textContent = c.blurb;
  ui.ht.textContent = c.title;
  ui.hs.textContent = c.sub;
  workspace.visible = false;
  updatePose(S.current, true);
  showWorkspace();
  setNote(`${c.title} 已加载。你可以切视角、看局部坐标系，或打开 Workspace 点云。`);
}

const rng = (a, b, n) => Array.from({length: n}, (_, i) => a + (b - a) * (i / Math.max(n - 1, 1)));
function workColor(p) {
  const c = new T.Color();
  const h = ((Math.atan2(p.y, p.x) / (Math.PI * 2)) + 1) % 1;
  const l = 0.46 + Math.min(p.length() / 2.2, 1) * 0.18;
  c.setHSL(h, 0.72, l);
  return c;
}
function makeWorkspace(name) {
  const c = ROBOTS[name], pos = [], col = [];
  const push = (raw) => {
    const p = viewP(raw, c), cc = workColor(raw);
    pos.push(p.x, p.y, p.z);
    col.push(cc.r, cc.g, cc.b);
  };
  if (name === 'rrr') rng(-180, 180, 19).forEach((q1) => rng(-165, 165, 17).forEach((q2) => rng(-150, 150, 13).forEach((q3) => push(rawPts(c.f([q1, q2, q3]), c).at(-1)))));
  else if (name === 'puma') rng(-170, 170, 17).forEach((q1) => rng(-115, 115, 11).forEach((q2) => rng(-120, 120, 11).forEach((q3) => push(rawPts(c.f([q1, q2, q3, 0, 0, 0]), c).at(-1)))));
  else if (name === 'rrrp') rng(-170, 170, 13).forEach((q1) => rng(-170, 170, 13).forEach((q2) => rng(-150, 150, 11).forEach((q3) => rng(0, 0.6, 7).forEach((d4) => push(rawPts(c.f([q1, q2, q3, d4]), c).at(-1))))));
  else if (name === 'prrr') rng(0, 0.8, 9).forEach((d1) => rng(-170, 170, 13).forEach((q1) => rng(-150, 150, 11).forEach((q2) => rng(-150, 150, 11).forEach((q3) => push(rawPts(c.f([d1, q1, q2, q3]), c).at(-1))))));
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new T.Float32BufferAttribute(col, 3));
  return new T.Points(g, new T.PointsMaterial({size: name === 'puma' ? 0.04 : 0.032, vertexColors: true, transparent: true, opacity: 0.2, sizeAttenuation: true, depthWrite: false}));
}
function showWorkspace() {
  workspace.visible = S.work;
  if (!S.work) return;
  workspace.clear();
  if (S.cache.has(S.robot)) {
    workspace.add(S.cache.get(S.robot).clone());
    setNote('工作空间点云已显示。首次采样后会直接复用缓存。');
    return;
  }
  if (S.busy) return;
  S.busy = true;
  setNote('正在采样当前机构的工作空间点云，请稍候...');
  setTimeout(() => {
    const cloud = makeWorkspace(S.robot);
    S.cache.set(S.robot, cloud);
    workspace.add(cloud.clone());
    S.busy = false;
    setNote('工作空间点云采样完成。现在可以更直观看到关节限制如何塑造可达域。');
  }, 20);
}

function resize() {
  const w = vp.clientWidth, h = vp.clientHeight;
  ren.setSize(w, h);
  composer.setSize(w, h);
  cam.aspect = w / h;
  cam.updateProjectionMatrix();
  trailMat.resolution.set(w, h);
}
window.addEventListener('resize', resize);
resize();

ui.frames.onclick = () => { S.frames = !S.frames; refreshButtons(); O.axes.forEach((a) => { a.visible = S.frames; }); };
ui.trail.onclick = () => { S.showTrail = !S.showTrail; if (!S.showTrail) trail.visible = false; refreshButtons(); };
ui.work.onclick = () => { S.work = !S.work; refreshButtons(); showWorkspace(); };
ui.auto.onclick = () => { S.auto = !S.auto; refreshButtons(); setNote(S.auto ? 'Autoplay 已开启。系统会连续扫过多组关节变量，帮助你观察姿态变化与轨迹。' : 'Autoplay 已关闭。你现在可以手动精调关节。'); };
ui.home.onclick = () => ui.presets.querySelector('.btn')?.click();
ui.fit.onclick = () => { fit(S.pts, S.view); setNote('视角已重新对准当前机构。'); };
ui.clear.onclick = () => { clearTrail(); setNote('末端轨迹残影已清空。'); };
ui.cams.forEach((b) => { b.onclick = () => { ui.cams.forEach((x) => x.classList.remove('active')); b.classList.add('active'); S.view = b.dataset.v; fit(S.pts, S.view); }; });

refreshButtons();
buildRobots();
setRobot(initialRobot);

const clock = new T.Clock();
let time = 0;
function autoVals(t) {
  const c = ROBOTS[S.robot];
  return c.j.map((j, i) => {
    const amp = (j.mx - j.mn) * (j.u === 'm' ? 0.22 : 0.26);
    const speed = 0.55 + i * 0.15;
    const phase = i * 0.9 + (S.robot === 'puma' ? 0.4 : 0);
    return T.MathUtils.clamp(j.v + amp * Math.sin(t * speed + phase), j.mn, j.mx);
  });
}
(function loop() {
  requestAnimationFrame(loop);
  const dt = Math.min(clock.getDelta(), 0.05);
  time += dt;
  if (S.auto) setTarget(autoVals(time), true);
  let changed = false;
  S.current = S.current.map((v, i) => {
    const n = T.MathUtils.damp(v, S.target[i], 8, dt);
    if (Math.abs(n - v) > 1e-4) changed = true;
    return n;
  });
  if (changed || S.auto) updatePose(S.current);
  glow.rotation.y += 0.01;
  glow.scale.setScalar(1 + Math.sin(time * 2.6) * 0.08);
  foot.scale.setScalar(1 + Math.sin(time * 3) * 0.08);
  ctrl.update();
  composer.render();
})();
