// Retargeter viewer: plays human (SOMA) and robot (G1) motions served by `retarget view`.
// World frame: Z-up, metres, +X forward — the same as the Swift side.

import * as THREE from 'three';
import { OrbitControls } from '/web/vendor/OrbitControls.js';
import { GLTFLoader } from '/web/vendor/GLTFLoader.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const $ = (id) => document.getElementById(id);
const ui = {
  mode: $('mode'), layout: $('layout'), robot: $('robot'), robotMotion: $('robotMotion'),
  dir: $('dir'), file: $('file'), showSurface: $('showSurface'), showSkeleton: $('showSkeleton'),
  contactMode: $('contactMode'), tracks: $('tracks'), trackCanvas: $('trackCanvas'),
  play: $('play'), scrub: $('scrub'), time: $('time'), speed: $('speed'), loop: $('loop'), status: $('status'),
};
const SIDE_OFFSET = new THREE.Vector3(0, 1.2, 0); // human shown to the robot's left in side-by-side layout

// ---------------------------------------------------------------------------------------------
// Scene

const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark ? 0x15181e : 0xf4f5f7);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(3.2, -2.6, 1.7);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.5, 0.85);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, -4, 6);
scene.add(sun);

const grid = new THREE.GridHelper(10, 20, dark ? 0x56607a : 0x9aa3b2, dark ? 0x2b313c : 0xd5d9e0);
grid.rotation.x = Math.PI / 2; // GridHelper lies in XZ; our ground is XY
scene.add(grid);
scene.add(new THREE.AxesHelper(0.3));

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

// ---------------------------------------------------------------------------------------------
// Data

/** Fetches a packed payload: [u32 header length][JSON header][typed-array blocks]. */
async function fetchPacked(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${await response.text()}`);
  const buffer = await response.arrayBuffer();
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  const base = 4 + headerLength;
  const blocks = {};
  for (const b of header.blocks) {
    const Type = b.type === 'i32' ? Int32Array : Float32Array;
    blocks[b.name] = new Type(buffer, base + b.offset, b.count);
  }
  return { header, blocks };
}

const fetchJSON = async (url) => (await fetch(url)).json();
const quat = (a, i) => new THREE.Quaternion(a[i + 1], a[i + 2], a[i + 3], a[i]); // wxyz -> three.js xyzw

// ---------------------------------------------------------------------------------------------
// Robot

const gltfLoader = new GLTFLoader();
const robot = { root: new THREE.Group(), links: [], motion: null, model: null };
scene.add(robot.root);

async function loadRobot(name) {
  const model = await fetchJSON(`/api/robot?name=${encodeURIComponent(name)}`);
  const gltf = await gltfLoader.loadAsync(model.mesh);
  // Each top-level glTF node is one exported /Geometries/<name> (a Mesh, or a Group of Meshes).
  const meshes = new Map(gltf.scene.children.map((o) => [o.name, o]));

  robot.root.clear();
  robot.links = model.links.map((link) => {
    const group = new THREE.Group();
    group.name = link.name;
    for (const visual of link.visuals) {
      const source = meshes.get(visual.mesh);
      if (!source) { console.warn('missing mesh', visual.mesh); continue; }
      const mesh = source.clone();
      mesh.position.fromArray(visual.pos);
      mesh.quaternion.copy(quat(visual.quat, 0));
      mesh.traverse((o) => {
        if (o.isMesh) { o.geometry.computeVertexNormals(); o.material = o.material.clone(); o.material.flatShading = false; }
      });
      group.add(mesh);
    }
    robot.root.add(group);
    return group;
  });
  robot.model = model;
}

async function loadRobotMotion() {
  const name = encodeURIComponent(ui.robot.value), kind = ui.robotMotion.value;
  robot.verdict = '';
  if (kind === 'retarget' || kind === 'offline') {
    if (!human.file) return; // loaded again once a human clip is selected
    const file = human.file;
    const previous = ui.status.textContent.replace(/ · robot: .*$/, '');
    ui.status.textContent = `${previous} · retargeting onto ${ui.robot.value} (${kind === 'offline' ? 'offline' : 'realtime'})…`;
    const motion = await fetchPacked(`/api/retarget?name=${name}&file=${encodeURIComponent(file)}&mode=${kind === 'offline' ? 'offline' : 'realtime'}`);
    if (human.file !== file || ui.robotMotion.value !== kind) return; // superseded
    robot.motion = motion;
    const v = motion.header.validation;
    robot.verdict = v ? (v.status === 'accepted' ? 'robot: accepted' : `robot: REJECTED (${v.reasons.join('; ')})`) : '';
    ui.status.textContent = `${previous} · ${robot.verdict}`;
  } else {
    robot.motion = await fetchPacked(`/api/robot_demo?name=${name}&demo=${kind}`);
  }
  resetTimeline();
}

function poseRobot(t) {
  const m = robot.motion;
  if (!m || !robot.links.length) return;
  const { fps, frames, links } = m.header;
  const f = frameAt(t, fps, frames);
  const { pos, rot } = m.blocks;
  for (let i = 0; i < links; i++) {
    const k = f * links + i;
    robot.links[i].position.set(pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]);
    robot.links[i].quaternion.copy(quat(rot, 4 * k));
  }
}

// ---------------------------------------------------------------------------------------------
// Human

const human = { root: new THREE.Group(), motion: null, surface: null, skeleton: null, body: null, contacts: null, file: null };
scene.add(human.root);

// Contact display: estimated ground plane, ZMP marker, CoM ground projection.
const groundPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(4, 4),
  new THREE.MeshBasicMaterial({ color: 0x3bb273, transparent: true, opacity: 0.12, depthWrite: false, side: THREE.DoubleSide }));
const zmpMarker = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.045, 32),
  new THREE.MeshBasicMaterial({ color: 0x8e44ad, depthTest: false, side: THREE.DoubleSide }));
const comMarker = new THREE.Mesh(new THREE.CircleGeometry(0.02, 24),
  new THREE.MeshBasicMaterial({ color: 0x2f6fde, depthTest: false, side: THREE.DoubleSide }));
for (const m of [groundPlane, zmpMarker, comMarker]) { m.visible = false; m.renderOrder = 3; human.root.add(m); }

const COLORS = {
  skin: new THREE.Color(0x8fb3e8),
  contact: new THREE.Color(0x2fbf71),
  env: new THREE.Color(0xe0453a),
};

function buildSkeleton(header) {
  const pairs = [];
  header.parents.forEach((p, j) => { if (p > 0) pairs.push(p, j); }); // skip the virtual Root (index 0)
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs.length * 3), 3));
  const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xe0592a, depthTest: false }));
  lines.renderOrder = 2;
  lines.userData.pairs = pairs;
  lines.frustumCulled = false;
  return lines;
}

function buildBody(surface) {
  const { header, blocks } = surface;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(header.vertices * 3), 3));
  const colors = new Float32Array(header.vertices * 3);
  for (let i = 0; i < header.vertices; i++) COLORS.skin.toArray(colors, 3 * i);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(blocks.triangles), 1));
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7, transparent: true, opacity: 0.85 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}

async function loadSurface() {
  try { human.surface = await fetchPacked('/api/surface'); } catch { human.surface = null; }
  if (human.surface) { human.body = buildBody(human.surface); human.root.add(human.body); }
}

async function loadHuman(file) {
  human.file = file;
  human.contacts = null;
  trackLayout = null;
  ui.status.textContent = `loading ${file}…`;
  const motion = await fetchPacked(`/api/human?file=${encodeURIComponent(file)}`);
  if (human.skeleton) human.root.remove(human.skeleton);
  human.skeleton = buildSkeleton(motion.header);
  human.root.add(human.skeleton);
  human.motion = motion;
  const { frames, fps, joints } = motion.header;
  human.info = `${file}: ${frames} frames @ ${fps.toFixed(1)} fps, ${joints.length} joints`;
  focusHuman();
  ui.status.textContent = human.info;
  resetTimeline();
  await Promise.all([loadContacts(), ['retarget', 'offline'].includes(ui.robotMotion.value) ? loadRobotMotion() : null]);
}

/** Points the camera at the human's pelvis in the first frame (clips start anywhere in the capture volume). */
function focusHuman() {
  const m = human.motion;
  if (!m) return;
  const hips = Math.max(0, m.header.joints.indexOf('Hips'));
  const p = m.blocks.pos;
  const target = new THREE.Vector3(p[3 * hips], p[3 * hips + 1], 0.8).add(human.root.position);
  if (ui.mode.value === 'both') target.lerp(new THREE.Vector3(0, 0, 0.8), 0.5);
  camera.position.add(target.clone().sub(controls.target));
  controls.target.copy(target);
}

async function loadContacts() {
  human.contacts = null;
  drawTracks();
  if (!human.file || ui.contactMode.value === 'off' || !human.surface) { applyVisibility(); return; }
  const file = human.file, mode = ui.contactMode.value;
  ui.status.textContent = `${human.info} · analysing contacts (${mode})…`;
  const contacts = await fetchPacked(`/api/contacts?file=${encodeURIComponent(file)}&mode=${mode}`);
  if (human.file !== file || ui.contactMode.value !== mode) return; // superseded
  human.contacts = contacts;
  const { flags, groundTilt } = contacts.header;
  ui.status.textContent = `${human.info} · ground ${contacts.blocks.ground[0].toFixed(3)} m, tilt ${groundTilt.toFixed(1)}° · `
    + (flags.length ? `flags: ${flags.join(', ')}` : 'no flags') + (robot.verdict ? ` · ${robot.verdict}` : '');
  drawTracks();
  applyVisibility();
}

// ---------------------------------------------------------------------------------------------
// Contact tracks: one row per body part that ever touches, plus clip-level flags.

const TRACK_ROW = 14, TRACK_LABEL = 120;
let trackLayout = null;

function trackRows(c) {
  const { groups, frames } = c.header;
  const G = groups.length;
  const rows = [];
  for (let g = 0; g < G; g++) {
    let any = false, env = false;
    for (let f = 0; f < frames && !(any && env); f++) {
      any ||= c.blocks.contact[f * G + g] === 1;
      env ||= c.blocks.envContact[f * G + g] === 1;
    }
    if (any || env) rows.push({ label: groups[g], color: '#2fbf71', on: (f) => c.blocks.contact[f * G + g] === 1,
                                env: (f) => c.blocks.envContact[f * G + g] === 1 });
  }
  const flag = (name, block, color) => rows.push({ label: name, color, on: (f) => c.blocks[block][f] === 1 });
  flag('env support', 'envSupport', '#e0453a');
  flag('unsupported', 'unsupported', '#e8a33d');
  flag('flight', 'flight', '#6aa0ff');
  flag('penetration', 'penetration', '#b05cd6');
  return rows;
}

function drawTracks() {
  const c = human.contacts;
  ui.tracks.hidden = !c || ui.mode.value === 'robot';
  if (!c) { trackLayout = null; return; }
  const rows = trackRows(c);
  const canvas = ui.trackCanvas, dpr = window.devicePixelRatio || 1;
  const width = ui.tracks.clientWidth - 32, height = rows.length * TRACK_ROW + 4;
  canvas.style.height = `${height}px`;
  canvas.width = Math.max(1, width * dpr);
  canvas.height = height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const style = getComputedStyle(document.documentElement);
  const frames = c.header.frames, plot = width - TRACK_LABEL;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  rows.forEach((row, i) => {
    const y = i * TRACK_ROW;
    ctx.fillStyle = style.getPropertyValue('--muted');
    ctx.fillText(row.label, 0, y + TRACK_ROW / 2);
    ctx.fillStyle = style.getPropertyValue('--border');
    ctx.fillRect(TRACK_LABEL, y + 3, plot, TRACK_ROW - 6);
    for (let f = 0; f < frames; f++) {
      const env = row.env && row.env(f);
      if (!env && !row.on(f)) continue;
      ctx.fillStyle = env ? '#e0453a' : row.color;
      const x0 = TRACK_LABEL + (f / frames) * plot;
      ctx.fillRect(x0, y + 3, Math.max(1, plot / frames + 0.5), TRACK_ROW - 6);
    }
  });
  const image = document.createElement('canvas');
  image.width = canvas.width;
  image.height = canvas.height;
  image.getContext('2d').drawImage(canvas, 0, 0);
  trackLayout = { image, width, height, plot, frames };
}

function drawPlayhead(t) {
  const c = human.contacts;
  if (!trackLayout || !c || ui.tracks.hidden) return;
  const ctx = ui.trackCanvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ui.trackCanvas.width, ui.trackCanvas.height);
  ctx.drawImage(trackLayout.image, 0, 0);
  const f = frameAt(t, c.header.fps, c.header.frames);
  const dpr = window.devicePixelRatio || 1;
  const x = (TRACK_LABEL + (f / trackLayout.frames) * trackLayout.plot) * dpr;
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text');
  ctx.fillRect(x, 0, Math.max(1, dpr), trackLayout.height * dpr);
}

ui.trackCanvas.addEventListener('click', (e) => {
  if (!trackLayout || !human.motion) return;
  const rect = ui.trackCanvas.getBoundingClientRect();
  const u = (e.clientX - rect.left - TRACK_LABEL) / trackLayout.plot;
  if (u < 0 || u > 1) return;
  clock.t = u * durationOf(human.motion);
  setPlaying(false);
});

// Scratch for skinning: per joint 3x3 rotation (row-major) + translation of world * inverse(bind).
let skinR = new Float32Array(0), skinT = new Float32Array(0);

function poseHuman(t) {
  const m = human.motion;
  if (!m) return;
  const { fps, frames, joints, parents } = m.header;
  const J = joints.length;
  const f = frameAt(t, fps, frames);
  const { pos, rot } = m.blocks;

  if (human.skeleton && human.skeleton.visible) {
    const attr = human.skeleton.geometry.attributes.position;
    const pairs = human.skeleton.userData.pairs;
    for (let i = 0; i < pairs.length; i++) {
      const k = f * J + pairs[i];
      attr.array[3 * i] = pos[3 * k]; attr.array[3 * i + 1] = pos[3 * k + 1]; attr.array[3 * i + 2] = pos[3 * k + 2];
    }
    attr.needsUpdate = true;
  }

  if (human.body && human.body.visible) {
    const { header, blocks } = human.surface;
    if (skinR.length !== J * 9) { skinR = new Float32Array(J * 9); skinT = new Float32Array(J * 3); }
    const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), v = new THREE.Vector3(), m3 = new THREE.Matrix4();
    for (let j = 0; j < J; j++) {
      const k = f * J + j;
      qa.copy(quat(rot, 4 * k));
      qb.copy(quat(blocks.invBindRot, 4 * j));
      const q = qa.clone().multiply(qb);
      m3.makeRotationFromQuaternion(q);
      const e = m3.elements; // column-major
      skinR.set([e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]], 9 * j);
      v.set(blocks.invBindPos[3 * j], blocks.invBindPos[3 * j + 1], blocks.invBindPos[3 * j + 2]).applyQuaternion(qa);
      skinT[3 * j] = v.x + pos[3 * k]; skinT[3 * j + 1] = v.y + pos[3 * k + 1]; skinT[3 * j + 2] = v.z + pos[3 * k + 2];
    }
    const out = human.body.geometry.attributes.position.array;
    const rest = blocks.rest, wj = blocks.weightJoints, wv = blocks.weightValues, K = header.influences;
    for (let i = 0; i < header.vertices; i++) {
      const x = rest[3 * i], y = rest[3 * i + 1], z = rest[3 * i + 2];
      let px = 0, py = 0, pz = 0;
      for (let k = 0; k < K; k++) {
        const w = wv[i * K + k];
        if (w === 0) continue;
        const j = wj[i * K + k], r = 9 * j, s = 3 * j;
        px += w * (skinR[r] * x + skinR[r + 1] * y + skinR[r + 2] * z + skinT[s]);
        py += w * (skinR[r + 3] * x + skinR[r + 4] * y + skinR[r + 5] * z + skinT[s + 1]);
        pz += w * (skinR[r + 6] * x + skinR[r + 7] * y + skinR[r + 8] * z + skinT[s + 2]);
      }
      out[3 * i] = px; out[3 * i + 1] = py; out[3 * i + 2] = pz;
    }
    human.body.geometry.attributes.position.needsUpdate = true;
    human.body.geometry.computeVertexNormals();
    colorContacts(f);
  }
  poseContactMarkers(f);
}

function colorContacts(f) {
  const c = human.contacts, vg = human.surface.blocks.vertexGroup;
  const colors = human.body.geometry.attributes.color;
  if (!vg) return;
  const G = c ? c.header.groups.length : 0;
  for (let i = 0; i < vg.length; i++) {
    const g = vg[i];
    let color = COLORS.skin;
    if (c && g >= 0) {
      if (c.blocks.envContact[f * G + g] === 1) color = COLORS.env;
      else if (c.blocks.contact[f * G + g] === 1) color = COLORS.contact;
    }
    color.toArray(colors.array, 3 * i);
  }
  colors.needsUpdate = true;
}

function poseContactMarkers(f) {
  const c = human.contacts;
  const show = !!c;
  groundPlane.visible = zmpMarker.visible = comMarker.visible = show;
  if (!c) return;
  const ground = c.blocks.ground[Math.min(f, c.header.frames - 1)];
  const com = c.blocks.com, zmp = c.blocks.zmp;
  groundPlane.position.set(com[3 * f], com[3 * f + 1], ground);
  groundPlane.material.color.set(c.blocks.envSupport[f] ? 0xe0453a : c.blocks.unsupported[f] ? 0xe8a33d : 0x2fbf71);
  zmpMarker.position.set(zmp[2 * f], zmp[2 * f + 1], ground + 0.003);
  comMarker.position.set(com[3 * f], com[3 * f + 1], ground + 0.004);
}

// ---------------------------------------------------------------------------------------------
// Playback

const clock = { t: 0, playing: true, last: performance.now() };

function durationOf(m) { return m ? m.header.frames / m.header.fps : 0; }
function activeMotions() {
  const mode = ui.mode.value;
  return [mode !== 'human' ? robot.motion : null, mode !== 'robot' ? human.motion : null].filter(Boolean);
}
function duration() { return Math.max(0, ...activeMotions().map(durationOf)); }

/** Frame index for time t; each motion loops on its own length when shorter than the timeline. */
function frameAt(t, fps, frames) {
  const f = Math.floor(t * fps);
  return ui.loop.checked ? ((f % frames) + frames) % frames : Math.min(Math.max(f, 0), frames - 1);
}

function resetTimeline() {
  clock.t = 0;
  applyVisibility();
}

function applyVisibility() {
  const mode = ui.mode.value;
  robot.root.visible = mode !== 'human';
  human.root.visible = mode !== 'robot';
  if (human.skeleton) human.skeleton.visible = ui.showSkeleton.checked;
  if (human.body) human.body.visible = ui.showSurface.checked;
  human.root.position.copy(mode === 'both' && ui.layout.value === 'side' ? SIDE_OFFSET : new THREE.Vector3());
  const tracksHidden = !human.contacts || mode === 'robot';
  if (ui.tracks.hidden !== tracksHidden) { ui.tracks.hidden = tracksHidden; drawTracks(); }
}

function tick(now) {
  const dt = (now - clock.last) / 1000;
  clock.last = now;
  const total = duration();
  if (clock.playing && total > 0) {
    clock.t += dt * Number(ui.speed.value);
    if (clock.t > total) clock.t = ui.loop.checked ? clock.t % total : total;
  }
  if (robot.root.visible) poseRobot(clock.t);
  if (human.root.visible) poseHuman(clock.t);
  ui.scrub.value = total > 0 ? Math.round((clock.t / total) * 1000) : 0;
  drawPlayhead(clock.t);
  ui.time.textContent = `${clock.t.toFixed(2)} s`;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

function setPlaying(playing) {
  clock.playing = playing;
  ui.play.textContent = playing ? '❚❚' : '▶';
}

// ---------------------------------------------------------------------------------------------
// UI wiring

ui.play.addEventListener('click', () => setPlaying(!clock.playing));
ui.scrub.addEventListener('input', () => { clock.t = (ui.scrub.value / 1000) * duration(); setPlaying(false); });
for (const el of [ui.mode, ui.layout, ui.showSurface, ui.showSkeleton]) el.addEventListener('change', applyVisibility);
ui.robot.addEventListener('change', async () => { await loadRobot(ui.robot.value); await loadRobotMotion(); });
ui.robotMotion.addEventListener('change', loadRobotMotion);
ui.dir.addEventListener('change', () => loadFiles(ui.dir.value));
ui.contactMode.addEventListener('change', loadContacts);
new ResizeObserver(() => drawTracks()).observe(ui.tracks);
ui.file.addEventListener('change', () => loadHuman(`${ui.dir.value}/${ui.file.value}`));
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!clock.playing); }
  const step = 1 / 120;
  if (e.code === 'ArrowRight') { setPlaying(false); clock.t = Math.min(duration(), clock.t + step); }
  if (e.code === 'ArrowLeft') { setPlaying(false); clock.t = Math.max(0, clock.t - step); }
});

function fillSelect(select, values, selected) {
  select.replaceChildren(...values.map((v) => new Option(v, v, false, v === selected)));
}

async function loadFiles(dir, selected) {
  const files = await fetchJSON(`/api/files?dir=${encodeURIComponent(dir)}`);
  fillSelect(ui.file, files, selected);
  if (files.length) await loadHuman(`${dir}/${ui.file.value}`);
}

async function init() {
  setPlaying(true);
  resize();
  requestAnimationFrame(tick);
  const robots = await fetchJSON('/api/robots');
  fillSelect(ui.robot, robots.map((r) => r.name));
  await loadRobot(ui.robot.value);
  await loadRobotMotion();
  await loadSurface();
  const dirs = await fetchJSON('/api/dirs');
  if (dirs.length) {
    fillSelect(ui.dir, dirs, dirs.includes('230322') ? '230322' : dirs[0]);
    await loadFiles(ui.dir.value, 'sweep_floor_side_walk_R_004__A298.bvh');
  } else {
    ui.status.textContent = 'no human motions found (retarget view --motions DIR)';
  }
}

init().catch((e) => { ui.status.textContent = `error: ${e.message}`; console.error(e); });

// Handles for debugging from the browser console.
window.viewer = { scene, camera, controls, robot, human, clock };
