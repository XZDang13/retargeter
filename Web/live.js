// Live teleop viewer: robot poses and human skeletons pushed by `retarget teleop --viewer-port N` over /ws.
// Each stream (server slot) gets its own robot, placed side by side; the human skeleton is drawn beside it.
// Binary frames (float32): [slot, seq, receive->emit ms, cost, flags, ground, links L, joints J,
//                          L x (pos 3, quat wxyz 4), J x pos 3]; text frames: {"type": "streams", ...}.

import * as THREE from 'three';
import { OrbitControls } from '/web/vendor/OrbitControls.js';
import { GLTFLoader } from '/web/vendor/GLTFLoader.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const $ = (id) => document.getElementById(id);
const SPACING = 1.6;      // m between streams along -Y
const HUMAN_OFFSET = 0.8; // human skeleton to the robot's left
const FLAGS = { 1: 'clamped', 2: 'cold', 4: 'unsupported', 8: 'penetration', 16: 'env support', 32: 'flight', 64: 'dropped' };

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
controls.target.set(0, 0, 0.85);
controls.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, -4, 6);
scene.add(sun);
const grid = new THREE.GridHelper(20, 40, dark ? 0x56607a : 0x9aa3b2, dark ? 0x2b313c : 0xd5d9e0);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

const quat = (a, i) => new THREE.Quaternion(a[i + 1], a[i + 2], a[i + 3], a[i]); // wxyz -> xyzw

// Robot model (links + visual meshes), cloned per stream.
let template = null;
async function loadTemplate() {
  const model = await (await fetch('/api/robot')).json();
  const gltf = await new GLTFLoader().loadAsync(model.mesh);
  const meshes = new Map(gltf.scene.children.map((o) => [o.name, o]));
  template = model.links.map((link) => {
    const group = new THREE.Group();
    for (const visual of link.visuals) {
      const source = meshes.get(visual.mesh);
      if (!source) continue;
      const mesh = source.clone();
      mesh.position.fromArray(visual.pos);
      mesh.quaternion.copy(quat(visual.quat, 0));
      mesh.traverse((o) => { if (o.isMesh) o.geometry.computeVertexNormals(); });
      group.add(mesh);
    }
    return group;
  });
}

const streams = new Map(); // slot -> { root, links, skeleton, pairs, info }

function makeStream(slot, parents) {
  const root = new THREE.Group();
  root.position.set(0, -SPACING * streams.size, 0);
  const links = template.map((g) => { const c = g.clone(); root.add(c); return c; });
  const pairs = [];
  parents.forEach((p, j) => { if (p > 0) pairs.push(p, j); }); // skip the virtual Root (index 0)
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs.length * 3), 3));
  const skeleton = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xe0592a }));
  skeleton.frustumCulled = false;
  skeleton.position.set(0, HUMAN_OFFSET, 0);
  root.add(skeleton);
  scene.add(root);
  return { root, links, skeleton, pairs, info: { frames: 0, since: performance.now(), rate: 0 } };
}

function updateStreams(message) {
  const wanted = new Map(message.streams.map((s) => [s.slot, s.parents]));
  for (const [slot, s] of streams) if (!wanted.has(slot)) { scene.remove(s.root); streams.delete(slot); }
  for (const [slot, parents] of wanted) if (!streams.has(slot)) streams.set(slot, makeStream(slot, parents));
  let k = 0;
  for (const s of [...streams.keys()].sort((a, b) => a - b).map((slot) => streams.get(slot))) s.root.position.y = -SPACING * k++;
  if (!streams.size) $('streams').textContent = 'no streams';
}

function applyFrame(buffer) {
  const v = new Float32Array(buffer);
  const slot = v[0], L = v[6], J = v[7];
  const s = streams.get(slot);
  if (!s || s.links.length !== L) return;
  let at = 8;
  for (let i = 0; i < L; i++, at += 7) {
    s.links[i].position.set(v[at], v[at + 1], v[at + 2]);
    s.links[i].quaternion.copy(quat(v, at + 3));
  }
  const joints = v.subarray(at, at + 3 * J);
  const pos = s.skeleton.geometry.attributes.position;
  s.pairs.forEach((j, k) => pos.setXYZ(k, joints[3 * j], joints[3 * j + 1], joints[3 * j + 2]));
  pos.needsUpdate = true;
  const flags = Object.entries(FLAGS).filter(([bit]) => v[4] & bit).map(([, name]) => name);
  s.info = { ...s.info, seq: v[1], server: v[2], cost: v[3], flags, frames: s.info.frames + 1, root: [v[8], v[9]] };
}

function statusLine() {
  const now = performance.now();
  const parts = [];
  for (const [slot, s] of [...streams].sort((a, b) => a[0] - b[0])) {
    const dt = (now - s.info.since) / 1000;
    if (dt >= 1) { s.info.rate = s.info.frames / dt; s.info.frames = 0; s.info.since = now; }
    parts.push(`slot ${slot}: ${s.info.rate.toFixed(0)} fps shown, seq ${s.info.seq ?? '-'}, server ${s.info.server?.toFixed(2) ?? '-'} ms`
      + (s.info.flags?.length ? ` [${s.info.flags.join(', ')}]` : ''));
  }
  if (parts.length) $('streams').textContent = parts.join('  ·  ');
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { $('status').textContent = 'connected'; };
  ws.onclose = () => { $('status').textContent = 'disconnected, retrying…'; setTimeout(connect, 1000); };
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      const message = JSON.parse(event.data);
      if (message.type === 'streams') {
        $('status').textContent = `connected · ${message.robot}`;
        updateStreams(message);
      }
    } else {
      applyFrame(event.data);
    }
  };
}

let lastStatus = 0;
function tick(now) {
  if (now - lastStatus > 250) { statusLine(); lastStatus = now; }
  const human = $('showHuman').checked;
  for (const s of streams.values()) s.skeleton.visible = human;
  if ($('follow').checked && streams.size) {
    const first = streams.get([...streams.keys()].sort((a, b) => a - b)[0]);
    if (first.info.root) {
      const target = new THREE.Vector3(first.info.root[0], first.info.root[1] + first.root.position.y, 0.85);
      const delta = target.clone().sub(controls.target);
      controls.target.add(delta);
      camera.position.add(delta);
    }
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

await loadTemplate();
connect();
requestAnimationFrame(tick);
