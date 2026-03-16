import * as THREE from "three";

const width = 1340;
const height = 620;

// --- レンダラー ---
const renderer = new THREE.WebGLRenderer({
  canvas: document.querySelector("#myCanvas"),
});
renderer.setSize(width, height);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0xeeeeee);

// --- シーン ---
const scene = new THREE.Scene();

// --- カメラ ---
const camera = new THREE.PerspectiveCamera(45, width / height, 1, 10000);
camera.position.set(0, -5000, 500); // 斜め上から見る
camera.lookAt(0, 0, 0);

// --- ライト ---
const light = new THREE.DirectionalLight(0xffffff, 3);
light.position.set(15, -500, 15);
scene.add(light);



// --- 壁 ---
const wallMaterial = new THREE.MeshStandardMaterial({
  color: 0x888888,
  roughness: 0.6,
  metalness: 0.1,
});
const wallThickness = 100;
const limit = 2250;
const depthLimit = 1500;

const createWall = (x, y, z, w, h, d) => {
  const wall = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), wallMaterial);
  wall.position.set(x, y, z);
  scene.add(wall);
  return wall;
};

// --- 汎用：壁の面にグリッド線を貼る関数 ---
function addGridToWallMesh(wallMesh, options = {}) {
  const {
    divisionsU = 20,    // 面の幅方向の分割数（縦線の本数）
    divisionsV = 20,    // 面の高さ方向の分割数（横線の本数）
    color = 0x000000,
    opacity = 0.6,
    offset = 0.2,       // 壁からの微小オフセット（z-fighting対策）
    alwaysOnTop = false // trueにすると他のメッシュより常に前に描画（depthTest=false）
  } = options;

  // geometryのパラメータ（BoxGeometryを前提）
  const geom = wallMesh.geometry;
  const w = geom.parameters.width;
  const h = geom.parameters.height;
  const d = geom.parameters.depth;

  // どの軸が「厚み（最小）」かを判別
  let normalAxis = 0; // 0 -> X, 1 -> Y, 2 -> Z
  const dims = [w, h, d];
  if (h <= w && h <= d) normalAxis = 1;
  else if (d <= w && d <= h) normalAxis = 2;
  else normalAxis = 0;

  // planeU, planeV の大きさ（グリッド面の二方向）
  let sizeU, sizeV;
  if (normalAxis === 0) { sizeU = h; sizeV = d; } // 厚みがXなら面は YZ
  else if (normalAxis === 1) { sizeU = w; sizeV = d; } // 厚みがYなら面は XZ (上面)
  else { sizeU = w; sizeV = h; } // 厚みがZなら面は XY (前面/背面)

  // 分割数に基づくステップ
  const stepU = sizeU / divisionsU;
  const stepV = sizeV / divisionsV;

  // 線の本数 = (divisionsU + 1) 縦線 + (divisionsV + 1) 横線
  const linesCount = (divisionsU + 1) + (divisionsV + 1);
  // 各線は2点 -> 総頂点数
  const positions = new Float32Array(linesCount * 2 * 3);
  let idx = 0;

  // 中心基準で作る（local座標で -size/2 〜 +size/2）
  // 縦線（U方向に沿って等間隔、各線は V全体を描く）
  for (let i = 0; i <= divisionsU; i++) {
    const u = -sizeU / 2 + i * stepU;
    // point A (u, -sizeV/2), point B (u, +sizeV/2)
    let ax, ay, az, bx, by, bz;
    if (normalAxis === 0) {
      // plane = YZ => map (u,v) -> (x=u_thickness? no) -> actually u->y, v->z, x = +thickness/2
      ax = 0; ay = u; az = -sizeV / 2;
      bx = 0; by = u; bz = sizeV / 2;
    } else if (normalAxis === 1) {
      // plane = XZ => u->x, v->z
      ax = u; ay = 0; az = -sizeV / 2;
      bx = u; by = 0; bz = sizeV / 2;
    } else {
      // normalAxis === 2, plane = XY => u->x, v->y
      ax = u; ay = -sizeV / 2; az = 0;
      bx = u; by = sizeV / 2; bz = 0;
    }
    positions[idx++] = ax; positions[idx++] = ay; positions[idx++] = az;
    positions[idx++] = bx; positions[idx++] = by; positions[idx++] = bz;
  }

  

  // 横線（V方向に沿って等間隔、各線は U全体を描く）
  for (let j = 0; j <= divisionsV; j++) {
    const v = -sizeV / 2 + j * stepV;
    let ax, ay, az, bx, by, bz;
    if (normalAxis === 0) {
      // plane = YZ => u->y, v->z
      ax = 0; ay = -sizeU / 2; az = v;
      bx = 0; by = sizeU / 2;  bz = v;
    } else if (normalAxis === 1) {
      // plane = XZ => u->x, v->z
      ax = -sizeU / 2; ay = 0; az = v;
      bx = sizeU / 2;  by = 0; bz = v;
    } else {
      // plane = XY => u->x, v->y
      ax = -sizeU / 2; ay = v; az = 0;
      bx = sizeU / 2;  by = v; bz = 0;
    }
    positions[idx++] = ax; positions[idx++] = ay; positions[idx++] = az;
    positions[idx++] = bx; positions[idx++] = by; positions[idx++] = bz;
  }

  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const lineMat = new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1 ? true : false,
    opacity: opacity,
    depthTest: !alwaysOnTop // alwaysOnTop -> depthTest=false
    // 注意: linewidth は多くの環境で効かない場合がある（WebGLの制約）
  });

  const lines = new THREE.LineSegments(lineGeom, lineMat);

  // ローカル平面で作ったので、wallMeshの回転・位置に合わせる
  // まず wall のローカル回転をコピー
  lines.rotation.copy(wallMesh.rotation);

  // lines の位置は wallMesh.position に合わせ、さらに法線方向へ微小オフセット
  // 法線ベクトル（ワールド軸に対応）を取得
  const normal = new THREE.Vector3();
  if (normalAxis === 0) normal.set(1, 0, 0);
  else if (normalAxis === 1) normal.set(0, 1, 0);
  else normal.set(0, 0, 1);

  // wallMesh の回転を考慮して法線をワールドに回す（回転のみ）
  normal.applyEuler(wallMesh.rotation);

  // オフセット量（壁の外側に出すなら +、内側に入れるなら -）
  const sign = 1; // 1: 表面の外側へ出す
  const offVec = normal.clone().multiplyScalar((Math.min(w, h, d) / 2) + offset * sign);

  // 最終位置 = wallMesh.position + offVec
  lines.position.copy(wallMesh.position).add(offVec);

  // z-fighting がまだ起きる時は renderOrder を上げるか depthTest=false を試す
  if (alwaysOnTop) {
    lines.renderOrder = 999;
  }
  

  scene.add(lines);
  return lines; // 必要なら参照を保持して後で削除できます
}



// --- 既存の壁作成を配列にして、グリッドを追加する例 ---
const walls = [];
walls.push(createWall(0, limit + wallThickness / 2, 0, 4600 + wallThickness, wallThickness, 3000)); // 上
walls.push(createWall(limit + wallThickness / 2, 0, 0, wallThickness, 5000 + wallThickness, 3000)); // 右
walls.push(createWall(-limit - wallThickness / 2, 0, 0, wallThickness, 5000 + wallThickness, 3000)); // 左
walls.push(createWall(0, 0, -depthLimit - wallThickness / 2, 4600 + wallThickness, 5000 + wallThickness, wallThickness)); // 奥
walls.push(createWall(0, 0, depthLimit + wallThickness / 2, 4600 + wallThickness, 5000 + wallThickness, wallThickness)); // 手前

// 壁それぞれにグリッドを貼る
// const grids = [];
// walls.forEach((w) => {
//   const isBackWall = w.position.z > 0; // 奥（Zマイナス側）の壁か？
//   const isRightWall = w.position.x > 0; // 右
//   const isTopWall = w.position.y > 0; // 上

//   // 奥・右・上の壁は裏向きなので offset を逆にする
//   const reverse = isBackWall || isRightWall || isTopWall;
//   const grid = addGridToWallMesh(w, {
//     divisionsU: 24,
//     divisionsV: 16,
//     color: 0xffffff,
//     opacity: 0.6,
//     offset: reverse ? -0.8 : 0.8,
//     alwaysOnTop: false,
//   });
//   grid.renderOrder = 1;
//   grids.push(grid);
// });
// 壁それぞれにグリッドを貼る
// 壁それぞれにグリッドを貼る
// 壁それぞれにグリッドを貼る
const grids = [];
walls.forEach((w) => {
  const isBackWall = w.position.z < 0; // 奥
  const isRightWall = w.position.x > 0; // 右
  const isTopWall = w.position.y > 0; // 上

  // 奥・右・上の壁は裏向きなので offset を逆にする
  const reverse = isBackWall || isRightWall || isTopWall;

  const grid = addGridToWallMesh(w, {
    divisionsU: 24,
    divisionsV: 16,
    color: 0xffffff,
    opacity: 0.7,
    offset: reverse ? -0.1 : 0.1, // ← 壁に非常に近い位置（0.1程度）
    alwaysOnTop: false,            // ← 壁より少し手前、でも他オブジェクトより奥
  });

  grid.renderOrder = 1; // 壁より少し後に描画
  grids.push(grid);
});

// 壁は renderOrder = 0 としておく（または省略）
walls.forEach(w => {
  w.renderOrder = 0;
});



// --- ボール ---
const radius = 150;
const ballGeo = new THREE.SphereGeometry(radius, 32, 32);
const ballMat = new THREE.MeshStandardMaterial({ color: 0xff5533, roughness: 0.4 });
const ball = new THREE.Mesh(ballGeo, ballMat);
scene.add(ball);
ball.position.set(0, -1500, 0);

// --- パドル ---
const paddleWidth = 800;
const paddleHeight = 200;
const paddleDepth = 800;
const paddleGeo = new THREE.BoxGeometry(paddleWidth, paddleHeight, paddleDepth);
const paddleMat = new THREE.MeshStandardMaterial({ color: 0x00cc88 });
const paddle = new THREE.Mesh(paddleGeo, paddleMat);
scene.add(paddle);
paddle.position.set(0, -2100, 0);



// --- ブロック（Z方向も複数層配置） ---
const blockGeo = new THREE.BoxGeometry(400, 400, 400);
const blockMat = new THREE.MeshStandardMaterial({ color: 0x3399ff });
const blocks = [];

for (let i = -3; i <= 3; i++) {
  for (let j = 1; j <= 3; j++) {
    for (let k = -1; k <= 1; k++) {
      const block = new THREE.Mesh(blockGeo, blockMat.clone());
      block.position.set(i * 500, j * 500 + 500, k * 500);
      scene.add(block);
      blocks.push(block);
    }
  }
}

function resetGame() {
  // ボール初期位置
  ball.position.set(0, -1500, 0);
  dir.set(1, 1, 1).normalize();
  speed = 12;

  // パドル初期位置
  paddle.position.set(0, -2100, 0);

  // ブロックを全削除して再作成
  blocks.forEach(b => scene.remove(b));
  blocks.length = 0;

  for (let i = -3; i <= 3; i++) {
    for (let j = 1; j <= 3; j++) {
      for (let k = -1; k <= 1; k++) {
        const block = new THREE.Mesh(blockGeo, blockMat.clone());
        block.position.set(i * 500, j * 500 + 500, k * 500);
        scene.add(block);
        blocks.push(block);
        score = 0; // スコアも初期化
      }
    }
  }

  isGameOver = false;
}

// --- 壁は奥（背景）に固定 ---
walls.forEach(w => {
  w.renderOrder = 0; // 一番奥
});

// --- ブロック・ボール・パドルは最前面に ---
[...blocks, ball, paddle].forEach(obj => {
  obj.renderOrder = 2; // 一番手前
});

// --- 操作キー ---
let keys = {};
window.addEventListener("keydown", (e) => (keys[e.key] = true));
window.addEventListener("keyup", (e) => (keys[e.key] = false));

// --- ゲームパラメータ ---
let dir = new THREE.Vector3(1, 1, 1).normalize(); // ← XYZ全方向対応
let speed = 12;
let isGameOver = false;
let score = 0; // スコア初期値

// --- メインループ ---
function tick() {
  if (!isGameOver) {
    document.getElementById("scoreDisplay").textContent = "Score: " + score;


    // パドル操作（左右 + 前後）
    if (keys["ArrowLeft"]) paddle.position.x -= 30;
    if (keys["ArrowRight"]) paddle.position.x += 30;
    if (keys["ArrowDown"]) paddle.position.z -= 30;   // 奥へ
    if (keys["ArrowUp"]) paddle.position.z += 30; // 手前へ

    paddle.position.x = THREE.MathUtils.clamp(paddle.position.x, -limit + paddleWidth / 2, limit - paddleWidth / 2);
    paddle.position.z = THREE.MathUtils.clamp(paddle.position.z, -depthLimit + paddleDepth / 2, depthLimit - paddleDepth / 2);

    // ボール移動
    ball.position.addScaledVector(dir, speed);

    // 壁反射（XYZ全方向）
    if (Math.abs(ball.position.x) > limit - radius) dir.x *= -1;
    if (ball.position.y > limit - radius) dir.y *= -1;
    if (Math.abs(ball.position.z) > depthLimit - radius) dir.z *= -1;

    // パドル衝突
    if (
      Math.abs(ball.position.x - paddle.position.x) < paddleWidth / 2 + radius &&
      Math.abs(ball.position.y - paddle.position.y) < paddleHeight / 2 + radius &&
      Math.abs(ball.position.z - paddle.position.z) < paddleDepth / 2 + radius &&
      dir.y < 0
    ) {
      dir.y = Math.abs(dir.y);
      const offsetX = (ball.position.x - paddle.position.x) / (paddleWidth / 2);
      const offsetZ = (ball.position.z - paddle.position.z) / (paddleDepth / 2);
      dir.x += offsetX * 0.5;
      dir.z += offsetZ * 0.5;
      dir.normalize();
    }

    // ブロック衝突
    for (let i = blocks.length - 1; i >= 0; i--) {
      const b = blocks[i];
      if (ball.position.distanceTo(b.position) < radius + 200) {
        scene.remove(b);
        blocks.splice(i, 1);
        dir.y *= -1;

        score += 10; // 1ブロック破壊で10点
        console.log("Score:", score); // とりあえずコンソール表示

        break;
      }
    }

    // 落下判定（下方向）
    if (ball.position.y < -limit) {
  isGameOver = true;
  console.log("ゲームオーバー！");
  alert("ゲームオーバー！");
  // 1秒後に自動で再スタート
  setTimeout(resetGame, 1000);
}

  }

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
