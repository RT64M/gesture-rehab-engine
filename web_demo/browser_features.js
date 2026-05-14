function normalizeLandmarks(rawLandmarks, config) {
  const points = rawLandmarks.map((point) => [Number(point.x ?? point[0]), Number(point.y ?? point[1])]);
  const wrist = points[config.wrist];
  const translated = points.map(([x, y]) => [x - wrist[0], y - wrist[1]]);
  const from = translated[config.scale_ref_from];
  const to = translated[config.scale_ref_to];
  const scale = Math.hypot(to[0] - from[0], to[1] - from[1]) || 1;
  return translated.map(([x, y]) => [x / scale, y / scale]);
}

function pcaAlign2d(points, config) {
  const fingerPoints = points.slice(1);
  const mean = fingerPoints.reduce(
    (acc, point) => [acc[0] + point[0] / fingerPoints.length, acc[1] + point[1] / fingerPoints.length],
    [0, 0],
  );
  let xx = 0;
  let xy = 0;
  let yy = 0;
  fingerPoints.forEach(([x, y]) => {
    const cx = x - mean[0];
    const cy = y - mean[1];
    xx += cx * cx;
    xy += cx * cy;
    yy += cy * cy;
  });
  const theta = 0.5 * Math.atan2(2 * xy, xx - yy);
  let axis = [Math.cos(theta), Math.sin(theta)];
  const target = [
    points[config.scale_ref_to][0] - points[config.wrist][0],
    points[config.scale_ref_to][1] - points[config.wrist][1],
  ];
  if (axis[0] * target[0] + axis[1] * target[1] < 0) {
    axis = [-axis[0], -axis[1]];
  }
  const [cosT, sinT] = axis;
  const rotation = [
    [sinT, -cosT],
    [cosT, sinT],
  ];
  return points.map(([x, y]) => [
    x * rotation[0][0] + y * rotation[0][1],
    x * rotation[1][0] + y * rotation[1][1],
  ]);
}

function jointAngle(points, a, b, c) {
  const v1 = [points[a][0] - points[b][0], points[a][1] - points[b][1]];
  const v2 = [points[c][0] - points[b][0], points[c][1] - points[b][1]];
  const n1 = Math.hypot(v1[0], v1[1]);
  const n2 = Math.hypot(v2[0], v2[1]);
  if (n1 < 1e-8 || n2 < 1e-8) {
    return 0;
  }
  const cosine = Math.min(1, Math.max(-1, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)));
  return Math.acos(cosine);
}

function distance(points, a, b) {
  return Math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]);
}

export function extractFeatureVector(rawLandmarks, landmarkConfig) {
  const normalized = normalizeLandmarks(rawLandmarks, landmarkConfig);
  const aligned = pcaAlign2d(normalized, landmarkConfig);
  const jointAngles = landmarkConfig.joint_triplets.map((item) => {
    const [a, b, c] = item.points;
    return jointAngle(aligned, a, b, c);
  });
  const tipDistances = landmarkConfig.tip_pairs.map((item) => {
    const [a, b] = item.points;
    return distance(aligned, a, b);
  });
  const spreadAngles = landmarkConfig.spread_triplets.map((item) => {
    const [a, b, c] = item.points;
    return jointAngle(aligned, a, b, c);
  });
  return [...jointAngles, ...tipDistances, ...spreadAngles];
}

export function normalizeForDrawing(rawLandmarks, landmarkConfig) {
  const normalized = normalizeLandmarks(rawLandmarks, landmarkConfig);
  const xs = normalized.map((point) => point[0]);
  const ys = normalized.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-8);
  const spanY = Math.max(maxY - minY, 1e-8);
  return normalized.map(([x, y]) => [(x - minX) / spanX, 1 - (y - minY) / spanY]);
}
