function dot(a, b) {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) {
    total += a[index] * b[index];
  }
  return total;
}

function subtract(a, b) {
  return a.map((value, index) => value - b[index]);
}

function multiplyMatrixVector(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function quadForm(vector, matrix) {
  return dot(vector, multiplyMatrixVector(matrix, vector));
}

function meanVector(samples) {
  const dimension = samples[0].length;
  const mean = new Array(dimension).fill(0);
  samples.forEach((sample) => {
    sample.forEach((value, index) => {
      mean[index] += value;
    });
  });
  return mean.map((value) => value / samples.length);
}

function covarianceMatrix(samples, fallback) {
  if (samples.length < 2) {
    return fallback.map((row) => row.slice());
  }
  const mean = meanVector(samples);
  const dimension = mean.length;
  const covariance = Array.from({ length: dimension }, () => new Array(dimension).fill(0));
  samples.forEach((sample) => {
    const diff = subtract(sample, mean);
    for (let row = 0; row < dimension; row += 1) {
      for (let column = 0; column < dimension; column += 1) {
        covariance[row][column] += diff[row] * diff[column];
      }
    }
  });
  const denominator = Math.max(samples.length - 1, 1);
  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < dimension; column += 1) {
      covariance[row][column] /= denominator;
    }
  }
  return covariance;
}

function invertMatrix(matrix) {
  const size = matrix.length;
  const augmented = matrix.map((row, index) => [
    ...row.map((value) => Number(value)),
    ...Array.from({ length: size }, (_, columnIndex) => (columnIndex === index ? 1 : 0)),
  ]);

  for (let pivot = 0; pivot < size; pivot += 1) {
    let maxRow = pivot;
    for (let row = pivot + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[maxRow][pivot])) {
        maxRow = row;
      }
    }
    [augmented[pivot], augmented[maxRow]] = [augmented[maxRow], augmented[pivot]];
    const pivotValue = augmented[pivot][pivot] || 1e-8;
    for (let column = 0; column < size * 2; column += 1) {
      augmented[pivot][column] /= pivotValue;
    }
    for (let row = 0; row < size; row += 1) {
      if (row === pivot) {
        continue;
      }
      const factor = augmented[row][pivot];
      for (let column = 0; column < size * 2; column += 1) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }

  return augmented.map((row) => row.slice(size));
}

function clip(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function humanizeFeatureName(name) {
  const fingerLabels = {
    thumb: "拇指",
    index: "食指",
    middle: "中指",
    ring: "无名指",
    pinky: "小指",
  };
  if (name.startsWith("dist_")) {
    const [left, right] = name.replace("dist_", "").replace("_tip", "").split("_");
    return `${fingerLabels[left] ?? left}-${fingerLabels[right] ?? right}指尖距离`;
  }
  if (name.startsWith("spread_")) {
    const [left, right] = name.replace("spread_", "").split("_");
    return `${fingerLabels[left] ?? left}-${fingerLabels[right] ?? right}张开角`;
  }
  const parts = name.split("_");
  if (parts.length < 2) {
    return name;
  }
  return `${fingerLabels[parts[0]] ?? parts[0]} ${parts[1].toUpperCase()} 关节角度`;
}

function directionText(name, zScore) {
  if (name.startsWith("dist_")) {
    return zScore > 0 ? "偏大" : "偏小";
  }
  if (name.startsWith("spread_")) {
    return zScore > 0 ? "张开过大" : "张开不足";
  }
  return zScore > 0 ? "偏大" : "偏小";
}

class BrowserUserModel {
  constructor(baseDescriptor, stage2Config) {
    this.baseDescriptor = baseDescriptor;
    this.stage2Config = stage2Config;
    this.samples = [];
    this.baselineDistances = [];
    this.cachedDescriptor = {
      mu: baseDescriptor.mu.slice(),
      sigma: baseDescriptor.sigma.map((row) => row.slice()),
      sigma_inv: baseDescriptor.sigma_inv.map((row) => row.slice()),
      sigma_diag: baseDescriptor.sigma_diag.slice(),
      feature_names: baseDescriptor.feature_names.slice(),
    };
  }

  get blendWeight() {
    return Math.min(this.samples.length / this.stage2Config.n_transition, this.stage2Config.lambda_max);
  }

  get adaptiveFactor() {
    if (this.baselineDistances.length === 0) {
      return 1;
    }
    const window = this.baselineDistances.slice(0, this.stage2Config.adaptive_window);
    const sorted = [...window].sort((left, right) => left - right);
    const median = sorted[Math.floor(sorted.length / 2)];
    return clip(median / this.baseDescriptor.expected_distance, 0.5, 2.0);
  }

  get tau() {
    return this.stage2Config.tau0 * this.adaptiveFactor;
  }

  mahalanobisAgainstBase(vector) {
    const diff = subtract(vector, this.baseDescriptor.mu);
    return Math.sqrt(Math.max(quadForm(diff, this.baseDescriptor.sigma_inv), 0));
  }

  recordAttempt(vector) {
    this.samples.push(vector.slice());
    this.baselineDistances.push(this.mahalanobisAgainstBase(vector));
    this.rebuildCache();
  }

  rebuildCache() {
    const weight = this.blendWeight;
    if (weight <= 0 || this.samples.length === 0) {
      this.cachedDescriptor = {
        mu: this.baseDescriptor.mu.slice(),
        sigma: this.baseDescriptor.sigma.map((row) => row.slice()),
        sigma_inv: this.baseDescriptor.sigma_inv.map((row) => row.slice()),
        sigma_diag: this.baseDescriptor.sigma_diag.slice(),
        feature_names: this.baseDescriptor.feature_names.slice(),
      };
      return;
    }

    const userMean = meanVector(this.samples);
    let userCovariance = covarianceMatrix(this.samples, this.baseDescriptor.sigma);
    userCovariance = userCovariance.map((row, rowIndex) =>
      row.map(
        (value, columnIndex) =>
          (1 - 0.1) * value + 0.1 * this.baseDescriptor.sigma[rowIndex][columnIndex] + (rowIndex === columnIndex ? 1e-4 : 0),
      ),
    );

    const blendedMu = this.baseDescriptor.mu.map((value, index) => (1 - weight) * value + weight * userMean[index]);
    const blendedSigma = this.baseDescriptor.sigma.map((row, rowIndex) =>
      row.map((value, columnIndex) => {
        const blended = (1 - weight) * value + weight * userCovariance[rowIndex][columnIndex];
        return blended + (rowIndex === columnIndex ? 1e-4 : 0);
      }),
    );

    this.cachedDescriptor = {
      mu: blendedMu,
      sigma: blendedSigma,
      sigma_inv: invertMatrix(blendedSigma),
      sigma_diag: blendedSigma.map((row, index) => Math.sqrt(Math.max(row[index], 1e-8))),
      feature_names: this.baseDescriptor.feature_names.slice(),
    };
  }
}

export class BrowserScorer {
  constructor(runtimeConfig) {
    this.runtimeConfig = runtimeConfig;
    this.stage2Config = runtimeConfig.stage2;
    this.descriptors = runtimeConfig.stage2.descriptors;
    this.gestureLabels = runtimeConfig.gesture_labels;
    this.userModels = new Map();
  }

  getUserModel(gesture) {
    if (!this.userModels.has(gesture)) {
      this.userModels.set(gesture, new BrowserUserModel(this.descriptors[gesture], this.stage2Config));
    }
    return this.userModels.get(gesture);
  }

  classify(vector) {
    const distances = {};
    Object.entries(this.descriptors).forEach(([gesture, descriptor]) => {
      const diff = subtract(vector, descriptor.mu);
      distances[gesture] = Math.sqrt(Math.max(quadForm(diff, descriptor.sigma_inv), 0));
    });
    const predictedGesture = Object.keys(distances).reduce((best, gesture) =>
      distances[gesture] < distances[best] ? gesture : best,
    );
    return { predictedGesture, populationDistances: distances };
  }

  summarizeDeviations(vector, descriptor, topK = 3) {
    const zScores = vector.map((value, index) => (value - descriptor.mu[index]) / descriptor.sigma_diag[index]);
    const sorted = descriptor.feature_names
      .map((feature, index) => ({
        feature,
        label: humanizeFeatureName(feature),
        z_score: zScores[index],
        magnitude: Math.abs(zScores[index]),
      }))
      .sort((left, right) => right.magnitude - left.magnitude)
      .slice(0, topK);
    return sorted;
  }

  generateFeedback(vector, descriptor, targetGesture, predictedGesture, topK = 3) {
    const deviations = this.summarizeDeviations(vector, descriptor, topK);
    const lines = [];
    if (predictedGesture !== targetGesture) {
      lines.push(
        `当前更像“${this.gestureLabels[predictedGesture] ?? predictedGesture}”，目标是“${this.gestureLabels[targetGesture] ?? targetGesture}”。`,
      );
    }
    if (deviations.length === 0 || deviations[0].magnitude < 0.6) {
      lines.push("姿势已经接近标准，可以继续保持。");
      return { deviations, feedback: lines };
    }
    deviations.forEach((item) => {
      lines.push(`${item.label}${directionText(item.feature, item.z_score)}（${Math.abs(item.z_score).toFixed(1)}σ）。`);
    });
    return { deviations, feedback: lines };
  }

  score(vector, targetGesture, options = {}) {
    const {
      updateUserModel = false,
      topKFeedback = 3,
      tauOverride = null,
    } = options;

    const userModel = this.getUserModel(targetGesture);
    const effectiveDescriptor = userModel.cachedDescriptor;
    const tau = Math.max(tauOverride ?? userModel.tau, 1e-6);
    const diff = subtract(vector, effectiveDescriptor.mu);
    const mahalanobisDistance = Math.sqrt(Math.max(quadForm(diff, effectiveDescriptor.sigma_inv), 0));
    const rawScore = Math.exp(-mahalanobisDistance / tau);
    const displayScore = Math.max(0, Math.min(100, Math.round((rawScore ** this.stage2Config.gamma) * 100)));

    const { predictedGesture, populationDistances } = this.classify(vector);
    const { deviations, feedback } = this.generateFeedback(
      vector,
      effectiveDescriptor,
      targetGesture,
      predictedGesture,
      topKFeedback,
    );

    const result = {
      target_gesture: targetGesture,
      predicted_gesture: predictedGesture,
      mahalanobis_distance: mahalanobisDistance,
      raw_score: rawScore,
      display_score: displayScore,
      matched: predictedGesture === targetGesture,
      feedback,
      tau,
      blend_weight: userModel.blendWeight,
      feature_deviations: deviations,
      population_distances: populationDistances,
      vector,
    };

    if (updateUserModel) {
      userModel.recordAttempt(vector);
    }
    return result;
  }
}

