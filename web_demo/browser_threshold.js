export const ZONE_LABELS = {
  too_hard: "太难",
  in_zone: "适中",
  too_easy: "太简单",
};

export const ZONE_DESCRIPTIONS = {
  too_hard: "当前难度偏高，容易产生挫败感。",
  in_zone: "当前难度位于有效挑战区。",
  too_easy: "当前难度偏低，训练刺激不足。",
};

export function getChallengeZone(score, low = 60, high = 85) {
  if (score < low) {
    return "too_hard";
  }
  if (score > high) {
    return "too_easy";
  }
  return "in_zone";
}

export class BrowserThresholdManager {
  constructor(config) {
    this.config = config;
    this.gestureState = new Map();
  }

  _runtime(gesture) {
    if (!this.gestureState.has(gesture)) {
      this.gestureState.set(gesture, {
        gesture,
        tau: this.config.tau_init,
        baseline: null,
        momentum: 0,
        roundIndex: 0,
        pendingScores: [],
        history: [],
        challengeZone: "in_zone",
      });
    }
    return this.gestureState.get(gesture);
  }

  registerAttempt(gesture, displayScore, rawScore = null) {
    const runtime = this._runtime(gesture);
    runtime.pendingScores.push(Number(displayScore));
    runtime.pendingRawScores ??= [];
    runtime.pendingRawScores.push(rawScore == null ? null : Number(rawScore));
  }

  hasFullRound(gesture) {
    return this._runtime(gesture).pendingScores.length >= this.config.round_size;
  }

  finalizeRound(gesture) {
    const runtime = this._runtime(gesture);
    if (runtime.pendingScores.length === 0) {
      throw new Error(`No pending attempts for ${gesture}`);
    }
    const scores = runtime.pendingScores.splice(0, this.config.round_size);
    const meanScore = scores.reduce((sum, value) => sum + value, 0) / scores.length;
    const tauOld = runtime.tau;
    const zone = getChallengeZone(meanScore, this.config.challenge_low, this.config.challenge_high);

    let update;
    if (runtime.baseline == null) {
      runtime.baseline = meanScore;
      runtime.roundIndex += 1;
      runtime.challengeZone = zone;
      update = {
        tau_old: tauOld,
        tau_new: tauOld,
        momentum: runtime.momentum,
        baseline: runtime.baseline,
        direction: "hold",
        reason: "initialize_baseline",
        challenge_zone: zone,
        mean_score: meanScore,
        round_index: runtime.roundIndex,
      };
      runtime.history.push(update);
      return update;
    }

    const performanceSignal = meanScore - runtime.baseline;
    const momentum = this.config.beta * runtime.momentum + (1 - this.config.beta) * performanceSignal;
    let tauNew = tauOld;
    let direction = "hold";
    let reason = "momentum_below_activation";

    if (meanScore < runtime.baseline * 0.7) {
      tauNew = Math.min(tauOld + this.config.eta_relax * Math.abs(performanceSignal), this.config.tau_max);
      direction = "relax";
      reason = "fatigue_protection";
    } else if (momentum > this.config.theta_activate) {
      tauNew = Math.max(tauOld - this.config.eta * momentum, this.config.tau_min);
      direction = "tighten";
      reason = "sustained_improvement";
    }

    runtime.tau = Math.max(this.config.tau_min, Math.min(this.config.tau_max, tauNew));
    runtime.baseline = this.config.alpha_baseline * runtime.baseline + (1 - this.config.alpha_baseline) * meanScore;
    runtime.momentum = momentum;
    runtime.roundIndex += 1;
    runtime.challengeZone = zone;

    update = {
      tau_old: tauOld,
      tau_new: runtime.tau,
      momentum: runtime.momentum,
      baseline: runtime.baseline,
      direction,
      reason,
      challenge_zone: zone,
      mean_score: meanScore,
      round_index: runtime.roundIndex,
    };
    runtime.history.push(update);
    return update;
  }

  getState(gesture) {
    const runtime = this._runtime(gesture);
    return {
      gesture,
      tau: runtime.tau,
      baseline: runtime.baseline ?? 70,
      momentum: runtime.momentum,
      round_index: runtime.roundIndex,
      challenge_zone: runtime.challengeZone,
      pending_attempts: runtime.pendingScores.length,
    };
  }

  exportHistory() {
    const rows = [];
    for (const runtime of this.gestureState.values()) {
      runtime.history.forEach((item) => rows.push({ ...item, gesture: runtime.gesture }));
    }
    return rows;
  }
}
