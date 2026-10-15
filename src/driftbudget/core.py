import heapq

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, recall_score


METHODS = [
    "driftbudget",
    "global_drift",
    "static_cost",
    "cheap_only",
    "full_features",
    "no_delayed_credit",
]


def conditional(mean, covariance, observed, values, target):
    missing = np.asarray(target, dtype=int)
    if not len(observed):
        return mean[missing], covariance[np.ix_(missing, missing)]
    observed = np.asarray(observed, dtype=int)
    cross = covariance[np.ix_(missing, observed)]
    inverse = np.linalg.inv(covariance[np.ix_(observed, observed)] + np.eye(len(observed)) * 1e-6)
    center = mean[missing] + cross @ inverse @ (values - mean[observed])
    uncertainty = covariance[np.ix_(missing, missing)] - cross @ inverse @ cross.T
    return center, (uncertainty + uncertainty.T) / 2


def encoded(mean, covariance, d, observed, values):
    complete = mean.copy()
    complete[observed] = values
    missing = np.array([i for i in range(d) if i not in observed])
    if len(missing):
        complete[missing] = conditional(mean, covariance, observed, values, missing)[0]
    mask = np.zeros(d)
    mask[observed] = 1
    return np.r_[complete, mask, 1.0]


def generate(rng, count):
    for index in range(count):
        d, labels, n = 8, 3, 240
        weight = np.array(
            [
                [rng.gauss(0, 1) if j % labels == label else 0 for label in range(labels)]
                for j in range(d)
            ]
        )
        warm = np.array([[rng.gauss(0, 1) for _ in range(d)] for _ in range(100)])
        warm_probability = expit(warm @ weight)
        warm_y = np.array([[rng.random() < p for p in row] for row in warm_probability], int)
        x = np.array([[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)])
        ys = []
        for t, row in enumerate(x):
            changed = weight.copy()
            if t >= n // 2:
                changed[:, 0] *= -1
            ys.append([int(rng.random() < p) for p in expit(row @ changed)])
        yield {
            "id": str(index),
            "family": f"stream-{index}",
            "warm_x": warm.tolist(),
            "warm_y": warm_y.tolist(),
            "x": x.tolist(),
            "y": ys,
            "delay": [rng.randrange(11) for _ in range(n)],
            "costs": [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "mandatory": [0, 1],
            "budget": 1.5,
            "seed": rng.randrange(2**31),
            "threshold": 0.001,
            "beta": 0.2,
            "learning_rate": 0.02,
            "drift_time": n // 2,
            "drift_labels": [0],
        }


def run_case(case, method):
    rng = np.random.default_rng(case["seed"])
    warm, warm_y = np.asarray(case["warm_x"]), np.asarray(case["warm_y"])
    stream, labels = np.asarray(case["x"]), np.asarray(case["y"])
    mean, covariance = warm.mean(0), np.cov(warm.T) + np.eye(warm.shape[1]) * 0.01
    d, classes = warm.shape[1], warm_y.shape[1]
    train_features = []
    for row in warm:
        observed = sorted(set(case["mandatory"]) | set(np.flatnonzero(rng.random(d) < 0.5)))
        train_features.append(encoded(mean, covariance, d, observed, row[observed]))
    X = np.asarray(train_features)

    def objective(flat):
        W = flat.reshape(X.shape[1], classes)
        logits = X @ W
        value = np.mean(np.logaddexp(0, logits) - warm_y * logits) + 0.001 * np.sum(W**2)
        gradient = X.T @ (expit(logits) - warm_y) / warm_y.size + 0.002 * W
        return value, gradient.ravel()

    fitted = minimize(objective, np.zeros(X.shape[1] * classes), jac=True, method="L-BFGS-B")
    if not fitted.success:
        raise RuntimeError("Initial feature predictor did not converge")
    W = fitted.x.reshape(X.shape[1], classes)
    prevalence = np.clip(warm_y.mean(0), 0.05, 0.95)
    positive, negative = 0.5 / prevalence, 0.5 / (1 - prevalence)
    risk = lambda p: np.minimum(positive * p, negative * (1 - p))
    drift = np.full(classes, 0.05)
    credit = np.zeros((d, classes))
    queue = []
    predictions, purchases, posteriors, value_pairs = [], [], [], []

    def release(time):
        nonlocal W, drift, credit
        while queue and queue[0][0] <= time:
            _, event, record = heapq.heappop(queue)
            target = labels[event]
            p = record["probability"]
            error = (p >= 0.5) != target
            stable_likelihood = np.where(error, 0.2, 0.8)
            changed_likelihood = 0.5
            drift = (
                drift
                * changed_likelihood
                / (drift * changed_likelihood + (1 - drift) * stable_likelihood)
            )
            if method != "no_delayed_credit":
                for feature, before, after, predicted in record["steps"]:
                    before_loss = -(
                        target * np.log(np.maximum(before, 1e-8))
                        + (1 - target) * np.log(np.maximum(1 - before, 1e-8))
                    )
                    after_loss = -(
                        target * np.log(np.maximum(after, 1e-8))
                        + (1 - target) * np.log(np.maximum(1 - after, 1e-8))
                    )
                    realized = before_loss - after_loss
                    credit[feature] = 0.95 * credit[feature] + 0.05 * (realized - predicted)
                    value_pairs.append([float(predicted.mean()), float(realized.mean())])
            W -= case["learning_rate"] * np.outer(record["features"], p - target)

    for t, raw in enumerate(stream):
        release(t - 1e-9)
        drift = 0.99 * drift + 0.01 * (1 - drift)
        posteriors.append(drift.tolist())
        observed = sorted(case["mandatory"])
        values = raw[observed]
        feature = encoded(mean, covariance, d, observed, values)
        probability = expit(feature @ W)
        spent, selected, steps = 0.0, [], []
        while method != "cheap_only":
            options = []
            for j in range(d):
                if (
                    j in observed
                    or spent + case["costs"][j] > case["budget"]
                    and method != "full_features"
                ):
                    continue
                center, uncertainty = conditional(mean, covariance, observed, values, [j])
                draws = center[0] + np.sqrt(max(uncertainty[0, 0], 0)) * rng.normal(size=16)
                prospective = []
                for value in draws:
                    additional = observed + [j]
                    future = encoded(mean, covariance, d, additional, np.r_[values, value])
                    prospective.append(risk(expit(future @ W)))
                delta = risk(probability) - np.mean(prospective, axis=0)
                adjusted = delta + credit[j]
                local = np.full(classes, drift.mean()) if method == "global_drift" else drift
                multiplier = (
                    np.ones(classes)
                    if method in ("static_cost", "full_features")
                    else (case["beta"] + (1 - case["beta"]) * local)
                )
                score = float((multiplier * adjusted).sum() / case["costs"][j])
                options.append((score, -j, j, delta))
            if not options:
                break
            _, _, j, predicted = max(options, key=lambda item: item[:2])
            if max(item[0] for item in options) <= case["threshold"] and method != "full_features":
                break
            before = probability.copy()
            observed.append(j)
            values = np.r_[values, raw[j]]
            feature = encoded(mean, covariance, d, observed, values)
            probability = expit(feature @ W)
            selected.append(j)
            spent += case["costs"][j]
            steps.append((j, before, probability.copy(), predicted))
        predictions.append(probability.tolist())
        purchases.append({"features": selected, "cost": spent})
        heapq.heappush(
            queue,
            (
                t + case["delay"][t],
                t,
                {"features": feature, "probability": probability, "steps": steps},
            ),
        )
        release(t)
    release(float("inf"))
    return {
        "probability": predictions,
        "purchases": purchases,
        "drift_posterior": posteriors,
        "value_pairs": value_pairs,
        "coefficients": W.tolist(),
    }


def score(case, result):
    p, y = np.asarray(result["probability"]), np.asarray(case["y"])
    prediction = p >= 0.5
    prevalence = np.clip(np.mean(case["warm_y"], axis=0), 0.05, 0.95)
    loss = (
        -(
            y * np.log(np.maximum(p, 1e-8)) / prevalence
            + (1 - y) * np.log(np.maximum(1 - p, 1e-8)) / (1 - prevalence)
        )
        / 2
    )
    tail = np.flatnonzero(prevalence <= np.median(prevalence))
    pairs = np.asarray(result["value_pairs"])
    posterior = np.asarray(result["drift_posterior"])
    delays = []
    for label in case.get("drift_labels", []):
        indices = np.flatnonzero(posterior[case["drift_time"] :, label] >= 0.5)
        if len(indices):
            delays.append(int(indices[0]))
    return {
        "class_balanced_risk": float(loss.mean()),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "tail_recall": float(
            recall_score(y[:, tail], prediction[:, tail], average="macro", zero_division=0)
        ),
        "mean_cost": float(np.mean([r["cost"] for r in result["purchases"]])),
        "budget_violation": float(
            np.mean([r["cost"] > case["budget"] + 1e-9 for r in result["purchases"]])
        ),
        "posterior_localization_delay": float(np.mean(delays)) if delays else None,
        "selected_value_spearman": float(spearmanr(pairs[:, 0], pairs[:, 1]).statistic)
        if len(pairs) > 2 and np.std(pairs[:, 0]) and np.std(pairs[:, 1])
        else None,
        "legal_effects": len(y),
    }
