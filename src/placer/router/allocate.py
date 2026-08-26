from __future__ import annotations

BISECTION_STEPS = 80

TIER_POLICY = {
    "fast": {"k1_cap": 0.0, "fill_budget": 0.85, "ax31_quantile": 0.30},
    "balanced": {"k1_cap": 0.0, "fill_budget": 0.80, "ax31_quantile": 0.20},
    "premium": {"k1_cap": 0.05, "fill_budget": 0.55, "ax31_quantile": 0.20},
}
K1_STAGE_BUDGET = 0.70


def _quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _bisect(step, cap):
    selection, total = step(0.0)
    if total <= cap:
        return selection
    low, high = 0.0, 1.0
    selection, total = step(high)
    while total > cap and high < 2.0 ** 60:
        low = high
        high *= 2.0
        selection, total = step(high)
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2.0
        candidate, candidate_total = step(middle)
        if candidate_total <= cap:
            high = middle
            selection, total = candidate, candidate_total
        else:
            low = middle
    return selection if total <= cap else None


def allocate(scores, costs, tier, budget_multiplier):
    count = len(scores)
    if count == 0:
        return []
    policy = TIER_POLICY[tier]
    light_total = sum(row[0] for row in costs)
    if light_total <= 0.0:
        return [0] * count

    limit = int(count * policy["k1_cap"])
    base = [0] * count
    if limit > 0:
        def stage_one(penalty):
            values = [
                (
                    s[0] - penalty * c[0] / light_total,
                    s[2] - penalty * c[2] / light_total,
                )
                for s, c in zip(scores, costs)
            ]
            selection = [2 if v[1] > v[0] else 0 for v in values]
            chosen = [i for i, v in enumerate(selection) if v == 2]
            if len(chosen) > limit:
                margin = [values[i][1] - values[i][0] for i in chosen]
                order = sorted(range(len(chosen)), key=lambda j: (margin[j], chosen[j]))
                for j in order[: len(chosen) - limit]:
                    selection[chosen[j]] = 0
            return selection, sum(costs[i][selection[i]] for i in range(count))

        found = _bisect(
            stage_one, light_total * max(1.0, budget_multiplier * K1_STAGE_BUDGET)
        )
        if found is not None:
            base = found

    current = sum(costs[i][base[i]] for i in range(count))
    fill_cap = max(current, light_total * budget_multiplier * policy["fill_budget"])
    free = [i for i in range(count) if base[i] == 0]
    gains = [scores[i][1] - scores[i][0] for i in range(count)]
    cut = _quantile([gains[i] for i in free], policy["ax31_quantile"]) if free else 0.0

    def stage_two(penalty):
        selection = list(base)
        for i in free:
            if gains[i] < cut:
                continue
            delta = gains[i] - penalty * (costs[i][1] - costs[i][0]) / light_total
            if delta > 0:
                selection[i] = 1
        return selection, sum(costs[i][selection[i]] for i in range(count))

    filled = _bisect(stage_two, fill_cap)
    return filled if filled is not None else base
