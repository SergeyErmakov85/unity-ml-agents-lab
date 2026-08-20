"""Что считать успешным эпизодом.

Долгое время в лаборатории хватало одного правила: «эпизод успешен, если его
суммарная награда выше порога». Оно работает, пока награда монотонно связана
с качеством поведения. В `E06_Hunter3D` эта связь рвётся.

Причина — в формировании награды. Потенциальная добавка
``F(s, s′) = γ·Φ(s′) − Φ(s)`` не меняет **дисконтированную** цель, которую
метод и оптимизирует. Но `Environment/Cumulative Reward` и оценка политики
считают сумму **недисконтированную**, а у неё есть смещение::

    Σ_t F_t = Φ(s_T) − Φ(s_0) + (1 − γ)·Σ_t (−Φ(s_{t+1}))

Второе слагаемое положительно всюду, где потенциал отрицателен, и растёт
с числом шагов. Численно, при Φ = −d/d_max, γ = 0.99 и 1000 шагов: агент,
простоявший весь эпизод в 25 метрах от цели, набирает **+7.84**, а агент,
поймавший цель за 200 шагов, — около **+1.5**. По сырой награде второй хуже
первого, хотя задачу решил только он.

Отсюда правило: **в средах с формированием награды успех определяется исходом
эпизода, а не его наградой.** Для этого достаточно того, что среда уже
сообщает: завершился эпизод сам (`terminated`) или был оборван по времени
(`truncated`).
"""

from __future__ import annotations

import numpy as np

#: Успех — суммарная награда эпизода выше порога.
#: Годится там, где награда прямо выражает качество: `E00`, `E01`, `E03`.
REWARD_ABOVE = "reward_above"

#: Успех — эпизод завершился сам: цель достигнута, ловушка не в счёт.
#: Для сред «дойди/поймай» с формированием награды: `E06`.
TERMINATED = "terminated"

#: Успех — эпизод дожил до `MaxStep`. Для сред «продержись как можно дольше»:
#: `E02_CartPoleUnity`, `E04_BallBalance`.
TRUNCATED = "truncated"

RULES = (REWARD_ABOVE, TERMINATED, TRUNCATED)


def check_rule(rule: str) -> str:
    """Проверяет имя правила и возвращает его же — для раннего падения на опечатке."""
    if rule not in RULES:
        raise ValueError(f"правило успеха должно быть одним из {RULES}, получено {rule!r}")
    return rule


def episode_succeeded(rule: str, episode_return: float, threshold: float, terminated: bool) -> bool:
    """Успешен ли завершившийся эпизод.

    Args:
        rule: одно из :data:`RULES`.
        episode_return: суммарная (недисконтированная) награда эпизода.
        threshold: порог для правила ``reward_above``.
        terminated: эпизод завершился сам (иначе — оборван по ``MaxStep``).
    """
    if rule == REWARD_ABOVE:
        return bool(episode_return > threshold)
    if rule == TERMINATED:
        return bool(terminated)
    return not bool(terminated)


def success_rate(rule: str, returns: np.ndarray, terminated: np.ndarray, threshold: float) -> float:
    """Доля успешных эпизодов в выборке."""
    returns = np.asarray(returns, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    if returns.size == 0:
        return 0.0
    if rule == REWARD_ABOVE:
        return float(np.mean(returns > threshold))
    if rule == TERMINATED:
        return float(np.mean(terminated))
    return float(np.mean(~terminated))
