"""Сопоставление идентификатора среды с полным именем поведения Unity.

Unity отдаёт имя вида ``E01_GridWorld?team=0``
(``BehaviorParameters.FullyQualifiedBehaviorName``), а идентификатор среды —
``E01_GridWorld``. Без учёта суффикса подключение падает с «в среде нет
поведения», хотя сцена совершенно корректна.
"""

from __future__ import annotations

import pytest

from labrl.envs.unity_env import resolve_behavior_name


def test_exact_name_matches():
    assert resolve_behavior_name("E01_GridWorld", ["E01_GridWorld"]) == "E01_GridWorld"


def test_team_suffix_matches():
    assert resolve_behavior_name("E01_GridWorld", ["E01_GridWorld?team=0"]) == "E01_GridWorld?team=0"


def test_other_behaviors_are_ignored():
    available = ["E03_RollerBall?team=0", "E01_GridWorld?team=0"]
    assert resolve_behavior_name("E01_GridWorld", available) == "E01_GridWorld?team=0"


def test_prefix_of_another_env_id_does_not_match():
    """`E01_Grid` не должен «поймать» `E01_GridWorld`."""
    with pytest.raises(KeyError, match="нет поведения"):
        resolve_behavior_name("E01_Grid", ["E01_GridWorld?team=0"])


def test_missing_behavior_reports_available():
    with pytest.raises(KeyError, match="E03_RollerBall"):
        resolve_behavior_name("E01_GridWorld", ["E03_RollerBall?team=0"])


def test_multiple_teams_require_explicit_choice():
    """Self-play даёт несколько команд одного поведения — выбирать за пользователя нельзя."""
    with pytest.raises(KeyError, match="несколько поведений"):
        resolve_behavior_name("E08_SoccerArena", ["E08_SoccerArena?team=0", "E08_SoccerArena?team=1"])
