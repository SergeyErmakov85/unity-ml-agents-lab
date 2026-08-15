"""Боковые каналы связи с Unity (требования 7.3, 7.4, 11.4).

Три канала используются во всех средах лаборатории:

``EngineConfigurationChannel``
    Скорость симуляции. При обучении ``time_scale=20`` даёт на порядок больше
    шагов в секунду; при отладке в редакторе — ``1``, иначе за агентом
    невозможно уследить.

``EnvironmentParametersChannel``
    Параметры среды: ``seed`` и ``difficulty`` как минимум. Именно так сид
    попадает из Python в Unity (требование 7.3) — общего генератора у процессов нет.

``StatsSideChannel``
    Метрики со стороны среды: доля успехов, число столкновений, время до цели.
    Отсюда они уходят в TensorBoard в неймспейс ``Env/`` (требование 11.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod, StatsSideChannel

#: Имена параметров, которые обязана понимать любая среда лаборатории (7.3).
SEED_PARAMETER = "seed"
DIFFICULTY_PARAMETER = "difficulty"


@dataclass
class LabSideChannels:
    """Набор каналов, общий для всех сред лаборатории.

    Экземпляр создаётся до запуска ``UnityEnvironment`` и передаётся ему
    списком ``side_channels``. После запуска через него задаются скорость
    симуляции и параметры среды, а также читаются метрики.
    """

    engine: EngineConfigurationChannel = field(default_factory=EngineConfigurationChannel)
    parameters: EnvironmentParametersChannel = field(default_factory=EnvironmentParametersChannel)
    stats: StatsSideChannel = field(default_factory=StatsSideChannel)

    def as_list(self) -> list:
        """Порядок значения не имеет: каналы различаются по UUID, а не по позиции."""
        return [self.engine, self.parameters, self.stats]

    def configure_engine(self, time_scale: float = 20.0, target_frame_rate: int = -1) -> None:
        """Задаёт скорость симуляции.

        ``time_scale`` выше ~20 ломает физику PhysX: шаг интегрирования
        становится слишком крупным, и поведение среды перестаёт совпадать
        с тем, что видно в редакторе. Для сред без физики (E01_GridWorld)
        ограничение не действует.
        """
        self.engine.set_configuration_parameters(
            time_scale=time_scale,
            target_frame_rate=target_frame_rate,
        )

    def set_seed(self, seed: int) -> None:
        """Передаёт сид в Unity. Действует со **следующего** сброса среды."""
        self.parameters.set_float_parameter(SEED_PARAMETER, float(seed))

    def set_difficulty(self, difficulty: float) -> None:
        self.parameters.set_float_parameter(DIFFICULTY_PARAMETER, float(difficulty))

    def set_parameters(self, values: Mapping[str, float]) -> None:
        """Произвольные параметры среды из блока ``env.env_parameters`` конфига."""
        for key, value in values.items():
            self.parameters.set_float_parameter(key, float(value))

    def drain_stats(self) -> dict[str, float]:
        """Забирает накопленные метрики среды и очищает буфер канала.

        Каждая метрика приходит списком значений с указанием способа
        агрегации — тем же, что задан в Unity через ``MetricsRecorder``.
        Здесь список сворачивается в одно число:

        * ``AVERAGE`` и ``HISTOGRAM`` — среднее (гистограмма в скалярный лог
          не помещается; распределение при необходимости логируется отдельно);
        * ``SUM`` — сумма;
        * ``MOST_RECENT`` — последнее значение.

        Возвращает имена **без** префикса: неймспейс ``Env/`` добавляет
        логгер, чтобы среда не знала о схеме TensorBoard.
        """
        raw = self.stats.get_and_reset_stats()
        result: dict[str, float] = {}
        for key, entries in raw.items():
            if not entries:
                continue
            values = [value for value, _ in entries]
            method = entries[-1][1]
            if method is StatsAggregationMethod.SUM:
                result[key] = float(sum(values))
            elif method is StatsAggregationMethod.MOST_RECENT:
                result[key] = float(values[-1])
            else:  # AVERAGE, HISTOGRAM
                result[key] = float(sum(values) / len(values))
        return result
