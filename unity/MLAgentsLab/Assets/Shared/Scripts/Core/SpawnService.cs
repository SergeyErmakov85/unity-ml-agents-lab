using System.Collections.Generic;
using UnityEngine;

namespace LabRL.Core
{
    /// <summary>
    /// Детерминированный подбор точек спавна внутри арены.
    ///
    /// Все методы принимают генератор арены (<c>System.Random</c>) явным
    /// аргументом и не обращаются к <c>UnityEngine.Random</c>: глобальный
    /// генератор общий на процесс, и при нескольких аренах порядок обращений
    /// к нему зависит от порядка вызова <c>OnEpisodeBegin</c>, то есть
    /// воспроизводимость при фиксированном сиде теряется (требование 7.3).
    ///
    /// Аллокаций в горячем пути нет: перегрузки, которым нужен список занятых
    /// точек, принимают его снаружи и не создают коллекций (требование 7.5).
    /// </summary>
    public static class SpawnService
    {
        /// <summary>Случайная точка в прямоугольнике XZ вокруг центра.</summary>
        /// <param name="rng">Генератор арены.</param>
        /// <param name="center">Центр области спавна в мировых координатах.</param>
        /// <param name="halfExtents">Половины сторон прямоугольника по X и Z.</param>
        /// <param name="y">Высота, на которой ставится объект.</param>
        public static Vector3 RandomPointInRect(System.Random rng, Vector3 center, Vector2 halfExtents, float y)
        {
            float x = center.x + (float)(rng.NextDouble() * 2.0 - 1.0) * halfExtents.x;
            float z = center.z + (float)(rng.NextDouble() * 2.0 - 1.0) * halfExtents.y;
            return new Vector3(x, y, z);
        }

        /// <summary>
        /// Точка в прямоугольнике, отстоящая не менее чем на <paramref name="minDistance"/>
        /// от каждой из уже занятых точек.
        /// </summary>
        /// <param name="occupied">Уже занятые точки. Не изменяется.</param>
        /// <param name="maxAttempts">
        /// Сколько раз пробовать, прежде чем вернуть последнюю попытку. Отказ
        /// от бесконечного цикла сознателен: при слишком плотной упаковке
        /// среда должна деградировать предсказуемо, а не зависать.
        /// </param>
        /// <returns>
        /// Найденная точка. Если за <paramref name="maxAttempts"/> попыток условие
        /// не выполнено, возвращается последняя сгенерированная точка,
        /// а <paramref name="satisfied"/> становится false.
        /// </returns>
        public static Vector3 RandomPointAwayFrom(
            System.Random rng,
            Vector3 center,
            Vector2 halfExtents,
            float y,
            IReadOnlyList<Vector3> occupied,
            float minDistance,
            out bool satisfied,
            int maxAttempts = 32)
        {
            float sqrMin = minDistance * minDistance;
            Vector3 candidate = center;

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                candidate = RandomPointInRect(rng, center, halfExtents, y);

                bool ok = true;
                for (int i = 0; i < occupied.Count; i++)
                {
                    if ((occupied[i] - candidate).sqrMagnitude < sqrMin)
                    {
                        ok = false;
                        break;
                    }
                }

                if (ok)
                {
                    satisfied = true;
                    return candidate;
                }
            }

            satisfied = false;
            return candidate;
        }

        /// <summary>
        /// Тасование Фишера — Йетса на месте. Нужно для перебора клеток сетки
        /// в случайном, но воспроизводимом порядке (например, выбор свободной клетки).
        /// </summary>
        public static void Shuffle<T>(System.Random rng, IList<T> items)
        {
            for (int i = items.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (items[i], items[j]) = (items[j], items[i]);
            }
        }

        /// <summary>Случайный элемент списка. Список должен быть непустым.</summary>
        public static T Pick<T>(System.Random rng, IReadOnlyList<T> items)
        {
            if (items.Count == 0)
                throw new System.ArgumentException("список пуст", nameof(items));
            return items[rng.Next(items.Count)];
        }
    }
}
