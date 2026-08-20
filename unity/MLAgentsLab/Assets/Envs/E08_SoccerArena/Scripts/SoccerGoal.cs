using UnityEngine;

/// <summary>
/// Триггер ворот среды `E08_SoccerArena` (TS-008).
///
/// Единственная задача — сообщить арене, что мяч пересёк створ **этих**
/// ворот. Кому засчитать гол и какую награду начислить, решает
/// <see cref="SoccerArea"/>: правила матча не должны быть размазаны
/// по коллайдерам.
/// </summary>
[RequireComponent(typeof(BoxCollider))]
public class SoccerGoal : MonoBehaviour
{
    [Tooltip("Чьи это ворота: сторона, которой засчитывается пропущенный мяч.")]
    public SoccerArea.Side defendedBy = SoccerArea.Side.West;

    [Tooltip("Арена. Заполняется Setup-скриптом.")]
    public SoccerArea area;

    void Awake()
    {
        if (area == null) area = GetComponentInParent<SoccerArea>();
        // Ворота обязаны быть триггером: физический коллайдер отбивал бы мяч
        // от створа, и гол не случился бы никогда.
        GetComponent<BoxCollider>().isTrigger = true;
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("ball"))
            area.RegisterGoal(defendedBy);
    }
}
