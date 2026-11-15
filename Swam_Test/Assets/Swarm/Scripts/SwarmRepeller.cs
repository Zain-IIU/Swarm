using System.Collections.Generic;
using UnityEngine;

namespace SwarmTest
{
    public class SwarmRepeller : MonoBehaviour
    {
        public static readonly List<SwarmRepeller> Active = new List<SwarmRepeller>();

        [Min(0f)] public float radius = 1.5f;
        [Tooltip("Extra distance outside the radius where creatures start turning away")]
        [Min(0f)] public float fearMargin = 0.6f;

        [Header("Light")]
        [Tooltip("Optional. Its range follows the radius, and switching it off disables the repeller")]
        public Light linkedLight;
        public float lightRangeScale = 1.8f;
        [Range(0f, 0.5f)] public float flickerAmount = 0.15f;
        public float flickerSpeed = 9f;

        float baseIntensity, seed;

        public bool IsLit
        {
            get { return isActiveAndEnabled && radius > 0f && (linkedLight == null || (linkedLight.enabled && linkedLight.gameObject.activeInHierarchy)); }
        }

        void OnEnable()
        {
            Active.Add(this);
            if (linkedLight != null) baseIntensity = linkedLight.intensity;
            seed = Random.value * 100f;
        }

        void OnDisable() { Active.Remove(this); }

        void Update()
        {
            if (linkedLight == null) return;
            linkedLight.range = radius * lightRangeScale;
            float n = Mathf.PerlinNoise(seed, Time.time * flickerSpeed) * 2f - 1f;
            linkedLight.intensity = baseIntensity * (1f + flickerAmount * n);
        }

        void OnDrawGizmos()
        {
            Vector3 c = transform.position;
            c.y = 0.02f;
            Gizmos.color = new Color(1f, 0.6f, 0.15f, 0.9f);
            DrawCircle(c, radius);
            Gizmos.color = new Color(1f, 0.6f, 0.15f, 0.3f);
            DrawCircle(c, radius + fearMargin);
        }

        static void DrawCircle(Vector3 c, float r)
        {
            const int n = 48;
            Vector3 prev = c + new Vector3(r, 0f, 0f);
            for (int i = 1; i <= n; i++)
            {
                float a = i * Mathf.PI * 2f / n;
                Vector3 p = c + new Vector3(Mathf.Cos(a) * r, 0f, Mathf.Sin(a) * r);
                Gizmos.DrawLine(prev, p);
                prev = p;
            }
        }
    }
}
