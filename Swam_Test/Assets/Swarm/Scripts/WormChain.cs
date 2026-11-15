using UnityEngine;

namespace SwarmTest
{
    [DisallowMultipleComponent]
    public class WormChain : MonoBehaviour
    {
        [Header("Rig")]
        public string bonePrefix = "Spine_";
        public int boneCount = 10;

        [Header("Steering")]
        public Transform target;
        public float crawlSpeed = 0.5f;
        [Tooltip("Degrees per second")]
        public float turnRate = 120f;
        public float arriveRadius = 0.25f;
        [Tooltip("Degrees")]
        public float wiggleAmplitude = 35f;
        [Tooltip("Cycles per second")]
        public float wiggleFrequency = 1.4f;

        [Header("Physics")]
        public Vector3 gravity = new Vector3(0f, -9.81f, 0f);
        [Range(1, 8)] public int substeps = 4;
        [Range(1, 16)] public int constraintIterations = 6;
        [Range(0f, 1f)] public float airDamping = 0.01f;
        [Range(0f, 1f)] public float lateralFriction = 0.6f;
        [Range(0f, 1f)] public float forwardFriction = 0.08f;
        [Range(0f, 1f)] public float bendStiffness = 0.08f;
        [Tooltip("Particle i and i+2 can't get closer than this fraction of two segments")]
        [Range(0.1f, 1f)] public float minFoldRatio = 0.55f;
        public float radiusScale = 1f;
        public LayerMask collisionMask = ~0;

        public bool drawParticles = true;

        Transform[] bones;
        Vector3[] pos, prev;
        float[] radius;
        bool[] grounded;
        float segLen;
        Quaternion[] restOffset;
        Vector3 heading;
        float phase;
        SphereCollider probe;
        readonly Collider[] hits = new Collider[16];

        public int ParticleCount { get { return pos == null ? 0 : pos.Length; } }
        public Vector3 HeadPosition { get { return pos[0]; } }

        void Start()
        {
            bones = new Transform[boneCount];
            for (int i = 0; i < boneCount; i++)
            {
                bones[i] = FindDeep(transform, bonePrefix + i.ToString("00"));
                if (bones[i] == null)
                {
                    Debug.LogError("[WormChain] Missing bone " + bonePrefix + i.ToString("00"), this);
                    enabled = false;
                    return;
                }
            }

            int n = boneCount + 1;
            pos = new Vector3[n];
            prev = new Vector3[n];
            radius = new float[n];
            grounded = new bool[n];
            restOffset = new Quaternion[boneCount];

            for (int i = 0; i < boneCount; i++) pos[i] = bones[i].position;
            segLen = Vector3.Distance(pos[0], pos[1]);
            Vector3 lastDir = (pos[boneCount - 1] - pos[boneCount - 2]).normalized;
            pos[n - 1] = pos[n - 2] + lastDir * segLen;

            // bone rotation relative to an (along-body, world-up) frame, so it can be rebuilt from the chain
            for (int i = 0; i < boneCount; i++)
            {
                Vector3 dir = (pos[i + 1] - pos[i]).normalized;
                Quaternion frame = Quaternion.LookRotation(dir, Vector3.up);
                restOffset[i] = Quaternion.Inverse(frame) * bones[i].rotation;
            }

            for (int i = 0; i < n; i++)
            {
                radius[i] = BodyRadius(i * segLen) * radiusScale;
                prev[i] = pos[i];
            }

            heading = pos[0] - pos[1];
            heading.y = 0f;
            heading = heading.sqrMagnitude > 1e-6f ? heading.normalized : Vector3.forward;

            // reusable trigger sphere for Physics.ComputePenetration queries
            var probeGo = new GameObject("WormCollisionProbe");
            probeGo.hideFlags = HideFlags.HideAndDontSave;
            probeGo.layer = 2;
            probe = probeGo.AddComponent<SphereCollider>();
            probe.isTrigger = true;
            collisionMask &= ~(1 << 2);

            foreach (var smr in GetComponentsInChildren<SkinnedMeshRenderer>())
                smr.updateWhenOffscreen = true;
        }

        void OnDestroy()
        {
            if (probe != null) Destroy(probe.gameObject);
        }

        // matches the taper of the Blender mesh
        static float BodyRadius(float s)
        {
            const float maxR = 0.05f, bodyR = 0.047f, tipR = 0.006f;
            const float taperStart = 0.55f, length = 1.2f;
            if (s < 0.06f) return Mathf.Lerp(0.035f, maxR, s / 0.06f);
            if (s < taperStart) return bodyR;
            float k = Mathf.Clamp01((s - taperStart) / (length - taperStart));
            return tipR + (bodyR - tipR) * (1f - k) * (1f + 0.6f * k);
        }

        void FixedUpdate()
        {
            if (pos == null) return;
            float dt = Time.fixedDeltaTime;
            Steer(dt);

            float h = dt / substeps;
            for (int s = 0; s < substeps; s++)
            {
                Integrate(h);
                DriveHead(h);
                for (int it = 0; it < constraintIterations; it++)
                {
                    SolveLengths();
                    SolveBending();
                }
                SolveCollisions();
                SolveLengths();
            }
        }

        Vector3 moveDir;
        bool arrived;

        void Steer(float dt)
        {
            Vector3 toTarget = target != null ? target.position - pos[0] : heading;
            toTarget.y = 0f;
            arrived = target != null && toTarget.magnitude < arriveRadius;

            if (!arrived && toTarget.sqrMagnitude > 1e-6f)
                heading = Vector3.RotateTowards(heading, toTarget.normalized, turnRate * Mathf.Deg2Rad * dt, 0f);

            phase += dt * wiggleFrequency * 2f * Mathf.PI * (arrived ? 0.25f : 1f);
            float swing = Mathf.Sin(phase) * wiggleAmplitude * (arrived ? 0.3f : 1f);
            moveDir = Quaternion.AngleAxis(swing, Vector3.up) * heading;
        }

        void DriveHead(float h)
        {
            if (arrived) return;
            if (!(grounded[0] || grounded[1])) return;
            pos[0] += moveDir * crawlSpeed * h;
        }

        void Integrate(float h)
        {
            Vector3 g = gravity * h * h;
            for (int i = 0; i < pos.Length; i++)
            {
                Vector3 v = (pos[i] - prev[i]) * (1f - airDamping);

                if (grounded[i])
                {
                    // anisotropic friction: slides along the body, grips sideways -> slithering
                    Vector3 t = BodyTangent(i);
                    Vector3 vAlong = Vector3.Dot(v, t) * t;
                    Vector3 vSide = v - vAlong;
                    v = vAlong * (1f - forwardFriction) + vSide * (1f - lateralFriction);
                }

                prev[i] = pos[i];
                pos[i] += v + g;
            }
        }

        Vector3 BodyTangent(int i)
        {
            int a = Mathf.Max(0, i - 1), b = Mathf.Min(pos.Length - 1, i + 1);
            Vector3 t = pos[a] - pos[b];
            return t.sqrMagnitude > 1e-8f ? t.normalized : heading;
        }

        void SolveLengths()
        {
            for (int i = 0; i < pos.Length - 1; i++)
            {
                Vector3 d = pos[i + 1] - pos[i];
                float len = d.magnitude;
                if (len < 1e-6f) continue;
                Vector3 corr = d * (0.5f * (len - segLen) / len);
                pos[i] += corr;
                pos[i + 1] -= corr;
            }
        }

        void SolveBending()
        {
            float rest = 2f * segLen;
            float minLen = rest * minFoldRatio;
            for (int i = 0; i < pos.Length - 2; i++)
            {
                Vector3 d = pos[i + 2] - pos[i];
                float len = d.magnitude;
                if (len < 1e-6f) continue;
                float goal = len < minLen ? minLen : Mathf.Lerp(len, rest, bendStiffness);
                Vector3 corr = d * (0.5f * (len - goal) / len);
                pos[i] += corr;
                pos[i + 2] -= corr;
            }
        }

        void SolveCollisions()
        {
            for (int i = 0; i < pos.Length; i++)
            {
                grounded[i] = false;
                float r = radius[i];
                probe.radius = r;
                int count = Physics.OverlapSphereNonAlloc(pos[i], r + 0.002f, hits, collisionMask, QueryTriggerInteraction.Ignore);
                for (int c = 0; c < count; c++)
                {
                    Collider col = hits[c];
                    if (col == probe) continue;
                    Vector3 dir; float dist;
                    if (Physics.ComputePenetration(probe, pos[i], Quaternion.identity,
                                                   col, col.transform.position, col.transform.rotation,
                                                   out dir, out dist))
                    {
                        pos[i] += dir * dist;
                    }
                    grounded[i] = true;
                }
            }
        }

        void LateUpdate()
        {
            if (pos == null) return;
            Vector3 up = Vector3.up;
            for (int i = 0; i < bones.Length; i++)
            {
                Vector3 dir = pos[i + 1] - pos[i];
                if (dir.sqrMagnitude < 1e-10f) continue;
                dir.Normalize();
                // parallel-transport the up vector down the chain so the body never twists
                up = Vector3.ProjectOnPlane(up, dir);
                if (up.sqrMagnitude < 1e-6f) up = Vector3.ProjectOnPlane(heading, dir);
                up.Normalize();
                bones[i].SetPositionAndRotation(pos[i], Quaternion.LookRotation(dir, up) * restOffset[i]);
            }
        }

        static Transform FindDeep(Transform root, string name)
        {
            if (root.name == name) return root;
            for (int i = 0; i < root.childCount; i++)
            {
                Transform t = FindDeep(root.GetChild(i), name);
                if (t != null) return t;
            }
            return null;
        }

        void OnDrawGizmos()
        {
            if (!drawParticles || !Application.isPlaying || pos == null || radius == null || grounded == null) return;
            for (int i = 0; i < pos.Length; i++)
            {
                Gizmos.color = grounded[i] ? new Color(0.2f, 1f, 0.9f) : Color.yellow;
                Gizmos.DrawWireSphere(pos[i], radius[i]);
                if (i < pos.Length - 1) Gizmos.DrawLine(pos[i], pos[i + 1]);
            }
            Gizmos.color = Color.magenta;
            Gizmos.DrawRay(pos[0], moveDir * 0.3f);
        }
    }
}
