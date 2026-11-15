using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;

namespace SwarmTest
{
    public class WormSwarm : MonoBehaviour
    {
        [Header("Swarm")]
        public GameObject wormPrefab;
        [Min(0)] public int maxWorms = 300;
        public float spawnPerSecond = 60f;
        public Transform spawnPoint;
        public float spawnRadius = 2f;
        public Vector2 spawnHeight = new Vector2(0.15f, 0.6f);
        public Transform target;

        [Header("Flow Field")]
        public Vector3 areaCenter = Vector3.zero;
        public Vector2 areaSize = new Vector2(14f, 14f);
        public float cellSize = 0.25f;
        public LayerMask obstacleMask = ~(1 << 2);
        [Tooltip("Taller than this is pathed around; lower is climbed over")]
        public float maxClimbHeight = 0.6f;
        public float obstacleInflate = 0.05f;

        [Header("Locomotion")]
        public float crawlSpeed = 0.55f;
        [Range(0f, 0.5f)] public float speedVariation = 0.25f;
        [Tooltip("Degrees per second")]
        public float turnRate = 160f;
        [Tooltip("Degrees")]
        public float wiggleAmplitude = 30f;
        public float wiggleFrequency = 1.4f;
        [Range(0f, 1f)] public float wanderStrength = 0.3f;
        [Tooltip("Inside this radius worms ignore the flow field and mill around the target")]
        public float arriveRadius = 0.7f;
        public float climbStrength = 1.0f;
        [Tooltip("Climbing over other worms: lower = flatter swarm, higher = taller mounds")]
        public float wormClimbStrength = 0.35f;
        [Tooltip("Touching body segments push themselves along the path toward the head")]
        [Range(0f, 1.5f)] public float bodyThrust = 0.6f;

        [Header("Physics")]
        public Vector3 gravity = new Vector3(0f, -9.81f, 0f);
        [Range(1, 6)] public int substeps = 2;
        [Range(1, 12)] public int constraintIterations = 4;
        [Range(0f, 1f)] public float airDamping = 0.01f;
        [Range(0f, 1f)] public float lateralFriction = 0.5f;
        [Range(0f, 1f)] public float forwardFriction = 0.08f;
        [Range(0f, 1f)] public float bendStiffness = 0.08f;
        [Range(0.1f, 1f)] public float minFoldRatio = 0.55f;
        [Range(0f, 1f)] public float wormCollisionStiffness = 0.8f;
        [Tooltip("Friction while resting only on other worms; low keeps piles flowing")]
        [Range(0f, 1f)] public float wormFriction = 0.12f;
        [Tooltip("Only BoxColliders on these layers are collided with")]
        public LayerMask environmentMask = ~(1 << 2);

        [Header("Rig")]
        public string bonePrefix = "Spine_";
        public int boneCount = 10;

        [Header("Debug")]
        public bool showHud = true;
        public bool showFlowField = true;
        public bool showParticles = false;

        // particle i of worm w lives at index w * K + i
        int K, B;
        NativeArray<float3> pos, prev, snap, delta, nLevel, nWorm;
        NativeArray<float> rad;
        NativeArray<byte> contact, onLevel;
        NativeArray<int> cellHash, sorted, checks;
        NativeArray<int> hashCount, hashStart;
        int tableSize;
        float segLen, hashCell;
        float[] restRadius;

        int count, capacity;
        NativeArray<float3> heading, moveDir;
        NativeArray<float> phase, speedMul, seed;

        NativeArray<quaternion> restOffset;
        NativeArray<float3> bonePos;
        NativeArray<quaternion> boneRot;
        TransformAccessArray boneTransforms;
        readonly List<Transform[]> wormBones = new List<Transform[]>();
        readonly List<GameObject> wormObjects = new List<GameObject>();

        NativeArray<LevelBox> boxes;
        NativeArray<float3> flowDir;
        public FlowField Field { get; private set; }
        public int Count { get { return count; } }

        float spawnAccumulator;

        readonly System.Diagnostics.Stopwatch simWatch = new System.Diagnostics.Stopwatch();
        float simMs, fpsSmoothed, hudTimer;
        string hudText = "";

        void Start()
        {
            if (wormPrefab == null) { Debug.LogError("[WormSwarm] No worm prefab", this); enabled = false; return; }
            B = boneCount;
            K = boneCount + 1;
            CaptureRestPose();
            Field = new FlowField(areaCenter, areaSize, cellSize);
            flowDir = new NativeArray<float3>(Field.dir.Length, Allocator.Persistent);
            RebakeObstacles();
            boneTransforms = new TransformAccessArray(Mathf.Max(16, maxWorms) * B);
            EnsureCapacity(Mathf.Max(16, maxWorms));
        }

        void OnDestroy()
        {
            Dispose(ref pos); Dispose(ref prev); Dispose(ref snap); Dispose(ref delta);
            Dispose(ref nLevel); Dispose(ref nWorm); Dispose(ref rad);
            Dispose(ref contact); Dispose(ref onLevel);
            Dispose(ref cellHash); Dispose(ref sorted); Dispose(ref checks);
            Dispose(ref hashCount); Dispose(ref hashStart);
            Dispose(ref heading); Dispose(ref moveDir); Dispose(ref phase); Dispose(ref speedMul); Dispose(ref seed);
            Dispose(ref restOffset); Dispose(ref bonePos); Dispose(ref boneRot);
            Dispose(ref boxes); Dispose(ref flowDir);
            if (boneTransforms.isCreated) boneTransforms.Dispose();
        }

        static void Dispose<T>(ref NativeArray<T> a) where T : struct { if (a.IsCreated) a.Dispose(); }

        static void Grow<T>(ref NativeArray<T> a, int n) where T : struct
        {
            var na = new NativeArray<T>(n, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            if (a.IsCreated)
            {
                NativeArray<T>.Copy(a, na, Mathf.Min(a.Length, n));
                a.Dispose();
            }
            a = na;
        }

        void CaptureRestPose()
        {
            GameObject probe = Instantiate(wormPrefab, Vector3.zero, Quaternion.identity);
            Transform[] b = FindBones(probe.transform);
            Vector3[] p = new Vector3[K];
            for (int i = 0; i < B; i++) p[i] = b[i].position;
            segLen = Vector3.Distance(p[0], p[1]);
            p[K - 1] = p[K - 2] + (p[K - 2] - p[K - 3]).normalized * segLen;

            // bone rotation relative to an (along-body, world-up) frame, so it can be rebuilt from the chain
            restOffset = new NativeArray<quaternion>(B, Allocator.Persistent);
            for (int i = 0; i < B; i++)
            {
                Vector3 d = (p[i + 1] - p[i]).normalized;
                restOffset[i] = Quaternion.Inverse(Quaternion.LookRotation(d, Vector3.up)) * b[i].rotation;
            }
            restRadius = new float[K];
            for (int k = 0; k < K; k++) restRadius[k] = BodyRadius(k * segLen);
            hashCell = 2f * 0.05f;
            Destroy(probe);
        }

        // matches the taper of the Blender mesh
        static float BodyRadius(float s)
        {
            const float maxR = 0.05f, bodyR = 0.047f, tipR = 0.006f, taperStart = 0.55f, length = 1.2f;
            if (s < 0.06f) return Mathf.Lerp(0.035f, maxR, s / 0.06f);
            if (s < taperStart) return bodyR;
            float k = Mathf.Clamp01((s - taperStart) / (length - taperStart));
            return Mathf.Max(0.012f, tipR + (bodyR - tipR) * (1f - k) * (1f + 0.6f * k));
        }

        Transform[] FindBones(Transform root)
        {
            var b = new Transform[B];
            for (int i = 0; i < B; i++)
            {
                b[i] = FindDeep(root, bonePrefix + i.ToString("00"));
                if (b[i] == null) Debug.LogError("[WormSwarm] Missing bone " + bonePrefix + i.ToString("00"));
            }
            return b;
        }

        static Transform FindDeep(Transform root, string n)
        {
            if (root.name == n) return root;
            for (int i = 0; i < root.childCount; i++)
            {
                var t = FindDeep(root.GetChild(i), n);
                if (t != null) return t;
            }
            return null;
        }

        void CollectLevelBoxes()
        {
            var list = new List<LevelBox>();
            foreach (var bc in FindObjectsByType<BoxCollider>(FindObjectsSortMode.None))
            {
                if (!bc.enabled || bc.isTrigger || ((1 << bc.gameObject.layer) & environmentMask) == 0) continue;
                Transform t = bc.transform;
                Vector3 s = t.lossyScale;
                Bounds aabb = bc.bounds;
                aabb.Expand(0.12f);
                LevelBox b;
                b.c = t.TransformPoint(bc.center);
                b.he = new float3(Mathf.Abs(bc.size.x * s.x), Mathf.Abs(bc.size.y * s.y), Mathf.Abs(bc.size.z * s.z)) * 0.5f;
                b.rot = t.rotation;
                b.inv = math.inverse(b.rot);
                b.aMin = aabb.min;
                b.aMax = aabb.max;
                list.Add(b);
            }
            Dispose(ref boxes);
            boxes = new NativeArray<LevelBox>(list.ToArray(), Allocator.Persistent);
        }

        [ContextMenu("Rebake Obstacles")]
        public void RebakeObstacles()
        {
            CollectLevelBoxes();
            if (Field == null) return;
            Field.BakeObstacles(obstacleMask, maxClimbHeight, obstacleInflate);
            if (target != null && Field.Build(target.position, true)) CopyFlow();
        }

        void CopyFlow() { flowDir.Reinterpret<Vector3>().CopyFrom(Field.dir); }

        void EnsureCapacity(int worms)
        {
            if (worms <= capacity) return;
            int newCap = Mathf.Max(worms, capacity * 2);
            int P = newCap * K;
            Grow(ref pos, P); Grow(ref prev, P); Grow(ref snap, P); Grow(ref delta, P);
            Grow(ref nLevel, P); Grow(ref nWorm, P); Grow(ref rad, P);
            Grow(ref contact, P); Grow(ref onLevel, P);
            Grow(ref cellHash, P); Grow(ref sorted, P); Grow(ref checks, P);
            Grow(ref heading, newCap); Grow(ref moveDir, newCap);
            Grow(ref phase, newCap); Grow(ref speedMul, newCap); Grow(ref seed, newCap);
            Grow(ref bonePos, newCap * B); Grow(ref boneRot, newCap * B);

            tableSize = Mathf.NextPowerOfTwo(P * 2);
            Dispose(ref hashCount); Dispose(ref hashStart);
            hashCount = new NativeArray<int>(tableSize, Allocator.Persistent);
            hashStart = new NativeArray<int>(tableSize, Allocator.Persistent);
            capacity = newCap;
        }

        void SpawnWorm()
        {
            EnsureCapacity(count + 1);
            int w = count++;

            if (wormObjects.Count <= w)
            {
                // root-level on purpose: IJobParallelForTransform only parallelises across root hierarchies
                GameObject go = Instantiate(wormPrefab, Vector3.zero, Quaternion.identity);
                go.name = "Worm_" + w.ToString("0000");
                var anim = go.GetComponent<Animator>();
                if (anim != null) Destroy(anim);
                foreach (var smr in go.GetComponentsInChildren<SkinnedMeshRenderer>()) smr.updateWhenOffscreen = true;
                Transform[] bones = FindBones(go.transform);
                // flatten the bone chain so each bone's world pose can be written independently
                Transform rig = bones[0].parent;
                for (int k = 1; k < B; k++) bones[k].SetParent(rig, true);
                wormObjects.Add(go);
                wormBones.Add(bones);
            }
            wormObjects[w].SetActive(true);
            if (boneTransforms.length == w * B)
                foreach (var t in wormBones[w]) boneTransforms.Add(t);

            Vector3 basePos = spawnPoint != null ? spawnPoint.position : transform.position;
            Vector3 fwd = Vector3.forward, head = basePos;
            for (int attempt = 0; attempt < 12; attempt++)
            {
                Vector2 disc = UnityEngine.Random.insideUnitCircle * spawnRadius;
                fwd = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f) * Vector3.forward;
                head = basePos + new Vector3(disc.x, UnityEngine.Random.Range(spawnHeight.x, spawnHeight.y), disc.y);
                Vector3 tail = head - fwd * (segLen * (K - 1));
                if (!Physics.CheckCapsule(head, tail, 0.06f, environmentMask, QueryTriggerInteraction.Ignore)) break;
            }

            for (int k = 0; k < K; k++)
            {
                int i = w * K + k;
                float3 p = head - fwd * (k * segLen);
                pos[i] = p; prev[i] = p; snap[i] = p;
                rad[i] = restRadius[k];
                contact[i] = 0; onLevel[i] = 0;
                nLevel[i] = float3.zero; nWorm[i] = float3.zero;
            }
            heading[w] = fwd;
            moveDir[w] = fwd;
            phase[w] = UnityEngine.Random.Range(0f, 10f);
            speedMul[w] = 1f + UnityEngine.Random.Range(-speedVariation, speedVariation);
            seed[w] = UnityEngine.Random.Range(0f, 1000f);
        }

        void UpdatePopulation(float dt)
        {
            if (count > maxWorms)
            {
                for (int w = maxWorms; w < count; w++) wormObjects[w].SetActive(false);
                count = maxWorms;
                // bone transforms must stay in worm order and hold only live worms
                boneTransforms.Dispose();
                boneTransforms = new TransformAccessArray(Mathf.Max(16, capacity * B));
                for (int w = 0; w < count; w++) foreach (var t in wormBones[w]) boneTransforms.Add(t);
            }
            if (count < maxWorms)
            {
                spawnAccumulator += spawnPerSecond * dt;
                while (spawnAccumulator >= 1f && count < maxWorms) { SpawnWorm(); spawnAccumulator -= 1f; }
            }
            else spawnAccumulator = 0f;
        }

        void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            UpdatePopulation(dt);
            if (target != null && Field.Build(target.position)) CopyFlow();
            if (count == 0) return;

            simWatch.Reset(); simWatch.Start();
            int P = count * K;
            NativeArray<float3>.Copy(pos, snap, P);

            JobHandle dep = new SteerJob
            {
                pos = pos, flowDir = flowDir, seed = seed, speedMul = speedMul,
                heading = heading, moveDir = moveDir, phase = phase,
                K = K, flowOrigin = Field.origin, cellSize = Field.cellSize, fw = Field.width, fh = Field.height,
                hasTarget = target != null, target = target != null ? (float3)target.position : float3.zero,
                arriveRadius = arriveRadius, time = Time.time, dt = dt,
                maxTurn = turnRate * Mathf.Deg2Rad * dt, wanderStrength = wanderStrength,
                wiggleFreq = wiggleFrequency, wiggleAmp = wiggleAmplitude * Mathf.Deg2Rad
            }.Schedule(count, 16);

            float h = dt / substeps;
            float invCell = 1f / hashCell;
            int mask = tableSize - 1;
            for (int s = 0; s < substeps; s++)
            {
                dep = new WormStepJob
                {
                    pos = pos, prev = prev, contact = contact, onLevel = onLevel,
                    nLevel = nLevel, nWorm = nWorm, moveDir = moveDir, speedMul = speedMul,
                    K = K, iterations = constraintIterations, h = h, segLen = segLen,
                    bendStiffness = bendStiffness, minFoldRatio = minFoldRatio, gravity = gravity,
                    keepAir = 1f - airDamping, keepSide = 1f - lateralFriction,
                    keepFwd = 1f - forwardFriction, keepWorm = 1f - wormFriction,
                    crawlSpeed = crawlSpeed, bodyThrust = bodyThrust, climbStrength = climbStrength,
                    wormClimbStrength = wormClimbStrength, floorY = areaCenter.y, maxClimbHeight = maxClimbHeight
                }.Schedule(count, 8, dep);

                dep = new HashJob { pos = pos, cellHash = cellHash, invCell = invCell, mask = mask }.Schedule(P, 128, dep);
                dep = new SortJob { cellHash = cellHash, hashCount = hashCount, hashStart = hashStart, sorted = sorted, P = P }.Schedule(dep);

                dep = new CollideJob
                {
                    pos = pos, rad = rad, sorted = sorted, hashStart = hashStart, hashCount = hashCount,
                    delta = delta, nWorm = nWorm, contact = contact, checks = checks,
                    K = K, mask = mask, invCell = invCell, stiff = 0.5f * wormCollisionStiffness
                }.Schedule(P, 64, dep);

                dep = new LevelCollideJob
                {
                    pos = pos, delta = delta, rad = rad, boxes = boxes,
                    contact = contact, onLevel = onLevel, nLevel = nLevel
                }.Schedule(P, 64, dep);

                dep = new ChainJob { pos = pos, K = K, segLen = segLen, bendStiffness = bendStiffness, minFoldRatio = minFoldRatio }
                    .Schedule(count, 8, dep);
            }
            dep.Complete();
            simWatch.Stop();
            simMs = Mathf.Lerp(simMs, (float)simWatch.Elapsed.TotalMilliseconds, 0.1f);
        }

        void LateUpdate()
        {
            if (!pos.IsCreated || count == 0) return;
            // interpolate between the last two physics steps for smooth motion at any frame rate
            float a = Mathf.Clamp01((Time.time - Time.fixedTime) / Time.fixedDeltaTime);

            JobHandle h = new BonePoseJob
            {
                pos = pos, snap = snap, heading = heading, restOffset = restOffset,
                bonePos = bonePos, boneRot = boneRot, K = K, B = B, alpha = a
            }.Schedule(count, 16);
            h = new ApplyBonesJob { bonePos = bonePos, boneRot = boneRot }.Schedule(boneTransforms, h);
            h.Complete();

            UpdateHud();
        }

        void UpdateHud()
        {
            fpsSmoothed = Mathf.Lerp(fpsSmoothed, 1f / Mathf.Max(1e-5f, Time.unscaledDeltaTime), 0.05f);
            hudTimer -= Time.unscaledDeltaTime;
            if (!showHud || hudTimer > 0f) return;
            hudTimer = 0.25f;
            long P = (long)count * K;
            long pairs = 0;
            for (int i = 0; i < P; i++) pairs += checks[i];
            pairs /= 2; // each pair is counted from both sides
            long brute = P * (P - 1) / 2;
            hudText = string.Format(
                "FPS {0:0}   [Jobs + Burst, {10} threads]\nWorms {1} / {2}   particles {3}\nSim {4:0.00} ms / step\nPair checks/substep {5:N0}\n(brute force would be {6:N0})\nFlow field {7}x{8}  rebuild {9:0.00} ms",
                fpsSmoothed, count, maxWorms, P, simMs, pairs, brute,
                Field.width, Field.height, Field.LastBuildMs, Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount + 1);
        }

        GUIStyle hudStyle;
        void OnGUI()
        {
            if (!showHud) return;
            if (hudStyle == null)
            {
                hudStyle = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, fontSize = 14 };
                hudStyle.normal.textColor = new Color(0.4f, 1f, 0.9f);
            }
            GUI.Box(new Rect(10, 10, 360, 120), hudText, hudStyle);
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying) return;
            if (showFlowField && Field != null)
            {
                float cs = Field.cellSize;
                float maxC = Mathf.Max(1f, Field.MaxCost);
                for (int z = 0; z < Field.height; z++)
                    for (int x = 0; x < Field.width; x++)
                    {
                        int i = Field.Index(x, z);
                        Vector3 c = Field.CellCenter(x, z) + Vector3.up * 0.01f;
                        if (Field.blocked[i])
                        {
                            Gizmos.color = new Color(1f, 0.2f, 0.2f, 0.35f);
                            Gizmos.DrawCube(c, new Vector3(cs * 0.9f, 0.01f, cs * 0.9f));
                            continue;
                        }
                        if (i == Field.TargetCell)
                        {
                            Gizmos.color = Color.white;
                            Gizmos.DrawCube(c, new Vector3(cs * 0.8f, 0.02f, cs * 0.8f));
                            continue;
                        }
                        float cost = Field.cost[i];
                        if (cost >= float.MaxValue) continue;
                        Gizmos.color = Color.Lerp(new Color(0.2f, 1f, 0.9f), new Color(0.1f, 0.3f, 0.6f), cost / maxC);
                        Vector3 d = Field.dir[i] * cs * 0.4f;
                        Gizmos.DrawLine(c - d, c + d);
                        Gizmos.DrawSphere(c + d, cs * 0.06f);
                    }
            }
            if (showParticles && pos.IsCreated)
            {
                for (int i = 0; i < count * K; i++)
                {
                    Gizmos.color = contact[i] != 0 ? new Color(0.2f, 1f, 0.9f) : Color.yellow;
                    Gizmos.DrawWireSphere(pos[i], rad[i]);
                }
            }
        }
    }
}
