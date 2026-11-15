using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;

namespace SwarmTest
{
    public class CreatureSwarm : MonoBehaviour
    {
        [Header("Swarm")]
        [Tooltip("Swap this (even in Play mode) to respawn the swarm as another creature")]
        public CreatureProfile creature;
        [Min(0)] public int maxCount = 300;
        public float spawnPerSecond = 60f;
        public Transform spawnPoint;
        public float spawnRadius = 2f;
        public Vector2 spawnHeight = new Vector2(0.15f, 0.6f);
        public Transform target;
        [Tooltip("Inside this radius creatures ignore the flow field and mill around the target")]
        public float arriveRadius = 0.7f;

        [Header("Flow Field")]
        public Vector3 areaCenter = Vector3.zero;
        public Vector2 areaSize = new Vector2(14f, 14f);
        public float cellSize = 0.25f;
        public LayerMask obstacleMask = ~(1 << 2);
        [Tooltip("Taller than this is pathed around; lower is climbed over")]
        public float maxClimbHeight = 0.6f;
        public float obstacleInflate = 0.05f;
        [Tooltip("Creatures start turning back this far from the edge of the area; they can never leave it")]
        public float edgeMargin = 0.6f;

        [Header("Physics")]
        public Vector3 gravity = new Vector3(0f, -9.81f, 0f);
        [Range(1, 6)] public int substeps = 2;
        [Range(1, 12)] public int constraintIterations = 4;
        [Range(0f, 1f)] public float airDamping = 0.01f;
        [Tooltip("Only BoxColliders on these layers are collided with")]
        public LayerMask environmentMask = ~(1 << 2);

        [Header("Repellers")]
        [Tooltip("Speed multiplier added at full fear (1 = twice as fast when right at the light's edge)")]
        public float fleeSpeedBoost = 1f;
        [Tooltip("Flow-field cost multiplier inside lit areas; high = route around, waiting at the edge if there's no other way")]
        public float lightPathPenalty = 40f;
        [Tooltip("Max speed (m/s) at which creatures caught inside a new light are pushed out")]
        public float repelPushSpeed = 4f;

        [Header("Debug")]
        public bool showHud = true;
        public bool showFlowField = true;
        public bool showParticles = false;

        // particle k of creature c lives at index c * K + k
        int K, B;
        NativeArray<float3> pos, prev, snap, delta, nLevel, nPile;
        NativeArray<float> rad;
        NativeArray<byte> contact, onLevel;
        NativeArray<int> cellHash, sorted, checks;
        NativeArray<int> hashCount, hashStart;
        int tableSize;
        float hashCell;

        NativeArray<float> restLen, restBend, lift;
        Vector3[] restShape;   // rest particle positions relative to the head, facing +Z
        float[] jointRadius;

        int count, capacity;
        NativeArray<float3> heading, moveDir;
        NativeArray<float> phase, speedMul, seed, panic;
        readonly List<Vector4> zones = new List<Vector4>();
        readonly List<Vector4> pathZones = new List<Vector4>();

        NativeArray<quaternion> restOffset;
        NativeArray<float3> bonePos;
        NativeArray<quaternion> boneRot;
        TransformAccessArray boneTransforms;
        readonly List<Transform[]> creatureBones = new List<Transform[]>();
        readonly List<GameObject> creatureObjects = new List<GameObject>();

        NativeArray<LevelBox> boxes;
        NativeArray<float3> flowDir;
        public FlowField Field { get; private set; }
        public int Count { get { return count; } }

        CreatureProfile active;
        float spawnAccumulator;

        readonly System.Diagnostics.Stopwatch simWatch = new System.Diagnostics.Stopwatch();
        float simMs, fpsSmoothed, hudTimer;
        string hudText = "";

        void Start()
        {
            Field = new FlowField(areaCenter, areaSize, cellSize);
            flowDir = new NativeArray<float3>(Field.dir.Length, Allocator.Persistent);
            RebakeObstacles();
            Activate(creature);
        }

        void OnDestroy()
        {
            DisposeCreatureData();
            Dispose(ref boxes); Dispose(ref flowDir);
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

        void DisposeCreatureData()
        {
            Dispose(ref pos); Dispose(ref prev); Dispose(ref snap); Dispose(ref delta);
            Dispose(ref nLevel); Dispose(ref nPile); Dispose(ref rad);
            Dispose(ref contact); Dispose(ref onLevel);
            Dispose(ref cellHash); Dispose(ref sorted); Dispose(ref checks);
            Dispose(ref hashCount); Dispose(ref hashStart);
            Dispose(ref heading); Dispose(ref moveDir); Dispose(ref phase); Dispose(ref speedMul); Dispose(ref seed); Dispose(ref panic);
            Dispose(ref restOffset); Dispose(ref bonePos); Dispose(ref boneRot);
            Dispose(ref restLen); Dispose(ref restBend); Dispose(ref lift);
            if (boneTransforms.isCreated) boneTransforms.Dispose();
            capacity = 0;
        }

        void Activate(CreatureProfile p)
        {
            foreach (var go in creatureObjects) if (go != null) Destroy(go);
            creatureObjects.Clear();
            creatureBones.Clear();
            count = 0;
            spawnAccumulator = 0f;
            DisposeCreatureData();
            active = p;
            if (p == null || p.prefab == null) return;

            B = p.boneCount;
            K = B + 1;
            if (!CaptureRestPose(p)) { active = null; return; }
            boneTransforms = new TransformAccessArray(Mathf.Max(16, maxCount) * B);
            EnsureCapacity(Mathf.Max(16, maxCount));
        }

        bool CaptureRestPose(CreatureProfile p)
        {
            GameObject probe = Instantiate(p.prefab, Vector3.zero, Quaternion.identity);
            Transform[] b = FindBones(probe.transform, p);
            if (b == null) { Destroy(probe); return false; }

            // lowest point of the mesh = the ground the creature stands on
            float ground = float.MaxValue;
            var smrs = probe.GetComponentsInChildren<SkinnedMeshRenderer>();
            var verts = new List<Vector3>();
            foreach (var smr in smrs)
            {
                smr.sharedMesh.GetVertices(verts);
                foreach (var v in verts) ground = Mathf.Min(ground, smr.transform.TransformPoint(v).y);
            }
            if (ground == float.MaxValue) ground = 0f;

            Vector3[] j = new Vector3[K];
            for (int i = 0; i < B; i++) j[i] = b[i].position;

            // tail tip: the last bone has no end in the FBX, so take the furthest vertex along its direction
            Vector3 lastDir = (j[B - 1] - j[B - 2]).normalized;
            float tip = 0f;
            foreach (var smr in smrs)
            {
                smr.sharedMesh.GetVertices(verts);
                foreach (var v in verts) tip = Mathf.Max(tip, Vector3.Dot(smr.transform.TransformPoint(v) - j[B - 1], lastDir));
            }
            j[K - 1] = j[B - 1] + lastDir * Mathf.Max(tip, 0.01f);

            jointRadius = new float[K];
            for (int k = 0; k < K; k++)
                jointRadius[k] = (p.jointRadius != null && k < p.jointRadius.Length) ? p.jointRadius[k] : 0.03f;
            if (p.jointRadius == null || p.jointRadius.Length != K)
                Debug.LogWarning("[CreatureSwarm] " + p.name + " needs " + K + " joint radii", p);

            // simulated particles rest on the ground at their radius; lift restores the modelled height when rendering
            Vector3[] s = new Vector3[K];
            lift = new NativeArray<float>(K, Allocator.Persistent);
            float maxR = 0f;
            for (int k = 0; k < K; k++)
            {
                float l = Mathf.Max(0f, (j[k].y - ground) - jointRadius[k]);
                lift[k] = l;
                s[k] = j[k] - Vector3.up * l;
                maxR = Mathf.Max(maxR, jointRadius[k]);
            }
            restShape = new Vector3[K];
            for (int k = 0; k < K; k++) restShape[k] = s[k] - s[0];
            restLen = new NativeArray<float>(K - 1, Allocator.Persistent);
            restBend = new NativeArray<float>(K - 2, Allocator.Persistent);
            for (int k = 0; k < K - 1; k++) restLen[k] = Vector3.Distance(s[k], s[k + 1]);
            for (int k = 0; k < K - 2; k++) restBend[k] = Vector3.Distance(s[k], s[k + 2]);

            // bone rotation relative to an (along-body, world-up) frame, so it can be rebuilt from the chain
            restOffset = new NativeArray<quaternion>(B, Allocator.Persistent);
            for (int i = 0; i < B; i++)
                restOffset[i] = Quaternion.Inverse(Quaternion.LookRotation((j[i + 1] - j[i]).normalized, Vector3.up)) * b[i].rotation;

            hashCell = 2f * maxR;
            Destroy(probe);
            return true;
        }

        Transform[] FindBones(Transform root, CreatureProfile p)
        {
            var b = new Transform[B];
            for (int i = 0; i < B; i++)
            {
                b[i] = FindDeep(root, p.bonePrefix + i.ToString("00"));
                if (b[i] == null)
                {
                    Debug.LogError("[CreatureSwarm] " + p.prefab.name + " is missing bone " + p.bonePrefix + i.ToString("00"), p);
                    return null;
                }
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
                Vector3 sc = t.lossyScale;
                Bounds aabb = bc.bounds;
                aabb.Expand(0.12f);
                LevelBox lb;
                lb.c = t.TransformPoint(bc.center);
                lb.he = new float3(Mathf.Abs(bc.size.x * sc.x), Mathf.Abs(bc.size.y * sc.y), Mathf.Abs(bc.size.z * sc.z)) * 0.5f;
                lb.rot = t.rotation;
                lb.inv = math.inverse(lb.rot);
                lb.aMin = aabb.min;
                lb.aMax = aabb.max;
                list.Add(lb);
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

        float2 AreaMin { get { return new float2(Field.origin.x, Field.origin.z); } }
        float2 AreaMax { get { return new float2(Field.origin.x + Field.width * Field.cellSize, Field.origin.z + Field.height * Field.cellSize); } }

        Vector3 ClampToArea(Vector3 p, float margin)
        {
            float2 mn = AreaMin + margin, mx = AreaMax - margin;
            p.x = Mathf.Clamp(p.x, mn.x, mx.x);
            p.z = Mathf.Clamp(p.z, mn.y, mx.y);
            return p;
        }

        void EnsureCapacity(int n)
        {
            if (n <= capacity) return;
            int newCap = Mathf.Max(n, capacity * 2);
            int P = newCap * K;
            Grow(ref pos, P); Grow(ref prev, P); Grow(ref snap, P); Grow(ref delta, P);
            Grow(ref nLevel, P); Grow(ref nPile, P); Grow(ref rad, P);
            Grow(ref contact, P); Grow(ref onLevel, P);
            Grow(ref cellHash, P); Grow(ref sorted, P); Grow(ref checks, P);
            Grow(ref heading, newCap); Grow(ref moveDir, newCap);
            Grow(ref phase, newCap); Grow(ref speedMul, newCap); Grow(ref seed, newCap); Grow(ref panic, newCap);
            Grow(ref bonePos, newCap * B); Grow(ref boneRot, newCap * B);

            tableSize = Mathf.NextPowerOfTwo(P * 2);
            Dispose(ref hashCount); Dispose(ref hashStart);
            hashCount = new NativeArray<int>(tableSize, Allocator.Persistent);
            hashStart = new NativeArray<int>(tableSize, Allocator.Persistent);
            capacity = newCap;
        }

        void Spawn()
        {
            EnsureCapacity(count + 1);
            int c = count++;

            if (creatureObjects.Count <= c)
            {
                // root-level on purpose: IJobParallelForTransform only parallelises across root hierarchies
                GameObject go = Instantiate(active.prefab, Vector3.zero, Quaternion.identity);
                go.name = active.prefab.name + "_" + c.ToString("0000");
                var anim = go.GetComponent<Animator>();
                if (anim != null) Destroy(anim);
                foreach (var smr in go.GetComponentsInChildren<SkinnedMeshRenderer>()) smr.updateWhenOffscreen = true;
                Transform[] bones = FindBones(go.transform, active);
                // flatten the bone chain so each bone's world pose can be written independently
                Transform rig = bones[0].parent;
                for (int k = 1; k < B; k++) bones[k].SetParent(rig, true);
                creatureObjects.Add(go);
                creatureBones.Add(bones);
            }
            creatureObjects[c].SetActive(true);
            if (boneTransforms.length == c * B)
                foreach (var t in creatureBones[c]) boneTransforms.Add(t);

            Place(c);
        }

        void Place(int c)
        {
            Vector3 basePos = spawnPoint != null ? spawnPoint.position : transform.position;
            Quaternion yaw = Quaternion.identity;
            Vector3 head = basePos;
            for (int attempt = 0; attempt < 12; attempt++)
            {
                Vector2 disc = UnityEngine.Random.insideUnitCircle * spawnRadius;
                yaw = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);
                head = basePos + new Vector3(disc.x, UnityEngine.Random.Range(spawnHeight.x, spawnHeight.y), disc.y);
                float bodyLen = restShape[K - 1].magnitude;
                head = ClampToArea(head, Mathf.Min(bodyLen, Mathf.Min(areaSize.x, areaSize.y) * 0.45f));
                if (!Physics.CheckCapsule(head, head + yaw * restShape[K - 1], 0.06f, environmentMask, QueryTriggerInteraction.Ignore)) break;
            }

            for (int k = 0; k < K; k++)
            {
                int i = c * K + k;
                float3 p = head + yaw * restShape[k];
                pos[i] = p; prev[i] = p; snap[i] = p;
                rad[i] = jointRadius[k];
                contact[i] = 0; onLevel[i] = 0;
                nLevel[i] = float3.zero; nPile[i] = float3.zero;
            }
            Vector3 fwd = yaw * Vector3.forward;
            heading[c] = fwd;
            moveDir[c] = fwd;
            phase[c] = UnityEngine.Random.Range(0f, 10f);
            speedMul[c] = 1f + UnityEngine.Random.Range(-active.speedVariation, active.speedVariation);
            seed[c] = UnityEngine.Random.Range(0f, 1000f);
            panic[c] = 1f;
        }

        void UpdatePopulation(float dt)
        {
            if (count > maxCount)
            {
                for (int c = maxCount; c < count; c++) creatureObjects[c].SetActive(false);
                count = maxCount;
                // bone transforms must stay in creature order and hold only live creatures
                boneTransforms.Dispose();
                boneTransforms = new TransformAccessArray(Mathf.Max(16, capacity * B));
                for (int c = 0; c < count; c++) foreach (var t in creatureBones[c]) boneTransforms.Add(t);
            }
            if (count < maxCount)
            {
                spawnAccumulator += spawnPerSecond * dt;
                while (spawnAccumulator >= 1f && count < maxCount) { Spawn(); spawnAccumulator -= 1f; }
            }
            else spawnAccumulator = 0f;
        }

        void FixedUpdate()
        {
            if (creature != active) Activate(creature);
            if (active == null) return;

            float dt = Time.fixedDeltaTime;
            UpdatePopulation(dt);
            zones.Clear(); pathZones.Clear();
            foreach (var r in SwarmRepeller.Active)
            {
                if (!r.IsLit) continue;
                Vector3 rp = r.transform.position;
                zones.Add(new Vector4(rp.x, rp.z, r.radius, r.fearMargin));
                pathZones.Add(new Vector4(rp.x, rp.z, r.radius + r.fearMargin * 0.5f, 0f));
            }
            bool lightsChanged = Field.SetPenalty(pathZones, lightPathPenalty);
            Vector3 targetPos = target != null ? ClampToArea(target.position, 0.05f) : Vector3.zero;
            if (target != null && Field.Build(targetPos, lightsChanged)) CopyFlow();
            if (count == 0) return;

            var rep = new NativeArray<float4>(zones.Count, Allocator.TempJob);
            for (int i = 0; i < zones.Count; i++) rep[i] = zones[i];

            simWatch.Reset(); simWatch.Start();
            int P = count * K;
            NativeArray<float3>.Copy(pos, snap, P);
            CreatureProfile cp = active;

            JobHandle dep = new SteerJob
            {
                pos = pos, flowDir = flowDir, seed = seed, speedMul = speedMul,
                heading = heading, moveDir = moveDir, phase = phase,
                K = K, flowOrigin = Field.origin, cellSize = Field.cellSize, fw = Field.width, fh = Field.height,
                hasTarget = target != null, target = targetPos,
                areaMin = AreaMin, areaMax = AreaMax, edgeMargin = Mathf.Max(0.01f, edgeMargin),
                arriveRadius = arriveRadius, time = Time.time, dt = dt,
                maxTurn = cp.turnRate * Mathf.Deg2Rad * dt, wanderStrength = cp.wanderStrength,
                wiggleFreq = cp.wiggleFrequency, wiggleAmp = cp.wiggleAmplitude * Mathf.Deg2Rad,
                repellers = rep, panic = panic, fleeBoost = fleeSpeedBoost
            }.Schedule(count, 16);

            float h = dt / substeps;
            float invCell = 1f / hashCell;
            int mask = tableSize - 1;
            for (int s = 0; s < substeps; s++)
            {
                dep = new StepJob
                {
                    pos = pos, prev = prev, contact = contact, onLevel = onLevel,
                    nLevel = nLevel, nPile = nPile, moveDir = moveDir, speedMul = speedMul, panic = panic,
                    restLen = restLen, restBend = restBend,
                    K = K, iterations = constraintIterations, h = h,
                    bendStiffness = cp.bendStiffness, minFoldRatio = cp.minFoldRatio, gravity = gravity,
                    keepAir = 1f - airDamping, keepSide = 1f - cp.lateralFriction,
                    keepFwd = 1f - cp.forwardFriction, keepPile = 1f - cp.pileFriction,
                    moveSpeed = cp.moveSpeed, bodyThrust = cp.bodyThrust, climbStrength = cp.climbStrength,
                    pileClimbStrength = cp.pileClimbStrength, floorY = areaCenter.y, maxClimbHeight = maxClimbHeight
                }.Schedule(count, 8, dep);

                dep = new HashJob { pos = pos, cellHash = cellHash, invCell = invCell, mask = mask }.Schedule(P, 128, dep);
                dep = new SortJob { cellHash = cellHash, hashCount = hashCount, hashStart = hashStart, sorted = sorted, P = P }.Schedule(dep);

                dep = new CollideJob
                {
                    pos = pos, rad = rad, sorted = sorted, hashStart = hashStart, hashCount = hashCount,
                    delta = delta, nPile = nPile, contact = contact, checks = checks,
                    K = K, mask = mask, invCell = invCell, stiff = 0.5f * cp.collisionStiffness
                }.Schedule(P, 64, dep);

                dep = new LevelCollideJob
                {
                    pos = pos, delta = delta, prev = prev, rad = rad, boxes = boxes,
                    contact = contact, onLevel = onLevel, nLevel = nLevel,
                    repellers = rep, maxRepelPush = repelPushSpeed * h,
                    areaMin = AreaMin, areaMax = AreaMax
                }.Schedule(P, 64, dep);

                dep = new ChainJob
                {
                    pos = pos, restLen = restLen, restBend = restBend, K = K,
                    bendStiffness = cp.bendStiffness, minFoldRatio = cp.minFoldRatio
                }.Schedule(count, 8, dep);
            }
            dep.Complete();
            rep.Dispose();

            for (int c = 0; c < count; c++)
                if (pos[c * K].y < areaCenter.y - 2f) Place(c);   // fell out of the level
            simWatch.Stop();
            simMs = Mathf.Lerp(simMs, (float)simWatch.Elapsed.TotalMilliseconds, 0.1f);
        }

        void LateUpdate()
        {
            if (active == null || !pos.IsCreated || count == 0) return;
            // interpolate between the last two physics steps for smooth motion at any frame rate
            float a = Mathf.Clamp01((Time.time - Time.fixedTime) / Time.fixedDeltaTime);

            JobHandle h = new BonePoseJob
            {
                pos = pos, snap = snap, heading = heading, restOffset = restOffset, lift = lift,
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
                "FPS {0:0}   [Jobs + Burst, {10} threads]\n{11} {1} / {2}   particles {3}\nSim {4:0.00} ms / step\nPair checks/substep {5:N0}\n(brute force would be {6:N0})\nFlow field {7}x{8}  rebuild {9:0.00} ms",
                fpsSmoothed, count, maxCount, P, simMs, pairs, brute,
                Field.width, Field.height, Field.LastBuildMs, Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount + 1,
                active != null ? active.name : "-");
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
            Gizmos.color = new Color(0.2f, 1f, 0.9f, 0.8f);
            Gizmos.DrawWireCube(new Vector3(areaCenter.x, areaCenter.y + 0.5f, areaCenter.z), new Vector3(areaSize.x, 1f, areaSize.y));
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
                        if (Field.penalty[i] > 0f)
                        {
                            Gizmos.color = new Color(1f, 0.6f, 0.15f, 0.25f);
                            Gizmos.DrawCube(c, new Vector3(cs * 0.9f, 0.01f, cs * 0.9f));
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
