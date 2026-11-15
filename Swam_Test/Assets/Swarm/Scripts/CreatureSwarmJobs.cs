using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Jobs;
using static Unity.Mathematics.math;

namespace SwarmTest
{
    public struct LevelBox
    {
        public float3 c, he;
        public quaternion rot, inv;
        public float3 aMin, aMax;
    }

    static class SwarmMath
    {
        // same sign convention as Quaternion.AngleAxis(angle, Vector3.up)
        public static float3 RotY(float3 v, float a)
        {
            float s = sin(a), c = cos(a);
            return new float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
        }

        public static float3 RotateTowards(float3 from, float3 to, float maxAngle)
        {
            float a = atan2(from.z * to.x - from.x * to.z, from.x * to.x + from.z * to.z);
            return RotY(from, clamp(a, -maxAngle, maxAngle));
        }

        public static int Hash(int3 c, int mask)
        {
            return ((c.x * 73856093) ^ (c.y * 19349663) ^ (c.z * 83492791)) & mask;
        }

        public static void SolveChain(NativeArray<float3> pos, int b, int K, NativeArray<float> restLen, NativeArray<float> restBend,
                                      float bendStiffness, float minFoldRatio)
        {
            for (int k = 0; k < K - 1; k++)
            {
                int i = b + k;
                float3 d = pos[i + 1] - pos[i];
                float len = length(d);
                if (len < 1e-6f) continue;
                float3 c = d * (0.5f * (len - restLen[k]) / len);
                pos[i] += c; pos[i + 1] -= c;
            }
            for (int k = 0; k < K - 2; k++)
            {
                int i = b + k;
                float3 d = pos[i + 2] - pos[i];
                float len = length(d);
                if (len < 1e-6f) continue;
                float rest2 = restBend[k], minLen = rest2 * minFoldRatio;
                float goal = len < minLen ? minLen : lerp(len, rest2, bendStiffness);
                float3 c = d * (0.5f * (len - goal) / len);
                pos[i] += c; pos[i + 2] -= c;
            }
        }
    }

    [BurstCompile]
    public struct SteerJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> pos;
        [ReadOnly] public NativeArray<float3> flowDir;
        [ReadOnly] public NativeArray<float> seed, speedMul;
        public NativeArray<float3> heading, moveDir;
        public NativeArray<float> phase;
        [ReadOnly] public NativeArray<float4> repellers;   // x, z, radius, fear margin
        [WriteOnly] public NativeArray<float> panic;
        public float fleeBoost;
        public float2 areaMin, areaMax;
        public float edgeMargin;

        public int K;
        public float3 flowOrigin; public float cellSize; public int fw, fh;
        public bool hasTarget; public float3 target; public float arriveRadius;
        public float time, dt, maxTurn, wanderStrength, wiggleFreq, wiggleAmp;

        public void Execute(int w)
        {
            float3 hp = pos[w * K];
            float3 toT = target - hp; toT.y = 0f;
            float dist = length(toT);
            bool near = !hasTarget || dist < arriveRadius;

            float3 desired = heading[w];
            if (hasTarget)
            {
                float3 flow = SampleFlow(hp);
                if (!near && lengthsq(flow) > 1e-4f) desired = normalize(flow);
                else if (dist > 1e-3f) desired = toT / dist;
            }

            float n = noise.cnoise(new float2(seed[w], time * 0.35f));
            desired = SwarmMath.RotY(desired, n * wanderStrength * radians(90f) * (near ? 2f : 1f));

            // fear: inside a repeller's margin, steer away from its centre (strongest at the radius)
            float fear = 0f;
            float3 away = default(float3);
            for (int r = 0; r < repellers.Length; r++)
            {
                float4 rp = repellers[r];
                float2 d = hp.xz - rp.xy;
                float dl = length(d);
                float outer = rp.z + rp.w;
                if (dl >= outer) continue;
                float k = saturate((outer - dl) / max(rp.w, 1e-3f));
                float2 dir2 = dl > 1e-4f ? d / dl : new float2(1f, 0f);
                away += new float3(dir2.x, 0f, dir2.y) * k;
                fear = max(fear, k);
            }
            if (fear > 0f) desired = normalizesafe(desired * (1f - fear) + normalizesafe(away) * (2f * fear), desired);

            // keep away from the edge of the flow-field area
            float2 inward = default(float2);
            inward.x += saturate((areaMin.x + edgeMargin - hp.x) / edgeMargin) - saturate((hp.x - (areaMax.x - edgeMargin)) / edgeMargin);
            inward.y += saturate((areaMin.y + edgeMargin - hp.z) / edgeMargin) - saturate((hp.z - (areaMax.y - edgeMargin)) / edgeMargin);
            float edge = length(inward);
            if (edge > 1e-4f)
            {
                desired = normalizesafe(desired + new float3(inward.x, 0f, inward.y) * 2f, desired);
            }
            panic[w] = 1f + fleeBoost * fear;

            float3 hd = SwarmMath.RotateTowards(heading[w], desired, maxTurn * (1f + 2f * max(fear, saturate(edge))));
            hd.y = 0f;
            hd = normalizesafe(hd, new float3(0, 0, 1));
            heading[w] = hd;

            float ph = phase[w] + dt * wiggleFreq * 2f * PI * speedMul[w];
            phase[w] = ph;
            moveDir[w] = SwarmMath.RotY(hd, sin(ph) * wiggleAmp);
        }

        float3 SampleFlow(float3 p)
        {
            float fx = (p.x - flowOrigin.x) / cellSize - 0.5f;
            float fz = (p.z - flowOrigin.z) / cellSize - 0.5f;
            int x0 = (int)floor(fx), z0 = (int)floor(fz);
            float tx = fx - x0, tz = fz - z0;
            return Cell(x0, z0) * ((1 - tx) * (1 - tz)) + Cell(x0 + 1, z0) * (tx * (1 - tz))
                 + Cell(x0, z0 + 1) * ((1 - tx) * tz) + Cell(x0 + 1, z0 + 1) * (tx * tz);
        }

        float3 Cell(int x, int z)
        {
            if (x < 0 || x >= fw || z < 0 || z >= fh) return default(float3);
            return flowDir[z * fw + x];
        }
    }

    [BurstCompile]
    public struct StepJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> pos;
        [NativeDisableParallelForRestriction] public NativeArray<float3> prev;
        [ReadOnly] public NativeArray<byte> contact, onLevel;
        [ReadOnly] public NativeArray<float3> nLevel, nPile, moveDir;
        [ReadOnly] public NativeArray<float> speedMul, restLen, restBend, panic;

        public int K, iterations;
        public float h, bendStiffness, minFoldRatio;
        public float3 gravity;
        public float keepAir, keepSide, keepFwd, keepPile;
        public float moveSpeed, bodyThrust, climbStrength, pileClimbStrength, floorY, maxClimbHeight;

        public void Execute(int w)
        {
            int b = w * K;
            float3 g = gravity * (h * h);

            for (int k = 0; k < K; k++)
            {
                int i = b + k;
                float3 v = (pos[i] - prev[i]) * keepAir;
                if (onLevel[i] != 0)
                {
                    // anisotropic friction: slides along the body, grips sideways
                    float3 t = pos[k == 0 ? i : i - 1] - pos[k == K - 1 ? i : i + 1];
                    float tl = length(t);
                    if (tl > 1e-6f)
                    {
                        t /= tl;
                        float3 along = dot(v, t) * t;
                        v = along * keepFwd + (v - along) * keepSide;
                    }
                }
                else if (contact[i] != 0) v *= keepPile;
                prev[i] = pos[i];
                pos[i] += v + g;
            }

            float step = moveSpeed * speedMul[w] * panic[w] * h;
            if (contact[b] != 0 || contact[b + 1] != 0)
            {
                float3 md = moveDir[w];
                float3 hl = nLevel[b] + nLevel[b + 1], hw = nPile[b] + nPile[b + 1];
                float3 hn = hl + hw;
                if (lengthsq(hn) > 1e-6f)
                {
                    hn = normalize(hn);
                    float into = dot(md, hn);
                    float3 slide = md - dot(md, hn) * hn;
                    if (into < -0.2f && pos[b].y < floorY + maxClimbHeight)
                    {
                        float climb = lengthsq(hl) > 1e-6f ? climbStrength : pileClimbStrength;
                        md = slide + new float3(0, 1, 0) * (climb * -into);
                    }
                    else md = slide;
                    md = normalizesafe(md);
                }
                pos[b] += md * step;
            }

            if (bodyThrust > 0f)
            {
                float bs = step * bodyThrust;
                for (int k = 1; k < K; k++)
                {
                    int j = b + k;
                    if (contact[j] == 0) continue;
                    float3 t = pos[j - 1] - pos[j];
                    float tl = length(t);
                    if (tl > 1e-6f) pos[j] += t * (bs / tl);
                }
            }

            for (int it = 0; it < iterations; it++)
                SwarmMath.SolveChain(pos, b, K, restLen, restBend, bendStiffness, minFoldRatio);
        }
    }

    [BurstCompile]
    public struct HashJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> pos;
        [WriteOnly] public NativeArray<int> cellHash;
        public float invCell; public int mask;

        public void Execute(int i)
        {
            cellHash[i] = SwarmMath.Hash((int3)floor(pos[i] * invCell), mask);
        }
    }

    // counting sort of particles into hash buckets
    [BurstCompile]
    public struct SortJob : IJob
    {
        [ReadOnly] public NativeArray<int> cellHash;
        public NativeArray<int> hashCount, hashStart, sorted;
        public int P;

        public void Execute()
        {
            for (int h = 0; h < hashCount.Length; h++) hashCount[h] = 0;
            for (int i = 0; i < P; i++) hashCount[cellHash[i]]++;
            int run = 0;
            for (int h = 0; h < hashCount.Length; h++) { run += hashCount[h]; hashStart[h] = run; }
            for (int i = 0; i < P; i++)
            {
                int h = cellHash[i];
                int s = hashStart[h] - 1;
                hashStart[h] = s;
                sorted[s] = i;
            }
        }
    }

    // Each particle accumulates only its own push (Jacobi), so this is safe to run in parallel.
    [BurstCompile]
    public struct CollideJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> pos;
        [ReadOnly] public NativeArray<float> rad;
        [ReadOnly] public NativeArray<int> sorted, hashStart, hashCount;
        [WriteOnly] public NativeArray<float3> delta, nPile;
        [WriteOnly] public NativeArray<byte> contact;
        [WriteOnly] public NativeArray<int> checks;
        public int K, mask; public float invCell, stiff;

        public void Execute(int i)
        {
            int wi = i / K;
            float3 pi = pos[i];
            float ri = rad[i];
            int3 c = (int3)floor(pi * invCell);
            float3 d = default(float3), nAcc = default(float3);
            byte hit = 0;
            int chk = 0;
            for (int ox = -1; ox <= 1; ox++)
                for (int oy = -1; oy <= 1; oy++)
                    for (int oz = -1; oz <= 1; oz++)
                    {
                        int h = SwarmMath.Hash(c + new int3(ox, oy, oz), mask);
                        int start = hashStart[h], end = start + hashCount[h];
                        for (int s = start; s < end; s++)
                        {
                            int j = sorted[s];
                            if (j == i) continue;
                            if (j / K == wi && abs(j - i) <= 2) continue;
                            chk++;
                            float3 dd = pos[j] - pi;
                            float rr = ri + rad[j];
                            float d2 = lengthsq(dd);
                            if (d2 >= rr * rr || d2 < 1e-12f) continue;
                            float dist = sqrt(d2);
                            float3 n = dd / dist;
                            d -= n * ((rr - dist) * stiff);
                            nAcc -= n;
                            hit = 1;
                        }
                    }
            delta[i] = d;
            nPile[i] = nAcc;
            contact[i] = hit;
            checks[i] = chk;
        }
    }

    [BurstCompile]
    public struct LevelCollideJob : IJobParallelFor
    {
        public NativeArray<float3> pos;
        [ReadOnly] public NativeArray<float3> delta, prev;
        [ReadOnly] public NativeArray<float> rad;
        [ReadOnly] public NativeArray<LevelBox> boxes;
        public NativeArray<byte> contact;
        [WriteOnly] public NativeArray<byte> onLevel;
        [WriteOnly] public NativeArray<float3> nLevel;
        [ReadOnly] public NativeArray<float4> repellers;
        public float maxRepelPush;
        public float2 areaMin, areaMax;

        public void Execute(int i)
        {
            float3 p = pos[i] + delta[i];
            float r = rad[i];
            byte lv = 0;
            float3 nl = default(float3);

            // repellers are vertical cylinders nothing may enter; push out at a capped speed so creatures scatter instead of teleporting
            for (int k = 0; k < repellers.Length; k++)
            {
                float4 rp = repellers[k];
                float2 d = p.xz - rp.xy;
                float minD = rp.z + r;
                float d2 = lengthsq(d);
                if (d2 >= minD * minD) continue;
                float dl = sqrt(d2);
                float2 n = dl > 1e-5f ? d / dl : new float2(1f, 0f);
                p.xz += n * min(minD - dl, maxRepelPush);
            }

            // the flow-field area is a hard boundary
            p.x = clamp(p.x, areaMin.x + r, areaMax.x - r);
            p.z = clamp(p.z, areaMin.y + r, areaMax.y - r);
            for (int b = 0; b < boxes.Length; b++)
            {
                LevelBox bx = boxes[b];
                if (any(p < bx.aMin) || any(p > bx.aMax)) continue;
                float3 q = mul(bx.inv, p - bx.c);
                float3 cl = clamp(q, -bx.he, bx.he);
                float3 d = q - cl;
                float d2 = lengthsq(d);
                if (d2 > r * r) continue;
                float3 nLocal;
                if (d2 > 1e-12f)
                {
                    nLocal = d / sqrt(d2);
                    q = cl + nLocal * r;
                }
                else
                {
                    // centre inside the box: leave through the face it came in from (where it was at the start of
                    // the step), falling back to the shallowest face. Avoids being shoved out of the far side.
                    float3 qp = mul(bx.inv, prev[i] - bx.c);
                    float3 outside = abs(qp) - bx.he;
                    int axis;
                    if (any(outside > 0f))
                        axis = outside.x >= outside.y && outside.x >= outside.z ? 0 : (outside.y >= outside.z ? 1 : 2);
                    else
                    {
                        float3 pen = bx.he - abs(q);
                        axis = pen.y <= pen.x && pen.y <= pen.z ? 1 : (pen.x <= pen.z ? 0 : 2);
                        qp = q;
                    }
                    float sg = qp[axis] >= 0f ? 1f : -1f;
                    nLocal = default(float3);
                    nLocal[axis] = sg;
                    q[axis] = sg * (bx.he[axis] + r);
                }
                p = bx.c + mul(bx.rot, q);
                lv = 1;
                nl += mul(bx.rot, nLocal);
            }
            pos[i] = p;
            onLevel[i] = lv;
            nLevel[i] = nl;
            if (lv != 0) contact[i] = 1;
        }
    }

    [BurstCompile]
    public struct ChainJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> pos;
        [ReadOnly] public NativeArray<float> restLen, restBend;
        public int K; public float bendStiffness, minFoldRatio;

        public void Execute(int w) { SwarmMath.SolveChain(pos, w * K, K, restLen, restBend, bendStiffness, minFoldRatio); }
    }

    [BurstCompile]
    public struct BonePoseJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> pos, snap, heading;
        [ReadOnly] public NativeArray<quaternion> restOffset;
        [ReadOnly] public NativeArray<float> lift;
        [NativeDisableParallelForRestriction] public NativeArray<float3> bonePos;
        [NativeDisableParallelForRestriction] public NativeArray<quaternion> boneRot;
        public int K, B; public float alpha;

        public void Execute(int w)
        {
            int b = w * K, o = w * B;
            float3 up = new float3(0, 1, 0);
            // particles sit on the ground at their collision radius; lift raises each joint back to its modelled height
            float3 p0 = lerp(snap[b], pos[b], alpha) + new float3(0, lift[0], 0);
            quaternion rot = Unity.Mathematics.quaternion.identity;
            for (int k = 0; k < B; k++)
            {
                float3 p1 = lerp(snap[b + k + 1], pos[b + k + 1], alpha) + new float3(0, lift[k + 1], 0);
                float3 dir = p1 - p0;
                if (lengthsq(dir) > 1e-10f)
                {
                    dir = normalize(dir);
                    up -= dot(up, dir) * dir;   // parallel transport so the body never twists
                    if (lengthsq(up) < 1e-6f) up = heading[w] - dot(heading[w], dir) * dir;
                    up = normalizesafe(up, new float3(0, 1, 0));
                    rot = mul(Unity.Mathematics.quaternion.LookRotation(dir, up), restOffset[k]);
                }
                bonePos[o + k] = p0;
                boneRot[o + k] = rot;
                p0 = p1;
            }
        }
    }

    [BurstCompile]
    public struct ApplyBonesJob : IJobParallelForTransform
    {
        [ReadOnly] public NativeArray<float3> bonePos;
        [ReadOnly] public NativeArray<quaternion> boneRot;

        public void Execute(int index, TransformAccess t)
        {
            t.SetPositionAndRotation(bonePos[index], boneRot[index]);
        }
    }
}
