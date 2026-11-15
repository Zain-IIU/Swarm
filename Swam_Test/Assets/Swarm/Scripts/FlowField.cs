using UnityEngine;

namespace SwarmTest
{
    public class FlowField
    {
        public readonly Vector3 origin;   // min corner (x, z); y = floor height
        public readonly float cellSize;
        public readonly int width, height;
        public readonly bool[] blocked;
        public readonly float[] cost;
        public readonly Vector3[] dir;

        public int TargetCell { get; private set; }
        public float LastBuildMs { get; private set; }
        public float MaxCost { get; private set; }

        const float Inf = float.MaxValue;
        static readonly int[] DX = { 1, -1, 0, 0, 1, 1, -1, -1 };
        static readonly int[] DZ = { 0, 0, 1, -1, 1, -1, 1, -1 };
        static readonly float[] DC = { 1f, 1f, 1f, 1f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f };

        readonly MinHeap heap;

        public FlowField(Vector3 center, Vector2 size, float cellSize)
        {
            this.cellSize = cellSize;
            width = Mathf.Max(2, Mathf.CeilToInt(size.x / cellSize));
            height = Mathf.Max(2, Mathf.CeilToInt(size.y / cellSize));
            origin = new Vector3(center.x - width * cellSize * 0.5f, center.y, center.z - height * cellSize * 0.5f);
            int n = width * height;
            blocked = new bool[n];
            cost = new float[n];
            dir = new Vector3[n];
            heap = new MinHeap(n);
            TargetCell = -1;
        }

        public int Index(int x, int z) { return z * width + x; }

        public bool WorldToCell(Vector3 p, out int x, out int z)
        {
            x = Mathf.FloorToInt((p.x - origin.x) / cellSize);
            z = Mathf.FloorToInt((p.z - origin.z) / cellSize);
            return x >= 0 && x < width && z >= 0 && z < height;
        }

        public Vector3 CellCenter(int x, int z)
        {
            return new Vector3(origin.x + (x + 0.5f) * cellSize, origin.y, origin.z + (z + 0.5f) * cellSize);
        }

        // A cell is blocked if any collider rises above the climb height inside it.
        public int BakeObstacles(LayerMask mask, float climbHeight, float inflate)
        {
            const float ceiling = 30f;
            var half = new Vector3(cellSize * 0.5f + inflate, (ceiling - climbHeight) * 0.5f, cellSize * 0.5f + inflate);
            int count = 0;
            for (int z = 0; z < height; z++)
                for (int x = 0; x < width; x++)
                {
                    Vector3 c = CellCenter(x, z);
                    c.y = origin.y + climbHeight + half.y;
                    bool b = Physics.CheckBox(c, half, Quaternion.identity, mask, QueryTriggerInteraction.Ignore);
                    blocked[Index(x, z)] = b;
                    if (b) count++;
                }
            TargetCell = -1;
            return count;
        }

        // Rebuilds only when the target enters a different cell. Returns true if rebuilt.
        public bool Build(Vector3 targetPos, bool force = false)
        {
            int tx, tz;
            if (!WorldToCell(targetPos, out tx, out tz)) return false;
            int t = Index(tx, tz);
            if (t == TargetCell && !force) return false;
            TargetCell = t;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            int n = cost.Length;
            for (int i = 0; i < n; i++) { cost[i] = Inf; dir[i] = Vector3.zero; }

            // Dijkstra from the target: path cost of every cell
            heap.Clear();
            cost[t] = 0f;
            heap.Push(t, 0f);
            MaxCost = 0f;
            while (heap.Count > 0)
            {
                int c; float cc;
                heap.Pop(out c, out cc);
                if (cc > cost[c]) continue;
                if (cc > MaxCost) MaxCost = cc;
                int cx = c % width, cz = c / width;
                for (int k = 0; k < 8; k++)
                {
                    int nx = cx + DX[k], nz = cz + DZ[k];
                    if (nx < 0 || nx >= width || nz < 0 || nz >= height) continue;
                    int ni = Index(nx, nz);
                    if (blocked[ni]) continue;
                    if (k >= 4 && (blocked[Index(nx, cz)] || blocked[Index(cx, nz)])) continue; // no corner cutting
                    float nc = cc + DC[k];
                    if (nc < cost[ni]) { cost[ni] = nc; heap.Push(ni, nc); }
                }
            }

            // each cell points at its cheapest neighbour
            for (int cz = 0; cz < height; cz++)
                for (int cx = 0; cx < width; cx++)
                {
                    int i = Index(cx, cz);
                    if (i == t) continue;
                    float best = cost[i];
                    Vector3 bestDir = Vector3.zero;
                    bool self = blocked[i];
                    for (int k = 0; k < 8; k++)
                    {
                        int nx = cx + DX[k], nz = cz + DZ[k];
                        if (nx < 0 || nx >= width || nz < 0 || nz >= height) continue;
                        int ni = Index(nx, nz);
                        if (blocked[ni]) continue;
                        if (!self && k >= 4 && (blocked[Index(nx, cz)] || blocked[Index(cx, nz)])) continue;
                        if (cost[ni] < best) { best = cost[ni]; bestDir = new Vector3(DX[k], 0f, DZ[k]); }
                    }
                    dir[i] = bestDir.normalized;
                }

            LastBuildMs = (float)sw.Elapsed.TotalMilliseconds;
            return true;
        }

        // Bilinear blend of the four nearest cell directions (not normalised; zero if unknown).
        public Vector3 Sample(Vector3 p)
        {
            float fx = (p.x - origin.x) / cellSize - 0.5f;
            float fz = (p.z - origin.z) / cellSize - 0.5f;
            int x0 = Mathf.FloorToInt(fx), z0 = Mathf.FloorToInt(fz);
            float tx = fx - x0, tz = fz - z0;
            Vector3 d = Vector3.zero;
            Acc(ref d, x0, z0, (1 - tx) * (1 - tz));
            Acc(ref d, x0 + 1, z0, tx * (1 - tz));
            Acc(ref d, x0, z0 + 1, (1 - tx) * tz);
            Acc(ref d, x0 + 1, z0 + 1, tx * tz);
            return d;
        }

        void Acc(ref Vector3 d, int x, int z, float w)
        {
            if (x < 0 || x >= width || z < 0 || z >= height) return;
            d += dir[Index(x, z)] * w;
        }

        public float CostAt(Vector3 p)
        {
            int x, z;
            return WorldToCell(p, out x, out z) ? cost[Index(x, z)] : Inf;
        }

        class MinHeap
        {
            int[] items; float[] keys;
            public int Count;
            public MinHeap(int capacity) { items = new int[capacity]; keys = new float[capacity]; }
            public void Clear() { Count = 0; }

            public void Push(int item, float key)
            {
                if (Count == items.Length)
                {
                    System.Array.Resize(ref items, Count * 2);
                    System.Array.Resize(ref keys, Count * 2);
                }
                int i = Count++;
                while (i > 0)
                {
                    int p = (i - 1) >> 1;
                    if (keys[p] <= key) break;
                    items[i] = items[p]; keys[i] = keys[p]; i = p;
                }
                items[i] = item; keys[i] = key;
            }

            public void Pop(out int item, out float key)
            {
                item = items[0]; key = keys[0];
                Count--;
                if (Count == 0) return;
                int lastI = items[Count]; float lastK = keys[Count];
                int i = 0;
                while (true)
                {
                    int l = 2 * i + 1;
                    if (l >= Count) break;
                    int r = l + 1;
                    int m = (r < Count && keys[r] < keys[l]) ? r : l;
                    if (keys[m] >= lastK) break;
                    items[i] = items[m]; keys[i] = keys[m]; i = m;
                }
                items[i] = lastI; keys[i] = lastK;
            }
        }
    }
}
