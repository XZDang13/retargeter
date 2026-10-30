// Offline whole-trajectory retargeting: one cooperative thread group per frame of the current colour.
//
// The global objective over all frames of a clip is the sum of per-frame terms (tracking, stance-foot
// locking, ground, ground contact of mapped body parts, self-collision, posture, limits) and coupling
// terms between neighbouring frames (velocity and acceleration smoothness). Frames are coloured t mod
// (2K + 1) (K >= 1, see the dynamics rows), so no two frames updated together share a residual; one
// dispatch updates every frame of one colour
// with a few LM steps on its local problem (neighbours fixed). Alternating the colours is block
// Gauss-Seidel on the global objective.
//
// Per-frame residual rows (dense, weighted, in groupshared memory):
//   6 per tracking target | 3 per sole point (stance lock to its anchor, else one-sided ground)
//   1 per link (lowest capsule point: equality to the ground while the mapped human part touches it,
//   else a one-sided barrier) | 1 per active self-collision pair (capsule clearance) |
//   6 per frame s = t-K, t, t+K: centroidal dynamics (see dynamics rows below).
//
// Centroidal dynamics rows: with point-mass links, the gravito-inertial wrench of frame s about its CoM,
//   F_s = sum m_i (a_i - g),  N_s = sum m_i (c_i - c) x (a_i - g),  a_i = (c_i,s+1 - 2 c_i,s + c_i,s-1) fps^2,
// has to be supplied by the ground at the frame's contact points (stance sole corners, mapped body-part
// contacts) through friction pyramids. The contact forces are re-solved in the kernel (non-negative least
// squares by coordinate descent) at every step; the residual wrench (F / mg, N / (mg l)) is penalised, with
// the contact forces held fixed for the Jacobian (the gradient of the squared distance to the cone).
// Accelerations use frames s ± K (K = dynamicsStride, ~25 ms: the band the validator judges); frame t's
// joints move the link CoMs of frame t, which enter the accelerations of s = t-K, t, t+K.
import retarget_ik_common;

static const uint O_MAX_PAIR_ROWS = 24;
static const uint SOLE_POINTS = 8; // 2 feet x 4
static const uint CONTACT_POINTS = 24;
static const uint DYN_ROWS = 18;   // 3 frames x 6
static const uint SMOOTH_LINKS = 2;
static const uint SMOOTH_ROWS = SMOOTH_LINKS * 9;   // per link: 3 frames x 3
// Threads per frame (two warps: twice the lanes for the wide loops and twice the resident warps per SM for the same
// groupshared memory). A multiple of 32; 256 made a fixed-lag window tick no faster.
#ifndef OFFLINE_GROUP
#define OFFLINE_GROUP 64
#endif
static const uint O_GROUP = OFFLINE_GROUP;
static const uint PAIR_WORDS = O_GROUP / 32;   // words of a collision-pair round mask (bit = lane)
static const uint O_MAX_VARS = 36;      // 6 + 30 dofs (groupshared memory, as below)
static const uint O_MAX_TARGETS = 18;   // groupshared memory: the rows below must fit in 48 KB
// Tracking-target rows are not stored: their Jacobian entries are cheap closed forms of the pose (see targetJ),
// evaluated where the normal equations need them.
static const uint O_MAX_ROWS = 3 * SOLE_POINTS + MAX_LINKS + O_MAX_PAIR_ROWS + DYN_ROWS + SMOOTH_ROWS;
static const float3 GRAVITY = float3(0.0, 0.0, -9.81);
static const float MOMENT_SCALE = 0.1;
static const float ZMP_MARGIN = 0.02;   // force-free mode: ZMP kept this far inside the support polygon

uniform StructuredBuffer<int> frameList;          // frames of the current colour (global frame index)
uniform StructuredBuffer<int2> frameRange;        // [frame] (first, end) frame of its clip
uniform RWStructuredBuffer<float4> trajRootPos;   // [frame] current iterate
uniform RWStructuredBuffer<float4> trajRootRot;   // [frame]
uniform RWStructuredBuffer<float> trajQ;          // [frame * numDofs]
uniform RWStructuredBuffer<float> outCostOffline; // [frame] tracking cost after the update

uniform StructuredBuffer<int> linkChainStart;     // [numLinks] into linkChainLinks
uniform StructuredBuffer<int> linkChainCount;
uniform StructuredBuffer<int> linkChainLinks;     // actuated ancestors (incl. self) of every link

uniform StructuredBuffer<int> soleLink;           // [2]
uniform StructuredBuffer<float4> solePoint;       // [8] link-frame sole points (foot k: 4k..4k+3)
uniform StructuredBuffer<float4> soleAnchor;      // [frame * 8] (x, y, vertical weight factor, weight); target
                                                  // height 0 (the ground); weight 0: not locked

uniform StructuredBuffer<float4> capsuleA;        // [numLinks] link frame, w = radius (0: no capsule)
uniform StructuredBuffer<float4> capsuleB;        // [numLinks]
uniform StructuredBuffer<float> medianRadius;     // [numLinks] radius of the median-radius capsule (validator's)

/// Required distance between the capsule axes of links a and b: the validator flags a link's surface points inside
/// the other link's median-radius capsule, so the axes must stay apart by the larger of (a's enclosing radius + b's
/// median radius) and the reverse. Surfaces may touch. (Two enclosing capsules apart kept a hand some 5 cm off the
/// boxy G1 torso, whose enclosing radius is 11.7 cm against 7.1 cm median, where the person's hands touch it.)
float requiredDistance(int a, int b)
{
    return max(capsuleA[a].w + medianRadius[b], medianRadius[a] + capsuleA[b].w);
}
uniform StructuredBuffer<float> linkGroundContact; // [frame * numLinks] equality weight (0: barrier only)
uniform StructuredBuffer<int2> collisionPairs;    // [numPairs]
uniform StructuredBuffer<int2> linkPointRange;    // [numLinks] (first, count) into linkPoints
uniform StructuredBuffer<float4> linkPoints;      // surface points, link frame
uniform StructuredBuffer<uint2> colTargets;       // [6 + numDofs] targets whose position (x) / rotation (y) rows are
                                                  // nonzero in the column (bit t; zero-weight parts excluded)
uniform StructuredBuffer<int> dofLink;            // [numDofs] link of each joint
uniform StructuredBuffer<uint> linkDepth;         // [numLinks] depth in the kinematic tree (root 0)
uniform uint maxDepth;
uniform StructuredBuffer<float4> linkBound;       // [numLinks] bounding sphere of the points (link frame, w = radius)

uniform StructuredBuffer<float4> linkCom;         // [numLinks] link-frame CoM, w = mass
uniform StructuredBuffer<uint2> linkAncestors;    // [numLinks] bit l set: link l is the link or an ancestor
uniform RWStructuredBuffer<float4> trajLinkCom;   // [frame * numLinks] world link CoMs of the current iterate
uniform StructuredBuffer<float4> contactPoint;    // [frame * CONTACT_POINTS] world, w = link + 1 (0: inactive)
uniform StructuredBuffer<float> frameFps;         // [frame]
uniform RWStructuredBuffer<float4> trajSole;      // [frame * 8] world sole points of the result (when writeFeatures)
uniform RWStructuredBuffer<float4> trajPatch;     // [frame * CONTACT_POINTS] ground contact patches (x, y, 0, link + 1)
                                                  // of the links in mapped contact, in link order; w = 0: none
uniform uint writeFeatures;                       // 1: write trajSole / trajPatch (the sweep before an anchor refresh)
uniform RWStructuredBuffer<float> contactLambda;  // [frame * CONTACT_POINTS * 4] friction-edge coefficients of the
                                                  // frame's contact-force solve, warm start for the next one

uniform uint numFrameList;
uniform uint numPairs;
uniform uint numHandPairs;
uniform StructuredBuffer<int2> handPairs;         // [numHandPairs] (left-hand link, right-hand link)
uniform StructuredBuffer<int4> linkSegment;       // [numLinks] limb links and pelvis: the human segment (two
                                                  // position targets) they stand for, -1: none; z: 1 arm, 0 body
uniform StructuredBuffer<int> armTargets;         // [6] shoulder, elbow, hand position targets, left then right (-1: none)
uniform uint sweepIterations;
uniform float velocityWeight;
uniform float accelerationWeight;
uniform float rootVelocityWeight;
uniform float groundWeight;
uniform float groundMargin;
uniform float collisionWeight;
uniform float collisionMargin;
uniform float collisionBand;     // smooth onset of the collision push over this distance before the margin
uniform float constraintScale;   // continuation: ramps the lock / ground / contact / collision weights up
uniform float rootAccelerationWeight;
// Fixed-lag windows: a prior pulling each frame to a reference pose (its realtime solve), so errors along directions
// nothing else holds (the arms' null space, the root under the dynamics term) cannot accumulate from frame to frame.
// Weights 0 (batch): off.
uniform float referenceWeight;       // joints (rad) and root rotation
uniform float referenceRootWeight;   // root position (m)
uniform StructuredBuffer<float4> referenceRootPos;   // [frame]
uniform StructuredBuffer<float4> referenceRootRot;   // [frame]
uniform StructuredBuffer<float> referenceQ;          // [frame * numDofs]
uniform float dynamicsWeight;
uniform float friction;
uniform float totalMass;
uniform uint contactIterations;
uniform uint dynamicsStride;    // K: dynamics accelerations use frames t ± K (frames are coloured mod 2K + 1)
uniform uint forceFree;         // 1: force-free CoM support terms instead of the contact-force solve
uniform int2 smoothLinks;       // links whose position acceleration is penalised (hands), -1: none
uniform float linkAccelerationWeight;
// Joint acceleration limits: a one-sided penalty on the second differences over frames s ± K (the validator's band)
// that exceed the effort-limited acceleration at the frame's warm-start pose, residual (|a| / limit - 1) per joint.
uniform float accelerationLimitWeight;           // 0: off
uniform StructuredBuffer<float> accelerationLimit;   // [frame * numDofs] rad/s² (already scaled by the margin)
uniform float maxJointStep;     // rad per step
uniform float maxRootStep;      // m per step

groupshared float oJ[O_MAX_ROWS * O_MAX_VARS];
groupshared float oR[O_MAX_ROWS];
groupshared float oTR[6 * O_MAX_TARGETS];   // weighted target residuals (position, then rotation)
static const uint HS = O_MAX_VARS + 1;        // oH row stride: odd, so a column read across lanes hits 32 banks
static const uint ROW_WORDS = (O_MAX_ROWS + 31) / 32;
groupshared float oH[O_MAX_VARS * HS];        // normal matrix, lower triangle (becomes the Cholesky factor)
groupshared uint oRowBits[O_MAX_VARS * ROW_WORDS];  // per column: the rows with a nonzero entry there
groupshared float oDiag[O_MAX_VARS];          // Cholesky diagonal
groupshared float oL[O_MAX_VARS];             // current Cholesky column
groupshared float oY[O_MAX_VARS];             // forward-substitution result

groupshared float oG[O_MAX_VARS];
groupshared float4 oRot[MAX_LINKS];
groupshared float3 oPos[MAX_LINKS];
groupshared float3 oAxis[MAX_LINKS];
groupshared float3 oOrigin[MAX_LINKS];
groupshared float oQ[MAX_DOFS];
groupshared float3 oRootPos;
groupshared float4 oRootRot;
groupshared uint oPairRows;
groupshared uint oPairMask[3 * PAIR_WORDS];   // active collision pairs of a round (3 rotating slots), bit = lane
groupshared uint oScan[MAX_LINKS];          // links whose lowest point is needed
groupshared float oMinZ[O_GROUP];
groupshared int oMinIndex[O_GROUP];
groupshared float3 dC[3 * MAX_LINKS];   // link CoMs of frames t-1, t, t+1
groupshared float3 dA[MAX_LINKS];       // link CoM accelerations of frame t, minus gravity
groupshared float3 dF[3];               // gravito-inertial force per slot
groupshared float3 dN[3];               // and moment about the slot's CoM
groupshared float3 dCom[3];             // CoM of the slot's frame (moment reference, held fixed)
groupshared float dRes[3 * 6];          // weighted residual wrench per slot
groupshared float3 dResForce[3];        // unweighted residual force per slot (units of m g)
// Each dynamics row r of slot k is linear in the slot's d F, d N and d CoM: J = cF . dF + cN . dN + cC . dc.
groupshared float3 dCoefF[3 * 6];
groupshared float3 dCoefN[3 * 6];
groupshared float3 dCoefC[3 * 6];
groupshared uint dValid[3];

/// Forward kinematics of the current pose, one tree level at a time (the links of a level in parallel).
void forwardOffline(uint lane)
{
    for (uint i = lane; i < numLinks; i += O_GROUP)
    {
        if (linkParent[i] < 0)
        {
            oRot[i] = oRootRot;
            oPos[i] = oRootPos;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint level = 1; level <= maxDepth; level++)
    {
        for (uint i = lane; i < numLinks; i += O_GROUP)
        {
            if (linkDepth[i] != level)
                continue;
            int parent = linkParent[i];
            float4 r = oRot[parent];
            float3 p = oPos[parent] + qrotate(r, jointOriginPos[i].xyz);
            r = qmul(r, jointOriginRot[i]);
            int dof = linkDof[i];
            if (dof >= 0)
            {
                float3 a = jointAxis[i].xyz;
                oAxis[i] = qrotate(r, a);
                oOrigin[i] = p;
                r = qmul(r, qaxisAngle(a, oQ[dof]));
            }
            oPos[i] = p + qrotate(r, jointToChildPos[i].xyz);
            oRot[i] = normalize(qmul(r, jointToChildRot[i]));
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

/// Marks `row` as nonzero in column `col` (rows are built by several lanes at once).
void markRow(uint row, uint col)
{
    // Row r is bit r / ROW_WORDS of word r % ROW_WORDS: rows built at the same time (by different lanes) mostly
    // land in different words, so the atomics rarely contend.
    InterlockedOr(oRowBits[col * ROW_WORDS + row % ROW_WORDS], 1u << (row / ROW_WORDS));
}

void clearRow(uint row, uint nv)
{
    for (uint c = 0; c < nv; c++)
        oJ[row * O_MAX_VARS + c] = 0.0;
    oR[row] = 0.0;
}

/// Tracking-target Jacobian columns (as the rows the solver used to store): position rows of target t in column a
/// (3 rows), rotation rows likewise; only for columns the target's rows reach (colTargets).
float3 targetPosJ(uint t, uint a)
{
    float w = targetWeight[t].x;
    float3 p = oPos[targetLink[t]];
    if (a < 3)
        return float3(a == 0 ? w : 0, a == 1 ? w : 0, a == 2 ? w : 0);
    if (a < 6)
        return w * cross(float3(a == 3 ? 1 : 0, a == 4 ? 1 : 0, a == 5 ? 1 : 0), p - oRootPos);
    int l = dofLink[a - 6];
    return w * cross(oAxis[l], p - oOrigin[l]);
}

float3 targetRotJ(uint t, uint a)
{
    float w = targetWeight[t].y;
    float4 twist = targetTwist[t];
    float3 axis = qrotate(oRot[targetLink[t]], twist.xyz);
    if (a < 6)
        return untwist(float3(a == 3 ? w : 0, a == 4 ? w : 0, a == 5 ? w : 0), axis, twist.w);
    return untwist(w * oAxis[dofLink[a - 6]], axis, twist.w);
}

/// Writes (or with `add`, adds) w * dir · dP/dx for the world point P fixed on `link` into a residual row, and
/// marks its columns. Only marked entries are ever read, so a row written once needs no clearing.
void pointRow(uint row, int link, float3 P, float3 dir, float w, bool add)
{
    float3 lever = cross(P - oRootPos, dir); // d/dw_k of dir·(w x d) = (d x dir)_k
    float root[6] = { w * dir.x, w * dir.y, w * dir.z, w * lever.x, w * lever.y, w * lever.z };
    for (uint c = 0; c < 6; c++)
    {
        oJ[row * O_MAX_VARS + c] = add ? oJ[row * O_MAX_VARS + c] + root[c] : root[c];
        markRow(row, c);
    }
    uint c0 = uint(linkChainStart[link]), cn = uint(linkChainCount[link]);
    for (uint c = 0; c < cn; c++)
    {
        int l = linkChainLinks[c0 + c];
        uint col = 6 + uint(linkDof[l]);
        float v = w * dot(dir, cross(oAxis[l], P - oOrigin[l]));
        oJ[row * O_MAX_VARS + col] = add ? oJ[row * O_MAX_VARS + col] + v : v;
        markRow(row, col);
    }
}

void setPointRow(uint row, int link, float3 P, float3 dir, float w) { pointRow(row, link, P, dir, w, false); }

/// Accumulating variant, for rows of two links (cleared first).
void addPointRow(uint row, int link, float3 P, float3 dir, float w) { pointRow(row, link, P, dir, w, true); }

float3 worldPoint(int link, float3 local) { return oPos[link] + qrotate(oRot[link], local); }

/// Smooth collision barrier on the gap g (m, negative inside the margin): zero for g >= band, a quadratic ramp
/// (value and slope continuous) over 0 <= g < band, linear inside. A plain hinge (r = w g for g < 0) switches its
/// full stiffness on at g = 0, and pairs hovering there (a hand swinging past the hip, hands near the torso)
/// were pushed on some frames and not on the next: visible 120 Hz shaking. Returns (residual, d residual / d g).
float2 collisionBarrier(float g, float band, float w)
{
    if (g >= band)
        return float2(0, 0);
    if (g >= 0.0)
        return float2(w * (band - g) * (band - g) / (2.0 * band), -w * (band - g) / band);
    return float2(w * (0.5 * band - g), -w);
}

float3 storedCom(uint frame, uint link) { return trajLinkCom[frame * numLinks + link].xyz; }

/// Closest points of two segments (Ericson): returns (s, t).
float2 segmentParameters(float3 p1, float3 q1, float3 p2, float3 q2)
{
    float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    float s = 0.0, t = 0.0;
    if (a <= 1e-12 && e <= 1e-12)
        return float2(0, 0);
    if (a <= 1e-12)
        return float2(0, clamp(f / e, 0.0, 1.0));
    float c = dot(d1, r);
    if (e <= 1e-12)
        return float2(clamp(-c / a, 0.0, 1.0), 0);
    float b = dot(d1, d2), denom = a * e - b * b;
    s = denom > 1e-12 ? clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
    t = (b * s + f) / e;
    if (t < 0.0)
    {
        t = 0.0;
        s = clamp(-c / a, 0.0, 1.0);
    }
    else if (t > 1.0)
    {
        t = 1.0;
        s = clamp((b - c) / a, 0.0, 1.0);
    }
    return float2(s, t);
}


/// Link CoMs of the slots s = t + (k-1) K (k = 0..2, K = dynamicsStride), frame t's accelerations, and each
/// valid slot's gravito-inertial wrench (a slot needs both neighbours s ± K inside the clip).
void dynamicsKinematics(uint frame, int2 range, uint lane)
{
    int f = int(frame), K = int(dynamicsStride);
    float rate = frameFps[frame] / float(K);
    float fps2 = rate * rate;
    bool hasM1 = f - K >= range.x, hasP1 = f + K < range.y;
    for (uint i = lane; i < numLinks; i += O_GROUP)
    {
        float3 live = oPos[i] + qrotate(oRot[i], linkCom[i].xyz);
        float3 cm1 = hasM1 ? storedCom(uint(f - K), i) : live, cp1 = hasP1 ? storedCom(uint(f + K), i) : live;
        dC[i] = cm1;
        dC[MAX_LINKS + i] = live;
        dC[2 * MAX_LINKS + i] = cp1;
        dA[i] = (cp1 - 2.0 * live + cm1) * fps2 - GRAVITY;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < 3)
    {
        int s = f + (int(lane) - 1) * K;
        bool valid = s - K >= range.x && s + K < range.y;
        dValid[lane] = valid ? 1 : 0;
        float3 com = float3(0, 0, 0);
        for (uint i = 0; i < numLinks; i++)
            com += linkCom[i].w * dC[lane * MAX_LINKS + i];
        com /= totalMass;
        dCom[lane] = com;
        float3 F = float3(0, 0, 0), N = float3(0, 0, 0);
        if (valid)
        {
            for (uint i = 0; i < numLinks; i++)
            {
                float3 a;
                if (lane == 1)
                    a = dA[i];
                else if (lane == 0) // s = t-K: c(t-2K), c(t-K), c(t)
                    a = (dC[MAX_LINKS + i] - 2.0 * dC[i] + storedCom(uint(f - 2 * K), i)) * fps2 - GRAVITY;
                else                // s = t+K: c(t), c(t+K), c(t+2K)
                    a = (storedCom(uint(f + 2 * K), i) - 2.0 * dC[2 * MAX_LINKS + i] + dC[MAX_LINKS + i]) * fps2 - GRAVITY;
                a *= linkCom[i].w;
                F += a;
                N += cross(dC[lane * MAX_LINKS + i] - com, a);
            }
        }
        dF[lane] = F;
        dN[lane] = N;
    }
    GroupMemoryBarrierWithGroupSync();
}

float3 frictionEdge(uint e)
{
    float2 t = e == 0 ? float2(1, 0) : (e == 1 ? float2(0, 1) : (e == 2 ? float2(-1, 0) : float2(0, -1)));
    return normalize(float3(friction * t, 1.0));
}

/// Residual wrench of slot k after the best non-negative contact forces (one lane per slot).
/// Signed distance of p to the convex hull of the slot's contact points (positive outside, negative inside:
/// minus the distance to the nearest edge) and the outward unit normal there (xy).
float3 supportDistance(uint s, float2 p)
{
    float2 pts[CONTACT_POINTS];
    uint count = 0;
    for (uint c = 0; c < CONTACT_POINTS; c++)
    {
        float4 q = contactPoint[s * CONTACT_POINTS + c];
        if (q.w > 0.0)
            pts[count++] = q.xy;
    }
    // Gift wrapping (counter-clockwise).
    uint hull[CONTACT_POINTS];
    uint h = 0;
    uint start = 0;
    for (uint i = 1; i < count; i++)
        if (pts[i].x < pts[start].x || (pts[i].x == pts[start].x && pts[i].y < pts[start].y))
            start = i;
    uint cur = start;
    for (uint guard = 0; guard < count; guard++)
    {
        hull[h++] = cur;
        uint next = (cur + 1) % count;
        for (uint i = 0; i < count; i++)
        {
            float2 a = pts[next] - pts[cur], b = pts[i] - pts[cur];
            float cr = a.x * b.y - a.y * b.x;
            if (cr < 0.0 || (cr == 0.0 && dot(b, b) > dot(a, a)))
                next = i;
        }
        cur = next;
        if (cur == start || h >= CONTACT_POINTS)
            break;
    }
    bool inside = h >= 3;
    float best = 1e9;
    float2 nearest = pts[hull[0]];
    for (uint e = 0; e < h; e++)
    {
        float2 a = pts[hull[e]], b = pts[hull[(e + 1) % h]];
        float2 ab = b - a;
        if (h >= 3 && ab.x * (p.y - a.y) - ab.y * (p.x - a.x) < 0.0)
            inside = false;
        float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
        float2 q = a + t * ab;
        float d = length(p - q);
        if (d < best)
        {
            best = d;
            nearest = q;
        }
    }
    if (inside)
    {
        // Nearest edge from the inside: outward normal of that edge (counter-clockwise hull: right of the edge).
        float bestEdge = 1e9;
        float2 normal = float2(0, 0);
        for (uint e = 0; e < h; e++)
        {
            float2 a = pts[hull[e]], b = pts[hull[(e + 1) % h]];
            float2 ab = b - a;
            float len = length(ab);
            if (len < 1e-6)
                continue;
            float2 nrm = float2(ab.y, -ab.x) / len;
            float d = -dot(p - a, nrm);
            if (d < bestEdge)
            {
                bestEdge = d;
                normal = nrm;
            }
        }
        return float3(-bestEdge, normal);
    }
    if (best < 1e-6)
        return float3(0, 0, 0);
    float2 n = (p - nearest) / best;
    return float3(best, n);
}

void dynamicsResidual(uint frame, uint k)
{
    float w = dynamicsWeight * constraintScale;
    for (uint r = 0; r < 6; r++)
    {
        dCoefF[k * 6 + r] = float3(0, 0, 0);
        dCoefN[k * 6 + r] = float3(0, 0, 0);
        dCoefC[k * 6 + r] = float3(0, 0, 0);
    }
    if (dValid[k] == 0 || w <= 0.0)
    {
        for (uint r = 0; r < 6; r++)
            dRes[k * 6 + r] = 0.0;
        dResForce[k] = float3(0, 0, 0);
        return;
    }
    uint s = uint(int(frame) + (int(k) - 1) * int(dynamicsStride));
    float mg = totalMass * 9.81;
    float3 com = dCom[k];
    // Wrench rows (contact-force solve, or flight in either mode): residual F / mg, N / (mg l).
    for (uint r = 0; r < 3; r++)
    {
        float3 e = float3(r == 0 ? 1 : 0, r == 1 ? 1 : 0, r == 2 ? 1 : 0);
        dCoefF[k * 6 + r] = w * e / mg;
        dCoefN[k * 6 + 3 + r] = w * e / (mg * MOMENT_SCALE);
    }
    bool anyContact = false;
    for (uint c = 0; c < CONTACT_POINTS; c++)
        anyContact = anyContact || contactPoint[s * CONTACT_POINTS + c].w > 0.0;
    if (forceFree != 0 && anyContact)
    {
        // Force-free CoM support: the ground cannot pull (F_z >= 0), the total horizontal force stays inside the
        // friction cone, and the ZMP of the motion stays inside the support polygon of the contact points. ZMP on z = 0 from the wrench about the CoM:
        //   p_x = c_x - (N_y + c_z F_x) / F_z,  p_y = c_y + (N_x - c_z F_y) / F_z.
        float3 F = dF[k], N = dN[k];
        for (uint r = 0; r < 6; r++)
        {
            dCoefF[k * 6 + r] = float3(0, 0, 0);
            dCoefN[k * 6 + r] = float3(0, 0, 0);
            dRes[k * 6 + r] = 0.0;
        }
        if (F.z < 0.0)
        {
            dRes[k * 6] = w * F.z / mg;
            dCoefF[k * 6] = float3(0, 0, w / mg);
        }
        // Friction on the total ground force: |F_xy| <= mu F_z (otherwise horizontal acceleration can put the
        // ZMP anywhere).
        float fh = length(F.xy);
        if (fh > friction * max(F.z, 0.0) && fh > 1e-6)
        {
            dRes[k * 6 + 2] = w * (fh - friction * max(F.z, 0.0)) / mg;
            dCoefF[k * 6 + 2] = w * float3(F.x / fh, F.y / fh, F.z > 0.0 ? -friction : 0.0) / mg;
        }
        // ZMP row only for frame t's own slot: from the neighbours' slots (t ± K, which frame t reaches only
        // through second-difference accelerations) it diverged.
        if (k == 1 && F.z > 0.3 * mg)
        {
            // Penalised as the moment the support cannot provide, d F_z (d: ZMP distance outside the polygon),
            // expressed as the horizontal force at the CoM that would supply it, d F_z / (mg c_z): linear in the
            // wrench for a fixed support edge (the same scale the contact-force solve effectively applies; the
            // stiffer d F_z / (mg l) diverged). n: outward normal (fixed);
            // d(d F_z) = n . (F_z dc - c_z dF_xy + dN_perp - F_xy dc_z) + (n . c - h) dF_z.
            float2 zmp = float2(com.x - (N.y + com.z * F.x) / F.z, com.y + (N.x - com.z * F.y) / F.z);
            float3 out = supportDistance(s, zmp);
            out.x += ZMP_MARGIN;   // active from ZMP_MARGIN inside the edge, so steps see the constraint coming
            if (out.x > 0.0)
            {
                float a = w / (mg * max(com.z, 0.2));
                float px = (N.y + com.z * F.x) / F.z, py = -(N.x - com.z * F.y) / F.z;
                float3 fx = float3(-com.z, 0, px), nx = float3(0, -1, 0), cx = float3(F.z, 0, -F.x);
                float3 fy = float3(0, -com.z, py), ny = float3(1, 0, 0), cy = float3(0, F.z, -F.y);
                dRes[k * 6 + 1] = a * out.x * F.z;
                dCoefF[k * 6 + 1] = a * (out.y * fx + out.z * fy + float3(0, 0, out.x));
                dCoefN[k * 6 + 1] = a * (out.y * nx + out.z * ny);
                dCoefC[k * 6 + 1] = a * (out.y * cx + out.z * cy);
            }
        }
        // N is about the slot's own CoM, which moves with frame t: d N includes -dc x F (applied to jN with
        // this force in buildDynamicsRows).
        dResForce[k] = F / mg;
        return;
    }
    // r = b - A lambda in units of m g (moment rows divided by l). Active contact points only, with their
    // friction-edge moment arms about the slot's CoM computed once; warm start from this frame's last solve
    // (frames updated together never share a slot frame s).
    float3 rf = dF[k] / mg, rn = dN[k] / (mg * MOMENT_SCALE);
    float3 edge[4] = { frictionEdge(0), frictionEdge(1), frictionEdge(2), frictionEdge(3) };
    uint active[CONTACT_POINTS];
    float3 arm[CONTACT_POINTS];
    float lambda[CONTACT_POINTS * 4];
    uint count = 0;
    for (uint c = 0; c < CONTACT_POINTS; c++)
    {
        float4 p = contactPoint[s * CONTACT_POINTS + c];
        if (p.w <= 0.0)
            continue;
        active[count] = c;
        arm[count] = (p.xyz - com) / MOMENT_SCALE;
        for (uint e = 0; e < 4; e++)
        {
            float l = contactLambda[(s * CONTACT_POINTS + c) * 4 + e];
            lambda[count * 4 + e] = l;
            if (l > 0.0)
            {
                rf -= l * edge[e];
                rn -= l * cross(arm[count], edge[e]);
            }
        }
        count++;
    }
    for (uint it = 0; it < contactIterations; it++)
    {
        for (uint a = 0; a < count; a++)
        {
            float3 lever = arm[a];
            [ForceUnroll]
            for (uint e = 0; e < 4; e++)
            {
                float3 d = edge[e];
                float3 m = cross(lever, d);
                float old = lambda[a * 4 + e];
                float next = max(0.0, old + (dot(d, rf) + dot(m, rn)) / (1.0 + dot(m, m) + 1e-6));
                float delta = next - old;
                rf -= delta * d;
                rn -= delta * m;
                lambda[a * 4 + e] = next;
            }
        }
    }
    for (uint c = 0; c < CONTACT_POINTS * 4; c++)
        contactLambda[s * CONTACT_POINTS * 4 + c] = 0.0;
    for (uint a = 0; a < count; a++)
        for (uint e = 0; e < 4; e++)
            contactLambda[(s * CONTACT_POINTS + active[a]) * 4 + e] = lambda[a * 4 + e];
    for (uint r = 0; r < 3; r++)
    {
        dRes[k * 6 + r] = w * rf[r];
        dRes[k * 6 + 3 + r] = w * rn[r];
    }
    dResForce[k] = rf;
}

bool isAncestor(uint ancestor, uint link)
{
    uint2 mask = linkAncestors[link];
    return ancestor < 32 ? ((mask.x >> ancestor) & 1) != 0 : ((mask.y >> (ancestor - 32)) & 1) != 0;
}

/// Dynamics rows (18 from `row0`): residuals and d residual / d (frame t variables), column-parallel.
void buildDynamicsRows(uint frame, uint lane, uint nv, uint row0)
{
    if (lane < 3)
        dynamicsResidual(frame, lane);
    GroupMemoryBarrierWithGroupSync();
    float fps = frameFps[frame] / float(dynamicsStride);
    float w = dynamicsWeight * constraintScale;
    float mg = totalMass * 9.81;
    for (uint col = lane; col < nv; col += O_GROUP)
    {
        float3 jF[3], jN[3];
        for (uint k = 0; k < 3; k++)
        {
            jF[k] = float3(0, 0, 0);
            jN[k] = float3(0, 0, 0);
        }
        int dofLink = -1;
        if (col >= 6)
        {
            for (uint l = 0; l < numLinks; l++)
                if (linkDof[l] == int(col) - 6)
                    dofLink = int(l);
        }
        for (uint i = 0; i < numLinks; i++)
        {
            float m = linkCom[i].w;
            if (m <= 0.0)
                continue;
            float3 ct = dC[MAX_LINKS + i];
            float3 J;
            if (col < 3)
                J = float3(col == 0 ? 1 : 0, col == 1 ? 1 : 0, col == 2 ? 1 : 0);
            else if (col < 6)
                J = cross(float3(col == 3 ? 1 : 0, col == 4 ? 1 : 0, col == 5 ? 1 : 0), ct - oRootPos);
            else
            {
                if (!isAncestor(uint(dofLink), i))
                    continue;
                J = cross(oAxis[dofLink], ct - oOrigin[dofLink]);
            }
            for (uint k = 0; k < 3; k++)
            {
                float coef = (k == 1 ? -2.0 : 1.0) * fps * fps;
                float3 dj = coef * J;
                jF[k] += m * dj;
                jN[k] += m * cross(dC[k * MAX_LINKS + i] - dCom[k], dj);
                if (k == 1)
                    jN[k] -= m * cross(dA[i], J);
            }
        }
        // Frame t's moment reference is its own CoM: moving it shifts the residual moment by -dc x F_residual
        // (force-free mode: the whole F, see dynamicsResidual).
        float3 jCom = jF[1] / (-2.0 * fps * fps * totalMass);
        jN[1] -= cross(jCom, dResForce[1] * mg);
        for (uint k = 0; k < 3; k++)
        {
            bool valid = dValid[k] != 0 && w > 0.0;
            float3 jC = k == 1 ? jCom : float3(0, 0, 0);   // only frame t's own CoM moves with its variables
            for (uint r = 0; r < 6; r++)
            {
                uint i = k * 6 + r;
                float v = dot(dCoefF[i], jF[k]) + dot(dCoefN[i], jN[k]) + dot(dCoefC[i], jC);
                oJ[(row0 + i) * O_MAX_VARS + col] = valid ? v : 0.0;
                if (valid)
                    markRow(row0 + i, col);
            }
        }
    }
    for (uint r = lane; r < DYN_ROWS; r += O_GROUP)
        oR[row0 + r] = dRes[r];
}

/// Link acceleration rows (SMOOTH_ROWS from `row0`): for the hands, the position second differences of frames
/// s = t-1, t, t+1 over their neighbours (stored link CoMs; frame t live), weighted. Joint-acceleration smoothing
/// barely sees a hand spike made of small changes in several joints; the collision terms create exactly that when
/// a hand has to get around another body part, and these rows make the solver plan the detour smoothly.
void buildSmoothRows(uint frame, int2 range, uint lane, uint nv, uint row0)
{
    for (uint r = lane; r < SMOOTH_ROWS; r += O_GROUP)
        oR[row0 + r] = 0.0;
    GroupMemoryBarrierWithGroupSync();
    if (lane >= SMOOTH_LINKS * 3)
        return;
    uint li = lane / 3, k = lane % 3;    // link slot, frame slot (s = t - 1 + k)
    int link = li == 0 ? smoothLinks.x : smoothLinks.y;
    int f = int(frame), sframe = f - 1 + int(k);
    if (link < 0 || sframe - 1 < range.x || sframe + 1 >= range.y || linkAccelerationWeight <= 0.0)
        return;
    float3 live = oPos[link] + qrotate(oRot[link], linkCom[link].xyz);
    float3 c[5];
    for (int d = -2; d <= 2; d++)
    {
        int g = f + d;
        c[d + 2] = d == 0 ? live : (g >= range.x && g < range.y ? storedCom(uint(g), uint(link)) : live);
    }
    float3 a = c[k + 2] - 2.0 * c[k + 1] + c[k];
    float coef = k == 1 ? -2.0 : 1.0;
    float w = linkAccelerationWeight;
    for (uint axis = 0; axis < 3; axis++)
    {
        uint row = row0 + li * 9 + k * 3 + axis;
        float3 dir = float3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
        setPointRow(row, link, live, dir, w * coef);
        oR[row] = w * a[axis];
    }
}

/// Self-collision pair p of `collisionPairs`: whether it needs a row (inside the band), with its closest points,
/// separation direction and barrier (residual, weight).
bool collisionPairRow(uint p, uint frame, out int2 pair, out float3 ca, out float3 cb, out float3 n, out float2 barrier)
{
    ca = float3(0, 0, 0);
    cb = float3(0, 0, 0);
    n = float3(0, 0, 0);
    barrier = float2(0, 0);
    pair = collisionPairs[p];
    float4 a0 = capsuleA[pair.x], a1 = capsuleB[pair.x], b0 = capsuleA[pair.y], b1 = capsuleB[pair.y];
    // Bounding spheres of the two capsules apart by more than the band (the required distance is at most
    // the sum of the enclosing radii): no row.
    float reach = 0.5 * (length(a1.xyz - a0.xyz) + length(b1.xyz - b0.xyz)) + a0.w + b0.w;
    float3 centreA = worldPoint(pair.x, 0.5 * (a0.xyz + a1.xyz)), centreB = worldPoint(pair.y, 0.5 * (b0.xyz + b1.xyz));
    if (length(centreA - centreB) - reach - collisionMargin >= collisionBand)
        return false;
    float3 pa0 = worldPoint(pair.x, a0.xyz), pa1 = worldPoint(pair.x, a1.xyz);
    float3 pb0 = worldPoint(pair.y, b0.xyz), pb1 = worldPoint(pair.y, b1.xyz);
    float2 st = segmentParameters(pa0, pa1, pb0, pb1);
    ca = lerp(pa0, pa1, st.x);
    cb = lerp(pb0, pb1, st.y);
    float d = length(ca - cb);
    float gap = d - (requiredDistance(pair.x, pair.y) + collisionMargin);
    // Separation direction: between the closest points, oriented from b's centre towards a's so it cannot
    // flip as the segments pass each other; at an actual crossing (d ~ 0) the centre-to-centre direction.
    float3 centres = 0.5 * (pa0 + pa1) - 0.5 * (pb0 + pb1);
    if (d > 1e-4)
    {
        n = (ca - cb) / d;
        if (dot(n, centres) < 0.0)
            n = -n;
    }
    else if (length(centres) > 1e-6)
        n = normalize(centres);
    else
        return false;
    // Between the arms: along the person's arms instead (closest points of the scaled human segments the links
    // stand for), so the robot keeps its arms apart on the same sides as the person's; with the robot's own
    // geometry a hand caught between the chest and the other forearm escaped below or above it, flipping sides.
    // An arm against the pelvis or a leg: the person's direction blended 2:1 with the robot's own (a hand
    // sliding down past the wider robot hip swung from its front to its side within a few frames; the person's
    // direction alone left the hand up to 1.4 cm inside the hip, where the robot's hip is in its way).
    int4 sa = linkSegment[pair.x], sb = linkSegment[pair.y];
    if (sa.x >= 0 && sb.x >= 0 && (sa.z == 1 || sb.z == 1))
    {
        uint base = frame * numTargets;
        float3 a0g = goalPos[base + uint(sa.x)].xyz, a1g = goalPos[base + uint(sa.y)].xyz;
        float3 b0g = goalPos[base + uint(sb.x)].xyz, b1g = goalPos[base + uint(sb.y)].xyz;
        float2 hs = segmentParameters(a0g, a1g, b0g, b1g);
        float3 h = lerp(a0g, a1g, hs.x) - lerp(b0g, b1g, hs.y);
        if (length(h) > 0.01)
        {
            h = normalize(h);
            if (sa.z == 1 && sb.z == 1)
                n = h;
            else if (dot(h, n) > -0.5)
                n = normalize(2.0 * h + n);
        }
    }
    barrier = collisionBarrier(gap, collisionBand, collisionWeight * constraintScale);
    return barrier.x > 0.0;
}

void buildRows(uint frame, uint lane, uint nv, bool features)
{
    uint rowSole = 0;
    uint rowGround = rowSole + 3 * SOLE_POINTS;
    uint rowPairs = rowGround + numLinks;

    // Tracking targets: weighted residuals only (6 per target).
    for (uint t = lane; t < numTargets; t += O_GROUP)
    {
        int link = targetLink[t];
        float2 w = targetWeight[t];
        float3 rp = w.x * (oPos[link] - goalPos[frame * numTargets + t].xyz);
        float4 twist = targetTwist[t];
        float3 rr = untwist(w.y * qlog(qmul(oRot[link], qconj(goalRot[frame * numTargets + t]))), qrotate(oRot[link], twist.xyz), twist.w);
        for (uint k = 0; k < 3; k++)
        {
            oTR[6 * t + k] = rp[k];
            oTR[6 * t + 3 + k] = rr[k];
        }
    }

    // Sole points: lock to the stance anchor, else keep above the ground.
    for (uint i = lane; i < SOLE_POINTS; i += O_GROUP)
    {
        int link = soleLink[i / 4];
        float3 P = worldPoint(link, solePoint[i].xyz);
        float4 anchor = soleAnchor[frame * SOLE_POINTS + i];
        anchor.w *= constraintScale;
        for (uint k = 0; k < 3; k++)
        {
            uint row = rowSole + 3 * i + k;
            oR[row] = 0.0;
            float3 dir = float3(k == 0 ? 1 : 0, k == 1 ? 1 : 0, k == 2 ? 1 : 0);
            if (anchor.w > 0.0)
            {
                // anchor.z holds the vertical weight factor (the target height is the ground, z = 0).
                float wk = k == 2 ? anchor.w * anchor.z : anchor.w;
                float target = k == 2 ? 0.0 : anchor[k];
                setPointRow(row, link, P, dir, wk);
                oR[row] = wk * (P[k] - target);
            }
            else if (k == 2 && P.z < groundMargin)
            {
                setPointRow(row, link, P, dir, groundWeight * constraintScale);
                oR[row] = groundWeight * constraintScale * (P.z - groundMargin);
            }
        }
    }

    // Lowest surface point of every link: ground contact equality or barrier. A link whose bounding sphere stays
    // above the margin and that is not in contact has no row; the others' points are scanned by all lanes.
    for (uint l = lane; l < numLinks; l += O_GROUP)
    {
        oR[rowGround + l] = 0.0;
        float4 bound = linkBound[l];
        bool clear = bound.w > 0.0 && worldPoint(int(l), bound.xyz).z - bound.w >= groundMargin;
        oScan[l] = linkPointRange[l].y > 0 && (!clear || linkGroundContact[frame * numLinks + l] > 0.0) ? 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint patchCount = 0;   // lane 0
    for (uint l = 0; l < numLinks; l++)
    {
        if (oScan[l] == 0)
            continue;
        int2 range = linkPointRange[l];
        // Height of a link-frame point: the third row of the link rotation.
        float4 r = oRot[l];
        float3 rowZ = float3(2.0 * (r.x * r.z - r.w * r.y), 2.0 * (r.y * r.z + r.w * r.x), 1.0 - 2.0 * (r.x * r.x + r.y * r.y));
        float best = 1e30;
        int bestIndex = 0;
        for (int k = int(lane); k < range.y; k += int(O_GROUP))
        {
            float z = dot(rowZ, linkPoints[range.x + k].xyz);
            if (z < best)
            {
                best = z;
                bestIndex = k;
            }
        }
        oMinZ[lane] = best;
        oMinIndex[lane] = bestIndex;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0)
        {
            for (uint k = 1; k < O_GROUP; k++)
            {
                if (oMinZ[k] < best || (oMinZ[k] == best && oMinIndex[k] < bestIndex))
                {
                    best = oMinZ[k];
                    bestIndex = oMinIndex[k];
                }
            }
            uint row = rowGround + l;
            float3 low = worldPoint(int(l), linkPoints[range.x + bestIndex].xyz);
            float z = low.z;
            float contact = linkGroundContact[frame * numLinks + l] * constraintScale;
            float wg = groundWeight * constraintScale;
            if (contact > 0.0)
            {
                // Pulled to the ground; below it at least at the ground weight (the contact weight ramps in and
                // out at the ends of a contact run, and a lightly weighted contact let a knee or hand sink in).
                float w = z < 0.0 ? max(contact, wg) : contact;
                setPointRow(row, int(l), low, float3(0, 0, 1), w);
                oR[row] = w * z;
            }
            else if (z < groundMargin)
            {
                setPointRow(row, int(l), low, float3(0, 0, 1), wg);
                oR[row] = wg * (z - groundMargin);
            }
            oMinZ[0] = z;
        }
        GroupMemoryBarrierWithGroupSync();
        if (!features || linkGroundContact[frame * numLinks + l] <= 0.0)
            continue;
        // Contact patch (as the validator's contact model): among the points within 1 cm of the lowest, the
        // extreme ones along +x, -x, +y, -y. Lane l scans direction l % 4, every 8th point.
        float limit = oMinZ[0] + 0.01;
        GroupMemoryBarrierWithGroupSync();
        float2 axis = (lane % 4) == 0 ? float2(1, 0) : ((lane % 4) == 1 ? float2(-1, 0) : ((lane % 4) == 2 ? float2(0, 1) : float2(0, -1)));
        float top = -1e30;
        int topIndex = -1;
        for (int k = int(lane / 4); k < range.y; k += int(O_GROUP / 4))
        {
            float3 p = worldPoint(int(l), linkPoints[range.x + k].xyz);
            float score = dot(axis, p.xy);
            if (p.z < limit && score > top)
            {
                top = score;
                topIndex = k;
            }
        }
        oMinZ[lane] = top;
        oMinIndex[lane] = topIndex;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0)
        {
            int chosen[4];
            for (uint d = 0; d < 4; d++)
            {
                float bestScore = -1e30;
                int bestPoint = -1;
                for (uint g = d; g < O_GROUP; g += 4)
                {
                    int index = oMinIndex[g];
                    if (index >= 0 && (oMinZ[g] > bestScore || (oMinZ[g] == bestScore && index < bestPoint)))
                    {
                        bestScore = oMinZ[g];
                        bestPoint = index;
                    }
                }
                chosen[d] = bestPoint;
                bool repeated = bestPoint < 0;
                for (uint e = 0; e < d; e++)
                    repeated = repeated || chosen[e] == bestPoint;
                if (!repeated && patchCount < CONTACT_POINTS)
                {
                    float3 p = worldPoint(int(l), linkPoints[range.x + bestPoint].xyz);
                    trajPatch[frame * CONTACT_POINTS + patchCount] = float4(p.xy, 0.0, float(l + 1));
                    patchCount++;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (features && lane == 0)
        for (uint k = patchCount; k < CONTACT_POINTS; k++)
            trajPatch[frame * CONTACT_POINTS + k] = float4(0, 0, 0, 0);
    if (features && lane < SOLE_POINTS)
        trajSole[frame * SOLE_POINTS + lane] = float4(worldPoint(soleLink[lane / 4], solePoint[lane].xyz), 0.0);

    // Self-collision: active capsule pairs claim rows in pair order (rows by an atomic counter came in thread
    // timing order: the sums of the normal equations, and which rows were dropped when they were full, changed from
    // run to run). Per round of one pair per lane, an active pair's row is the count of active pairs before it. The
    // active masks rotate over three slots, so one barrier per round suffices: the slot cleared in a round was last
    // read two rounds before.
    if (lane < PAIR_WORDS)
        oPairMask[lane] = 0;
    for (uint k = lane; k < O_MAX_PAIR_ROWS; k += O_GROUP)
        clearRow(rowPairs + k, nv);
    GroupMemoryBarrierWithGroupSync();
    uint pairRows = 0;
    for (uint first = 0, round = 0; first < numPairs; first += O_GROUP, round = (round + 1) % 3)
    {
        int2 pair = int2(0, 0);
        float3 ca = float3(0, 0, 0), cb = float3(0, 0, 0), n = float3(0, 0, 0);
        float2 barrier = float2(0, 0);
        bool active = false;
        if (first + lane < numPairs)
            active = collisionPairRow(first + lane, frame, pair, ca, cb, n, barrier);
        if (active)
            InterlockedOr(oPairMask[PAIR_WORDS * round + lane / 32], 1u << (lane % 32));
        if (lane < PAIR_WORDS)
            oPairMask[PAIR_WORDS * ((round + 1) % 3) + lane] = 0;
        GroupMemoryBarrierWithGroupSync();
        uint slot = pairRows, total = 0;
        for (uint w = 0; w < PAIR_WORDS; w++)
        {
            uint bits = oPairMask[PAIR_WORDS * round + w];
            total += countbits(bits);
            if (w < lane / 32)
                slot += countbits(bits);
            else if (w == lane / 32)
                slot += countbits(bits & ((1u << (lane % 32)) - 1u));
        }
        pairRows += total;
        if (active && slot < O_MAX_PAIR_ROWS)
        {
            uint row = rowPairs + slot;
            // d gap / dx = n . (dP_a - dP_b); the residual decreases as the gap opens.
            addPointRow(row, pair.x, ca, n, barrier.y);
            addPointRow(row, pair.y, cb, -n, barrier.y);
            oR[row] = barrier.x;
        }
    }
    if (lane == 0)
        oPairRows = pairRows;

    // Hand against hand: one row for all link pairs between the two hands (wrist-roll link and below). Its depth
    // is the deepest overlap among them, and it pushes along one direction, from the right hand's goal to the
    // left's (the person's hands, scaled): the robot separates its hands the way the person's are and switches only
    // when they do. As separate pairs, each with its own closest-point direction (or with the robot's own hand
    // centres), clasped and rubbing hands were pushed one way, then another, and jumped above / below each other.
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && numHandPairs > 0)
    {
        float worst = collisionBand;
        int2 worstPair = int2(-1, -1);
        float3 wa = float3(0, 0, 0), wb = float3(0, 0, 0);
        for (uint p = 0; p < numHandPairs; p++)
        {
            int2 pair = handPairs[p];
            float4 a0 = capsuleA[pair.x], a1 = capsuleB[pair.x], b0 = capsuleA[pair.y], b1 = capsuleB[pair.y];
            float3 pa0 = worldPoint(pair.x, a0.xyz), pa1 = worldPoint(pair.x, a1.xyz);
            float3 pb0 = worldPoint(pair.y, b0.xyz), pb1 = worldPoint(pair.y, b1.xyz);
            float2 st = segmentParameters(pa0, pa1, pb0, pb1);
            float3 ca = lerp(pa0, pa1, st.x), cb = lerp(pb0, pb1, st.y);
            float gap = length(ca - cb) - (requiredDistance(pair.x, pair.y) + collisionMargin);
            if (gap < worst)
            {
                worst = gap;
                worstPair = pair;
                wa = ca;
                wb = cb;
            }
        }
        // Direction: from the right hand's goal to the left's (the person's hands, scaled).
        float3 c = armTargets[2] >= 0 && armTargets[5] >= 0
            ? goalPos[frame * numTargets + uint(armTargets[2])].xyz - goalPos[frame * numTargets + uint(armTargets[5])].xyz
            : float3(0, 0, 0);
        if (worstPair.x >= 0 && length(c) > 1e-6 && oPairRows < O_MAX_PAIR_ROWS)
        {
            uint row = rowPairs + oPairRows;
            oPairRows += 1;
            float3 n = normalize(c);
            float2 barrier = collisionBarrier(worst, collisionBand, collisionWeight * constraintScale);
            addPointRow(row, worstPair.x, wa, n, barrier.y);
            addPointRow(row, worstPair.y, wb, -n, barrier.y);
            oR[row] = barrier.x;
        }
    }
}

/// One Gauss-Newton update of `frame` against its (fixed) neighbours; the group's lanes work together.
/// `finalRows`: evaluate the rows once more after the update, for the cost and the features (contact patches, sole
/// points) of the result; without it (fixed-lag windows: no features) only the link transforms are updated for the
/// stored link CoMs, and the cost is that of the pose before the update (the row pass is a fifth of a frame solve).
void solveFrame(uint frame, uint lane, bool finalRows)
{
    int2 range = frameRange[frame];
    uint n = numDofs;
    uint nv = 6 + n;
    uint rowDynamics = 3 * SOLE_POINTS + numLinks + O_MAX_PAIR_ROWS;
    uint rowSmooth = rowDynamics + DYN_ROWS;

    if (lane == 0)
    {
        oRootPos = trajRootPos[frame].xyz;
        oRootRot = trajRootRot[frame];
        for (uint j = 0; j < n; j++)
            oQ[j] = trajQ[frame * n + j];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it <= sweepIterations; it++)
    {
        forwardOffline(lane);
        if (it == sweepIterations && !finalRows)
            break;
        for (uint e = lane; e < nv * ROW_WORDS; e += O_GROUP)
            oRowBits[e] = 0;
        GroupMemoryBarrierWithGroupSync();
        buildRows(frame, lane, nv, it == sweepIterations && writeFeatures != 0);
        GroupMemoryBarrierWithGroupSync();
        if (it == sweepIterations)
            break;
        if (dynamicsWeight > 0.0)   // weight 0 (fixed-lag windows): no CoM kinematics, no contact-force solve
        {
            dynamicsKinematics(frame, range, lane);
            buildDynamicsRows(frame, lane, nv, rowDynamics);
            GroupMemoryBarrierWithGroupSync();
        }
        buildSmoothRows(frame, range, lane, nv, rowSmooth);
        GroupMemoryBarrierWithGroupSync();

        // H = J^T J (lower triangle), g = J^T r, each entry over the rows nonzero in both columns.
        for (uint p = lane; p < nv * (nv + 1) / 2; p += O_GROUP)
        {
            uint b = uint((sqrt(float(8 * p + 1)) - 1.0) * 0.5);
            if ((b + 1) * (b + 2) / 2 <= p)
                b++;
            else if (b * (b + 1) / 2 > p)
                b--;
            uint a = p - b * (b + 1) / 2;
            float sum = 0.0;
            for (uint word = 0; word < ROW_WORDS; word++)
            {
                uint m = oRowBits[a * ROW_WORDS + word] & oRowBits[b * ROW_WORDS + word];
                while (m != 0)
                {
                    uint r = firstbitlow(m) * ROW_WORDS + word;
                    m &= m - 1;
                    sum += oJ[r * O_MAX_VARS + a] * oJ[r * O_MAX_VARS + b];
                }
            }
            uint2 both = colTargets[a] & colTargets[b];
            for (uint m = both.x; m != 0; m &= m - 1)
            {
                uint t = firstbitlow(m);
                sum += dot(targetPosJ(t, a), targetPosJ(t, b));
            }
            for (uint m = both.y; m != 0; m &= m - 1)
            {
                uint t = firstbitlow(m);
                sum += dot(targetRotJ(t, a), targetRotJ(t, b));
            }
            oH[b * HS + a] = sum;
        }
        for (uint a = lane; a < nv; a += O_GROUP)
        {
            float sum = 0.0;
            for (uint word = 0; word < ROW_WORDS; word++)
            {
                uint m = oRowBits[a * ROW_WORDS + word];
                while (m != 0)
                {
                    uint r = firstbitlow(m) * ROW_WORDS + word;
                    m &= m - 1;
                    sum += oJ[r * O_MAX_VARS + a] * oR[r];
                }
            }
            uint2 targets = colTargets[a];
            for (uint m = targets.x; m != 0; m &= m - 1)
            {
                uint t = firstbitlow(m);
                sum += dot(targetPosJ(t, a), float3(oTR[6 * t], oTR[6 * t + 1], oTR[6 * t + 2]));
            }
            for (uint m = targets.y; m != 0; m &= m - 1)
            {
                uint t = firstbitlow(m);
                sum += dot(targetRotJ(t, a), float3(oTR[6 * t + 3], oTR[6 * t + 4], oTR[6 * t + 5]));
            }
            oG[a] = sum;
        }
        GroupMemoryBarrierWithGroupSync();

        // Temporal coupling with the (fixed) neighbours, posture and limits — diagonal terms.
        int f = int(frame);
        bool m1 = f - 1 >= range.x, p1 = f + 1 < range.y, m2 = f - 2 >= range.x, p2 = f + 2 < range.y;
        float wv = velocityWeight * velocityWeight, wa = accelerationWeight * accelerationWeight;
        for (uint j = lane; j < n; j += O_GROUP)
        {
            uint v = 6 + j;
            float q = oQ[j];
            float qm1 = m1 ? trajQ[uint(f - 1) * n + j] : q, qp1 = p1 ? trajQ[uint(f + 1) * n + j] : q;
            float qm2 = m2 ? trajQ[uint(f - 2) * n + j] : qm1, qp2 = p2 ? trajQ[uint(f + 2) * n + j] : qp1;
            float h = 0.0, gv = 0.0;
            if (m1) { h += wv; gv += wv * (q - qm1); }
            if (p1) { h += wv; gv += wv * (q - qp1); }
            if (m1 && m2) { h += wa; gv += wa * (q - 2.0 * qm1 + qm2); }
            if (m1 && p1) { h += 4.0 * wa; gv += -2.0 * wa * (qp1 - 2.0 * q + qm1); }
            if (p1 && p2) { h += wa; gv += wa * (qp2 - 2.0 * qp1 + q); }
            if (accelerationLimitWeight > 0.0)
            {
                // Second differences centred at c = t - K, t, t + K (frame t's coefficient 1, -2, 1).
                int K = int(dynamicsStride);
                float dt2 = float(K * K) / (frameFps[frame] * frameFps[frame]);
                for (int k = -1; k <= 1; k++)
                {
                    int c = f + k * K;
                    if (c - K < range.x || c + K >= range.y)
                        continue;
                    float qa = c - K == f ? q : trajQ[uint(c - K) * n + j];
                    float qc = c == f ? q : trajQ[uint(c) * n + j];
                    float qb = c + K == f ? q : trajQ[uint(c + K) * n + j];
                    float d = qb - 2.0 * qc + qa, dmax = accelerationLimit[uint(c) * n + j] * dt2;
                    if (abs(d) <= dmax)
                        continue;
                    float u = accelerationLimitWeight / dmax, coef = k == 0 ? -2.0 : 1.0;
                    h += u * u * coef * coef;
                    gv += u * u * coef * sign(d) * (abs(d) - dmax);
                }
            }
            float wp = posture * posture;
            h += wp;
            gv += wp * q;
            if (referenceWeight > 0.0)
            {
                float wq = referenceWeight * referenceWeight;
                h += wq;
                gv += wq * (q - referenceQ[frame * n + j]);
            }
            float2 lim = dofLimit[j];
            float mid = 0.5 * (lim.x + lim.y), half = 0.5 * (lim.y - lim.x) * limitMargin;
            float excess = abs(q - mid) - half;
            if (excess > 0.0)
            {
                float wl = limitWeight * limitWeight;
                h += wl;
                gv += wl * excess * sign(q - mid);
            }
            oH[v * HS + v] += h;
            oG[v] += gv;
        }
        if (lane == 0)
        {
            // Root acceleration: second differences of position and of rotation (via logs of relative
            // rotations), same stencil as the joints.
            float wra = rootAccelerationWeight * rootAccelerationWeight;
            float3 pm1 = m1 ? trajRootPos[uint(f - 1)].xyz : oRootPos, pp1 = p1 ? trajRootPos[uint(f + 1)].xyz : oRootPos;
            float3 pm2 = m2 ? trajRootPos[uint(f - 2)].xyz : pm1, pp2 = p2 ? trajRootPos[uint(f + 2)].xyz : pp1;
            float4 rm1 = m1 ? trajRootRot[uint(f - 1)] : oRootRot, rp1 = p1 ? trajRootRot[uint(f + 1)] : oRootRot;
            float4 rm2 = m2 ? trajRootRot[uint(f - 2)] : rm1, rp2 = p2 ? trajRootRot[uint(f + 2)] : rp1;
            float3 eA = qlog(qmul(rm1, qconj(rm2))), eB = qlog(qmul(oRootRot, qconj(rm1)));
            float3 eC = qlog(qmul(rp1, qconj(oRootRot))), eD = qlog(qmul(rp2, qconj(rp1)));
            float hr = 0.0;
            float3 gp = float3(0, 0, 0), gr = float3(0, 0, 0);
            if (m1 && m2) { hr += wra; gp += wra * (oRootPos - 2.0 * pm1 + pm2); gr += wra * (eB - eA); }
            if (m1 && p1) { hr += 4.0 * wra; gp += -2.0 * wra * (pp1 - 2.0 * oRootPos + pm1); gr += -2.0 * wra * (eC - eB); }
            if (p1 && p2) { hr += wra; gp += wra * (pp2 - 2.0 * pp1 + oRootPos); gr += wra * (eD - eC); }
            for (uint k = 0; k < 3; k++)
            {
                oH[k * HS + k] += hr;
                oG[k] += gp[k];
                oH[(3 + k) * HS + 3 + k] += hr;
                oG[3 + k] += gr[k];
            }
            if (referenceWeight > 0.0 || referenceRootWeight > 0.0)
            {
                float wp = referenceRootWeight * referenceRootWeight, wq = referenceWeight * referenceWeight;
                float3 dp = oRootPos - referenceRootPos[frame].xyz;
                float3 dr = qlog(qmul(oRootRot, qconj(referenceRootRot[frame])));
                for (uint k = 0; k < 3; k++)
                {
                    oH[k * HS + k] += wp;
                    oG[k] += wp * dp[k];
                    oH[(3 + k) * HS + 3 + k] += wq;
                    oG[3 + k] += wq * dr[k];
                }
            }
            float wr = rootVelocityWeight * rootVelocityWeight;
            if (m1)
            {
                float3 dp = oRootPos - trajRootPos[uint(f - 1)].xyz;
                float3 dr = qlog(qmul(oRootRot, qconj(trajRootRot[uint(f - 1)])));
                for (uint k = 0; k < 3; k++)
                {
                    oH[k * HS + k] += wr;
                    oG[k] += wr * dp[k];
                    oH[(3 + k) * HS + 3 + k] += wr;
                    oG[3 + k] += wr * dr[k];
                }
            }
            if (p1)
            {
                float3 dp = oRootPos - trajRootPos[uint(f + 1)].xyz;
                float3 dr = qlog(qmul(oRootRot, qconj(trajRootRot[uint(f + 1)])));
                for (uint k = 0; k < 3; k++)
                {
                    oH[k * HS + k] += wr;
                    oG[k] += wr * dp[k];
                    oH[(3 + k) * HS + 3 + k] += wr;
                    oG[3 + k] += wr * dr[k];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint v = lane; v < nv; v += O_GROUP)
            oH[v * HS + v] += damping * (1.0 + oH[v * HS + v]);
        GroupMemoryBarrierWithGroupSync();

        // Right-looking Cholesky (lower triangle of oH becomes L, diagonal in oDiag), fused with the forward
        // substitution L y = g: step j scales column j and updates g below j, then the trailing entries, the
        // (i, k <= i) pairs spread over the lanes.
        for (uint j = 0; j < nv; j++)
        {
            float d = sqrt(max(oH[j * HS + j], 1e-12));
            float y = oG[j] / d;
            for (uint i = j + 1 + lane; i < nv; i += O_GROUP)
            {
                float l = oH[i * HS + j] / d;
                oH[i * HS + j] = l;
                oL[i] = l;
                oG[i] -= l * y;
            }
            if (lane == 0)
            {
                oDiag[j] = d;
                oY[j] = y;
            }
            GroupMemoryBarrierWithGroupSync();
            uint m = nv - j - 1;
            for (uint p = lane; p < m * (m + 1) / 2; p += O_GROUP)
            {
                uint e = uint((sqrt(float(8 * p + 1)) - 1.0) * 0.5);
                if ((e + 1) * (e + 2) / 2 <= p)
                    e++;
                else if (e * (e + 1) / 2 > p)
                    e--;
                uint i = j + 1 + e, k = j + 1 + p - e * (e + 1) / 2;
                oH[i * HS + k] -= oL[i] * oL[k];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        // Backward substitution L^T x = y; the step -x (H x = -g) goes to oG.
        for (int i = int(nv) - 1; i >= 0; i--)
        {
            uint u = uint(i);
            float x = oY[u] / oDiag[u];
            for (uint k = lane; k < u; k += O_GROUP)
                oY[k] -= oH[u * HS + k] * x;
            if (lane == 0)
                oG[u] = -x;
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0)
        {
            // Trust region: one Gauss-Newton step per sweep on a stiff, non-convex problem; scale the step so
            // no joint moves more than maxJointStep and the root no more than maxRootStep.
            float big = 0.0;
            for (uint j = 0; j < n; j++)
                big = max(big, abs(oG[6 + j]) / maxJointStep);
            big = max(big, length(float3(oG[0], oG[1], oG[2])) / maxRootStep);
            big = max(big, length(float3(oG[3], oG[4], oG[5])) / maxJointStep);
            float scale = big > 1.0 ? 1.0 / big : 1.0;
            oRootPos += scale * float3(oG[0], oG[1], oG[2]);
            oRootRot = normalize(qmul(qexp(scale * float3(oG[3], oG[4], oG[5])), oRootRot));
            for (uint j = 0; j < n; j++)
            {
                float2 lim = dofLimit[j];
                oQ[j] = clamp(oQ[j] + scale * oG[6 + j], lim.x, lim.y);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0)
    {
        float cost = 0.0;
        for (uint r = 0; r < 6 * numTargets; r++)
            cost += oTR[r] * oTR[r];
        trajRootPos[frame] = float4(oRootPos, 0.0);
        trajRootRot[frame] = oRootRot;
        for (uint j = 0; j < n; j++)
            trajQ[frame * n + j] = oQ[j];
        outCostOffline[frame] = cost;
    }
    for (uint i = lane; i < numLinks; i += O_GROUP)
        trajLinkCom[frame * numLinks + i] = float4(oPos[i] + qrotate(oRot[i], linkCom[i].xyz), 0.0);
}

[shader("compute")]
[numthreads(OFFLINE_GROUP, 1, 1)]
void sweep(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID)
{
    if (gid.x >= numFrameList)
        return;
    solveFrame(uint(frameList[gid.x]), gtid.x, true);
}

// Fixed-lag windows (FixedLagSolver): one group per stream runs `windowSweeps` symmetric Gauss-Seidel sweeps over
// its frames [first, end) — forward, then backward — in a single dispatch. Frame by frame dispatches spent ~1.3 ms
// of launch and scheduling per frame solve (~0.3 ms of work), which made a window tick ~100 ms.
uniform StructuredBuffer<int2> windowFrames;    // [numWindows] (first, end) frames solved, global indices
uniform uint numWindows;
uniform uint windowSweeps;

[shader("compute")]
[numthreads(OFFLINE_GROUP, 1, 1)]
void sweepWindow(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID)
{
    if (gid.x >= numWindows)
        return;
    int2 w = windowFrames[gid.x];
    uint count = uint(w.y - w.x);
    for (uint s = 0; s < windowSweeps; s++)
    {
        for (uint i = 0; i < count; i++)
        {
            uint frame = uint(s % 2 == 0 ? w.x + int(i) : w.y - 1 - int(i));
            solveFrame(frame, gtid.x, false);
            // The next frame reads this one's pose and link CoMs from device memory.
            AllMemoryBarrierWithGroupSync();
        }
    }
}
