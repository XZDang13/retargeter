// Retargeting IK, cooperative variant: one 32-thread group per stream (GPU). Same maths as the serial
// variant in retarget_ik.slang (see retarget_ik_common.slang), plus a causal centroidal-dynamics term that
// only this variant has:
//
// With point-mass links, the gravito-inertial wrench of the current frame t about its CoM,
//   F = sum m_i (a_i - g),  N = sum m_i (c_i - c) x (a_i - g),  a_i = (c_i,t - 2 c_i,t-K + c_i,t-2K) rate^2,
// (backward differences over the stream's previous frames, rate = fps / K) has to be supplied by the ground at
// the frame's contact points (from the causal human contact tracker: sole corners of feet in contact, the
// ground projection of other mapped links in contact) through friction pyramids. The best non-negative
// contact forces are solved in the kernel; the residual wrench (F / mg, N / (mg l)) is penalised with the
// forces held fixed (as in offline_ik_group.slang). Link CoMs of past frames live in a per-stream ring.
import retarget_ik_common;

static const uint RT_CONTACT_POINTS = 16;
static const uint HISTORY = 16;          // ring of past link CoMs per stream (>= 2K)
static const uint RT_ROWS = MAX_ROWS + 6;
static const float3 RT_GRAVITY = float3(0.0, 0.0, -9.81);
static const float RT_MOMENT_SCALE = 0.1;

// Global parameters, one parameter block: on Metal an argument buffer (a stage has only 31 buffer slots, the
// solvers bind more); elsewhere a plain uniform struct. The host binds fields by name (rtcompute looks inside).
struct RealtimeParams
{
#ifdef __METAL__
    RWStructuredBuffer<float> jacobianScratch;   // [stream * JACOBIAN_SIZE] (see GJ)
#endif
    StructuredBuffer<float4> linkCom;          // [numLinks] link-frame CoM, w = mass
    StructuredBuffer<uint2> linkAncestors;     // [numLinks] bit l: link l is the link or an ancestor
    StructuredBuffer<int> soleLink;            // [2]
    StructuredBuffer<float4> solePoint;        // [8] link frame
    StructuredBuffer<float> soleContact;       // [frame * 8] 1: the human heel / toe under this corner touches
    StructuredBuffer<float> linkContactRT;     // [frame * numLinks] 1: mapped body part in contact (not feet)
    StructuredBuffer<float> streamFps;         // [stream]
    StructuredBuffer<uint> factorStructure;    // see setupShared
    RWStructuredBuffer<float4> stateLinkCom;   // [(stream * HISTORY + slot) * numLinks + link]
    RWStructuredBuffer<int> stateHistory;      // [stream] frames pushed into the ring
    StructuredBuffer<int> dofArm;              // [numDofs] 0, 1 left arm, 2 right arm
    StructuredBuffer<int> targetArm;           // [numTargets] the arm a target belongs to (0: none)
    RWStructuredBuffer<float> stateBranchQ;    // [stream * numDofs] the branch being blended in (arm dofs)
    RWStructuredBuffer<int> stateBranch;       // [stream * 16] per arm: active, blend position, idle, direction, evidence
    uint branchInterval;
    float branchError;
    uint branchIterations;
    uint branchBlend;
    float branchHysteresis;
    uint branchKeep;
    uint branchDwell;
    float dynamicsWeight;
    uint dynamicsStride;
    float friction;
    float totalMass;
    uint contactIterations;
};
#ifdef __METAL__
ParameterBlock<RealtimeParams> rt;
#else
uniform RealtimeParams rt;
#endif
// Arm branch switch: a warm-started arm that has run into a joint-limit corner (jump_right_003__A030: the arm raised
// sideways reached through shoulder abduction, roll and yaw at their stops, 14 cm off for two seconds) cannot leave
// it while the target moves on, and eventually snaps across in a frame or two. Every `branchInterval` frames an arm
// whose elbow or hand is more than `branchError` off is solved again from its joints at 0 (`branchIterations`,
// no smoothness pull). If that more than halves the arm's tracking cost, the new branch is tracked alongside the
// current one (warm, its own smoothness pull) and the output arm eases between them with a smoothstep over
// `branchBlend` frames, toward whichever tracks better (see branchSwitch). Switching in one frame moved joints up
// to 3 rad; dropping the old branch at once switched crowd_cheer_001__A029 into one that went bad a second later.
// branchBlend 0: off.

#ifdef __METAL__
// Metal has 32 KB of threadgroup memory, too little for all of the below: the dense Jacobian (22.8 KB) lives in a
// per-group slice of device scratch there, and the group barriers also order device memory (groupSync).
groupshared uint gJBase;
#define GJ(i) rt.jacobianScratch[gJBase + (i)]
#else
groupshared float gJ[RT_ROWS * MAX_VARS];   // dense weighted Jacobian rows (6 per target, 6 dynamics)
#define GJ(i) gJ[i]
#endif
static const uint JACOBIAN_SIZE = RT_ROWS * MAX_VARS;   // floats of jacobianScratch per stream (Metal)

void groupSync()
{
#ifdef __METAL__
    AllMemoryBarrierWithGroupSync();
#else
    GroupMemoryBarrierWithGroupSync();
#endif
}
groupshared float gR[RT_ROWS];              // weighted residuals
static const uint HS = MAX_VARS + 1;        // gH row stride: odd, so a column read across lanes hits 32 banks
groupshared float gH[MAX_VARS * HS];        // normal matrix, lower triangle (becomes the Cholesky factor)
groupshared float gG[MAX_VARS];
groupshared float gDiag[MAX_VARS];          // Cholesky diagonal
groupshared float gL[MAX_VARS];             // current Cholesky column
groupshared float gCol[MAX_VARS];           // forward-substitution result
// Per column: the residual row triples (3 rows each: a target's position, its rotation, then the two dynamics
// triples) with a nonzero Jacobian entry there; the normal equations only sum over triples both columns share.
groupshared uint2 gColMask[MAX_VARS];
groupshared uint gPairs[MAX_VARS * (MAX_VARS + 1) / 2];  // (a, b) column pairs, b >= a, sharing a triple
groupshared uint gPairCount;
// Sparse Cholesky: variables in minimum-degree elimination order (limbs first, root last, so little fill-in);
// the normal equations are stored in that order. Per factor column j, its nonzero rows below j (gLCol) and
// per row i, its nonzero columns left of i (gLRow), both ascending, from gLColStart / gLRowStart.
static const uint MAX_FACTOR = MAX_VARS * (MAX_VARS - 1) / 2;
groupshared uint gInv[MAX_VARS];            // variable -> elimination position
groupshared uint gLColStart[MAX_VARS + 1];
groupshared uint gLRowStart[MAX_VARS + 1];
groupshared uint8_t gLCol[MAX_FACTOR];
groupshared uint8_t gLRow[MAX_FACTOR];
// Model, cached once per dispatch; links by tree depth for the level-parallel forward kinematics.
groupshared float4 gJointOriginRot[MAX_LINKS];
groupshared float3 gJointOriginPos[MAX_LINKS];
groupshared float4 gJointToChildRot[MAX_LINKS];
groupshared float3 gJointToChildPos[MAX_LINKS];
groupshared float3 gJointAxis[MAX_LINKS];
groupshared int gLinkParent[MAX_LINKS];
groupshared int gLinkDof[MAX_LINKS];
groupshared uint gDepth[MAX_LINKS];
groupshared float3 gAxisLocal[MAX_LINKS];  // joint axis in the parent link frame (joint origin rotation applied)
groupshared float4 gLocalRot[MAX_LINKS];   // link relative to its parent, current pose
groupshared float3 gLocalPos[MAX_LINKS];
groupshared uint gMaxDepth;
groupshared float4 gRot[MAX_LINKS];
groupshared float3 gPos[MAX_LINKS];
groupshared float3 gAxis[MAX_LINKS];
groupshared float3 gOrigin[MAX_LINKS];
groupshared float gQ[MAX_DOFS];
groupshared float gQPrev[MAX_DOFS];
groupshared float gQMain[MAX_DOFS];         // branch switch: the current branch while another one is solved
groupshared float gQMainPrev[MAX_DOFS];
groupshared float gQBranch[MAX_DOFS];
groupshared float3 gRootPosMain;
groupshared float4 gRootRotMain;
groupshared int gBranch[16];
groupshared float2 gArmMain[2];
groupshared float gArmCost, gArmError;
groupshared uint gKeep;
groupshared float3 gRootPos;
groupshared float4 gRootRot;
groupshared float3 gC[MAX_LINKS];           // link CoMs of the current pose
groupshared float3 gA[MAX_LINKS];           // link CoM accelerations minus gravity
groupshared float3 gCom;
groupshared float3 gResForce;               // unweighted residual force (units of m g)
groupshared float gDyn[6];

/// Forward kinematics of the current pose: every link's joint-local transform in parallel, then composed down
/// the tree one level at a time (the links of a level in parallel).
void forwardShared(uint lane)
{
    for (uint i = lane; i < ik.numLinks; i += GROUP)
    {
        int dof = gLinkDof[i];
        float4 r = gJointOriginRot[i];
        if (dof >= 0)
            r = qmul(r, qaxisAngle(gJointAxis[i], gQ[dof]));
        gLocalPos[i] = gJointOriginPos[i] + qrotate(r, gJointToChildPos[i]);
        gLocalRot[i] = qmul(r, gJointToChildRot[i]);
        if (gLinkParent[i] < 0)
        {
            gRot[i] = gRootRot;
            gPos[i] = gRootPos;
        }
    }
    groupSync();
    for (uint level = 1; level <= gMaxDepth; level++)
    {
        for (uint i = lane; i < ik.numLinks; i += GROUP)
        {
            if (gDepth[i] != level)
                continue;
            int parent = gLinkParent[i];
            float4 r = gRot[parent];
            float3 p = gPos[parent];
            if (gLinkDof[i] >= 0)
            {
                gAxis[i] = qrotate(r, gAxisLocal[i]);
                gOrigin[i] = p + qrotate(r, gJointOriginPos[i]);
            }
            gPos[i] = p + qrotate(r, gLocalPos[i]);
            gRot[i] = normalize(qmul(r, gLocalRot[i]));
        }
        groupSync();
    }
}

/// Once per dispatch: model cache, link depths, column triple masks and the column pairs to sum.
void setupShared(uint lane, uint nv)
{
    for (uint i = lane; i < ik.numLinks; i += GROUP)
    {
        gJointOriginRot[i] = ik.jointOriginRot[i];
        gJointOriginPos[i] = ik.jointOriginPos[i].xyz;
        gJointToChildRot[i] = ik.jointToChildRot[i];
        gJointToChildPos[i] = ik.jointToChildPos[i].xyz;
        gJointAxis[i] = ik.jointAxis[i].xyz;
        gAxisLocal[i] = qrotate(ik.jointOriginRot[i], ik.jointAxis[i].xyz);
        gLinkParent[i] = ik.linkParent[i];
        gLinkDof[i] = ik.linkDof[i];
        uint depth = 0;
        for (int k = ik.linkParent[i]; k >= 0; k = ik.linkParent[k])
            depth++;
        gDepth[i] = depth;
    }
    // Sparsity structure from the host (RetargetSolver.factorStructure: column masks, the column pairs that share
    // a row triple heaviest first, the minimum-degree elimination order and the factor's row / column lists). It
    // depends only on the model and the weights; derived here on one lane it cost ~3 ms per dispatch, which
    // dominated one-frame (teleop) dispatches.
    uint pairCount = rt.factorStructure[0], offPairs = rt.factorStructure[1], offMask = rt.factorStructure[2];
    uint offInv = rt.factorStructure[3], offColStart = rt.factorStructure[4], offCol = rt.factorStructure[5];
    uint offRowStart = rt.factorStructure[6], offRow = rt.factorStructure[7], factorCount = rt.factorStructure[8];
    for (uint k = lane; k < pairCount; k += GROUP)
        gPairs[k] = rt.factorStructure[offPairs + k];
    for (uint a = lane; a < nv; a += GROUP)
    {
        gColMask[a] = uint2(rt.factorStructure[offMask + 2 * a], rt.factorStructure[offMask + 2 * a + 1]);
        gInv[a] = rt.factorStructure[offInv + a];
    }
    for (uint j = lane; j <= nv; j += GROUP)
    {
        gLColStart[j] = rt.factorStructure[offColStart + j];
        gLRowStart[j] = rt.factorStructure[offRowStart + j];
    }
    for (uint e = lane; e < factorCount; e += GROUP)
    {
        gLCol[e] = uint8_t(rt.factorStructure[offCol + e]);
        gLRow[e] = uint8_t(rt.factorStructure[offRow + e]);
    }
    groupSync();
    if (lane == 0)
    {
        uint depth = 0;
        for (uint i = 0; i < ik.numLinks; i++)
            depth = max(depth, gDepth[i]);
        gMaxDepth = depth;
        gPairCount = pairCount;
    }
    groupSync();
}

/// Sum over the rows of the triples in `mask` of gJ[row][a] * gJ[row][b].
float tripleDot(uint2 mask, uint a, uint b)
{
    float sum = 0.0;
    for (uint word = 0; word < 2; word++)
    {
        uint m = mask[word];
        while (m != 0)
        {
            uint bit = firstbitlow(m);
            m &= m - 1;
            uint row = 3 * (32 * word + bit);
            sum += GJ(row * MAX_VARS + a) * GJ(row * MAX_VARS + b)
                 + GJ((row + 1) * MAX_VARS + a) * GJ((row + 1) * MAX_VARS + b)
                 + GJ((row + 2) * MAX_VARS + a) * GJ((row + 2) * MAX_VARS + b);
        }
    }
    return sum;
}

float tripleResidual(uint2 mask, uint a)
{
    float sum = 0.0;
    for (uint word = 0; word < 2; word++)
    {
        uint m = mask[word];
        while (m != 0)
        {
            uint bit = firstbitlow(m);
            m &= m - 1;
            uint row = 3 * (32 * word + bit);
            sum += GJ(row * MAX_VARS + a) * gR[row] + GJ((row + 1) * MAX_VARS + a) * gR[row + 1]
                 + GJ((row + 2) * MAX_VARS + a) * gR[row + 2];
        }
    }
    return sum;
}

/// Fills the six rows of target t (position then rotation) for the current pose.
void targetRows(uint t, uint frame, uint nv)
{
    // Only the entries the column masks select are read (see setupShared), and those are all written here
    // (no clearing): position rows in the root columns, rotation rows in the root rotation columns, the chain.
    int link = ik.targetLink[t];
    float2 w = ik.targetWeight[t];
    float3 p = gPos[link];
    uint chain0 = uint(ik.chainStart[t]), chainN = uint(ik.chainCount[t]);
    uint rowP = 6 * t, rowR = 6 * t + 3;

    float3 rp = w.x * (p - ik.goalPos[frame * ik.numTargets + t].xyz);
    float3 d = p - gRootPos;
    float4 twist = ik.targetTwist[t];
    float3 axis = qrotate(gRot[link], twist.xyz);
    float3 rr = untwist(w.y * qlog(qmul(gRot[link], qconj(ik.goalRot[frame * ik.numTargets + t]))), axis, twist.w);
    for (uint row = 0; row < 3; row++)
    {
        gR[rowP + row] = rp[row];
        gR[rowR + row] = rr[row];
        for (uint c = 0; c < 3; c++)
            GJ((rowP + row) * MAX_VARS + c) = row == c ? w.x : 0.0;
    }
    for (uint c = 0; c < 3; c++)
    {
        float3 col = untwist(float3(c == 0 ? w.y : 0.0, c == 1 ? w.y : 0.0, c == 2 ? w.y : 0.0), axis, twist.w);
        for (uint row = 0; row < 3; row++)
            GJ((rowR + row) * MAX_VARS + 3 + c) = col[row];
    }
    float3 c0 = cross(float3(1, 0, 0), d), c1 = cross(float3(0, 1, 0), d), c2 = cross(float3(0, 0, 1), d);
    for (uint row = 0; row < 3; row++)
    {
        GJ((rowP + row) * MAX_VARS + 3) = w.x * c0[row];
        GJ((rowP + row) * MAX_VARS + 4) = w.x * c1[row];
        GJ((rowP + row) * MAX_VARS + 5) = w.x * c2[row];
    }
    for (uint c = 0; c < chainN; c++)
    {
        int l = ik.chainLinks[chain0 + c];
        uint col = 6 + uint(ik.linkDof[l]);
        float3 jp = cross(gAxis[l], p - gOrigin[l]), jr = untwist(w.y * gAxis[l], axis, twist.w);
        for (uint row = 0; row < 3; row++)
        {
            GJ((rowP + row) * MAX_VARS + col) = w.x * jp[row];
            GJ((rowR + row) * MAX_VARS + col) = jr[row];
        }
    }
}

float3 rtStoredCom(uint stream, int frameNumber, uint link)
{
    uint slot = uint(frameNumber) % HISTORY;
    return rt.stateLinkCom[(stream * HISTORY + slot) * ik.numLinks + link].xyz;
}

bool rtAncestor(uint ancestor, uint link)
{
    uint2 mask = rt.linkAncestors[link];
    return ancestor < 32 ? ((mask.x >> ancestor) & 1) != 0 : ((mask.y >> (ancestor - 32)) & 1) != 0;
}

/// Dynamics rows (6 from `row0`) of the current frame; zero when the history is too short or disabled.
void dynamicsRows(uint stream, uint frame, int history, uint lane, uint nv, uint row0)
{
    int K = int(rt.dynamicsStride);
    bool valid = rt.dynamicsWeight > 0.0 && history >= 2 * K;
    float rate = rt.streamFps[stream] / float(K);
    float mg = rt.totalMass * 9.81;
    for (uint i = lane; i < ik.numLinks; i += GROUP)
    {
        float3 c = gPos[i] + qrotate(gRot[i], rt.linkCom[i].xyz);
        gC[i] = c;
        gA[i] = valid ? (c - 2.0 * rtStoredCom(stream, history - K, i) + rtStoredCom(stream, history - 2 * K, i)) * (rate * rate) - RT_GRAVITY
                      : float3(0, 0, 0);
    }
    groupSync();
    if (lane == 0)
    {
        float3 com = float3(0, 0, 0);
        for (uint i = 0; i < ik.numLinks; i++)
            com += rt.linkCom[i].w * gC[i];
        com /= rt.totalMass;
        gCom = com;
        float3 F = float3(0, 0, 0), N = float3(0, 0, 0);
        for (uint i = 0; i < ik.numLinks; i++)
        {
            float3 a = rt.linkCom[i].w * gA[i];
            F += a;
            N += cross(gC[i] - com, a);
        }
        float3 rf = F / mg, rn = N / (mg * RT_MOMENT_SCALE);
        // Contact points of this frame.
        float3 points[RT_CONTACT_POINTS];
        uint count = 0;
        for (uint k = 0; k < 8; k++)
        {
            if (rt.soleContact[frame * 8 + k] <= 0.0 || count >= RT_CONTACT_POINTS)
                continue;
            int l = rt.soleLink[k / 4];
            float3 p = gPos[l] + qrotate(gRot[l], rt.solePoint[k].xyz);
            points[count++] = float3(p.xy, 0.0);
        }
        for (uint l = 0; l < ik.numLinks; l++)
        {
            if (rt.linkContactRT[frame * ik.numLinks + l] <= 0.0 || count >= RT_CONTACT_POINTS)
                continue;
            points[count++] = float3(gPos[l].xy, 0.0);
        }
        // Non-negative least squares by coordinate descent on the friction-pyramid edge coefficients.
        float lambda[RT_CONTACT_POINTS * 4];
        for (uint c = 0; c < RT_CONTACT_POINTS * 4; c++)
            lambda[c] = 0.0;
        for (uint it = 0; valid && it < rt.contactIterations; it++)
        {
            for (uint c = 0; c < count; c++)
            {
                for (uint e = 0; e < 4; e++)
                {
                    float2 t = e == 0 ? float2(1, 0) : (e == 1 ? float2(0, 1) : (e == 2 ? float2(-1, 0) : float2(0, -1)));
                    float3 d = normalize(float3(rt.friction * t, 1.0));
                    float3 m = cross(points[c] - com, d) / RT_MOMENT_SCALE;
                    float old = lambda[c * 4 + e];
                    float next = max(0.0, old + (dot(d, rf) + dot(m, rn)) / (dot(d, d) + dot(m, m) + 1e-6));
                    rf -= (next - old) * d;
                    rn -= (next - old) * m;
                    lambda[c * 4 + e] = next;
                }
            }
        }
        for (uint r = 0; r < 3; r++)
        {
            gDyn[r] = valid ? rt.dynamicsWeight * rf[r] : 0.0;
            gDyn[3 + r] = valid ? rt.dynamicsWeight * rn[r] : 0.0;
        }
        gResForce = valid ? rf : float3(0, 0, 0);
    }
    groupSync();
    for (uint col = lane; col < nv; col += GROUP)
    {
        float3 jF = float3(0, 0, 0), jN = float3(0, 0, 0), jC = float3(0, 0, 0);
        int dofLink = -1;
        if (col >= 6)
            for (uint l = 0; l < ik.numLinks; l++)
                if (ik.linkDof[l] == int(col) - 6)
                    dofLink = int(l);
        for (uint i = 0; valid && i < ik.numLinks; i++)
        {
            float m = rt.linkCom[i].w;
            if (m <= 0.0)
                continue;
            float3 J;
            if (col < 3)
                J = float3(col == 0 ? 1 : 0, col == 1 ? 1 : 0, col == 2 ? 1 : 0);
            else if (col < 6)
                J = cross(float3(col == 3 ? 1 : 0, col == 4 ? 1 : 0, col == 5 ? 1 : 0), gC[i] - gRootPos);
            else
            {
                if (!rtAncestor(uint(dofLink), i))
                    continue;
                J = cross(gAxis[dofLink], gC[i] - gOrigin[dofLink]);
            }
            float3 dj = (rate * rate) * J;
            jF += m * dj;
            jN += m * (cross(gC[i] - gCom, dj) - cross(gA[i], J));
            jC += m * J;
        }
        // The moment reference (the CoM) moves too: -dc x F_residual.
        jN -= cross(jC / rt.totalMass, gResForce * mg);
        for (uint r = 0; r < 3; r++)
        {
            GJ((row0 + r) * MAX_VARS + col) = valid ? rt.dynamicsWeight * jF[r] / mg : 0.0;
            GJ((row0 + 3 + r) * MAX_VARS + col) = valid ? rt.dynamicsWeight * jN[r] / (mg * RT_MOMENT_SCALE) : 0.0;
        }
    }
    for (uint r = lane; r < 6; r += GROUP)
        gR[row0 + r] = gDyn[r];
}

/// Gauss-Newton iterations on the current frame from the current pose (gQ, gRootPos / gRootRot); the last pass
/// only evaluates the rows (gR holds the final residuals, gPos / gRot the final pose).
void gaussNewton(uint s, uint frame, uint frameIterations, float frameSmoothness, int dynamicsHistory, uint lane, uint nv,
                 uint rowDynamics)
{
    uint n = ik.numDofs;
    for (uint it = 0; it <= frameIterations; it++)
    {
        forwardShared(lane);
        for (uint t = lane; t < ik.numTargets; t += GROUP)
            targetRows(t, frame, nv);
        groupSync();
        if (it == frameIterations)
            break;
        if (rt.dynamicsWeight > 0.0)
        {
            dynamicsRows(s, frame, dynamicsHistory, lane, nv, rowDynamics);
            groupSync();
        }

        // H = J^T J (lower triangle), g = J^T r, over the row triples each column pair shares; plus the
        // joint regularisers and damping on the diagonal.
        for (uint e = lane; e < nv * HS; e += GROUP)
            gH[e] = 0.0;
        groupSync();
        for (uint k = lane; k < gPairCount; k += GROUP)
        {
            uint a = gPairs[k] & 0xff, b = gPairs[k] >> 8;
            uint ia = gInv[a], ib = gInv[b];
            gH[max(ia, ib) * HS + min(ia, ib)] = tripleDot(gColMask[a] & gColMask[b], a, b);
        }
        for (uint a = lane; a < nv; a += GROUP)
            gG[gInv[a]] = tripleResidual(gColMask[a], a);
        groupSync();
        for (uint j = lane; j < n; j += GROUP)
        {
            uint v = 6 + j;
            float2 lim = ik.dofLimit[j];
            float mid = 0.5 * (lim.x + lim.y), half = 0.5 * (lim.y - lim.x) * ik.limitMargin;
            float ws = frameSmoothness * frameSmoothness, wp = ik.posture * ik.posture;
            float h = ws + wp;
            float gv = ws * (gQ[j] - gQPrev[j]) + wp * gQ[j];
            float excess = abs(gQ[j] - mid) - half;
            if (excess > 0.0)
            {
                float wl = ik.limitWeight * ik.limitWeight;
                h += wl;
                gv += wl * excess * sign(gQ[j] - mid);
            }
            uint pv = gInv[v];
            gH[pv * HS + pv] += h;
            gG[pv] += gv;
        }
        groupSync();
        for (uint v = lane; v < nv; v += GROUP)
            gH[v * HS + v] += ik.damping * (1.0 + gH[v * HS + v]);
        groupSync();

        // Right-looking sparse Cholesky in elimination order (lower triangle of gH becomes L, diagonal in
        // gDiag), fused with the forward substitution L y = g (y in gCol): step j scales column j's nonzero
        // rows and updates g below j, then the trailing entries among those rows.
        for (uint j = 0; j < nv; j++)
        {
            float d = sqrt(max(gH[j * HS + j], 1e-12));
            float y = gG[j] / d;
            uint c0 = gLColStart[j], cn = gLColStart[j + 1] - c0;
            for (uint e = lane; e < cn; e += GROUP)
            {
                uint i = gLCol[c0 + e];
                float l = gH[i * HS + j] / d;
                gH[i * HS + j] = l;
                gL[i] = l;
                gG[i] -= l * y;
            }
            if (lane == 0)
            {
                gDiag[j] = d;
                gCol[j] = y;
            }
            groupSync();
            if (cn == 0)
                continue;
            // Trailing update: the (e, e2 <= e) pairs of the column's rows, spread over the lanes.
            for (uint p = lane; p < cn * (cn + 1) / 2; p += GROUP)
            {
                uint e = uint((sqrt(float(8 * p + 1)) - 1.0) * 0.5);
                if ((e + 1) * (e + 2) / 2 <= p)
                    e++;
                else if (e * (e + 1) / 2 > p)
                    e--;
                uint i = gLCol[c0 + e], k = gLCol[c0 + p - e * (e + 1) / 2];
                gH[i * HS + k] -= gL[i] * gL[k];
            }
            groupSync();
        }
        // Backward substitution L^T x = y (y in gCol), step -x into gG.
        for (int i = int(nv) - 1; i >= 0; i--)
        {
            uint u = uint(i);
            float x = gCol[u] / gDiag[u];
            uint r0 = gLRowStart[u], rn = gLRowStart[u + 1] - r0;
            for (uint e = lane; e < rn; e += GROUP)
            {
                uint k = gLRow[r0 + e];
                gCol[k] -= gH[u * HS + k] * x;
            }
            if (lane == 0)
                gG[u] = -x;
            groupSync();
        }
        if (lane == 0)
        {
            gRootPos += float3(gG[gInv[0]], gG[gInv[1]], gG[gInv[2]]);
            gRootRot = normalize(qmul(qexp(float3(gG[gInv[3]], gG[gInv[4]], gG[gInv[5]])), gRootRot));
        }
        for (uint j = lane; j < n; j += GROUP)
        {
            float2 lim = ik.dofLimit[j];
            gQ[j] = clamp(gQ[j] + gG[gInv[6 + j]], lim.x, lim.y);
        }
        groupSync();
    }
}

/// Tracking cost of one arm's targets (from gR) and the largest position error among them (m); lane 0.
float2 armTracking(int arm)
{
    float cost = 0.0, error = 0.0;
    for (uint t = 0; t < ik.numTargets; t++)
    {
        if (rt.targetArm[t] != arm)
            continue;
        for (uint r = 0; r < 6; r++)
            cost += gR[6 * t + r] * gR[6 * t + r];
        float w = ik.targetWeight[t].x;
        if (w > 0.0)
            error = max(error, length(float3(gR[6 * t], gR[6 * t + 1], gR[6 * t + 2])) / w);
    }
    return float2(cost, error);
}

/// Saves the current solve (joints, previous joints, root) before another branch is solved in its place.
void saveMain(uint lane)
{
    for (uint j = lane; j < ik.numDofs; j += GROUP)
    {
        gQMain[j] = gQ[j];
        gQMainPrev[j] = gQPrev[j];
    }
    if (lane == 0)
    {
        gRootPosMain = gRootPos;
        gRootRotMain = gRootRot;
    }
    groupSync();
}

/// Copies the solved `arm`'s joints to gQBranch and puts the saved solve back.
void restoreMain(uint lane, int arm)
{
    for (uint j = lane; j < ik.numDofs; j += GROUP)
    {
        if (rt.dofArm[j] == arm)
            gQBranch[j] = gQ[j];
        gQ[j] = gQMain[j];
        gQPrev[j] = gQMainPrev[j];
    }
    if (lane == 0)
    {
        gRootPos = gRootPosMain;
        gRootRot = gRootRotMain;
    }
    groupSync();
}

/// The branch switch after a frame's solve (see `branchBlend`). Leaves the output pose in gQ / gRootPos / gRootRot
/// and the state to carry on from in gQMain; returns true if they may differ (an arm has a second branch).
/// Per arm, gBranch holds (active, blend position 0..branchBlend, frames idle, direction, evidence): while active,
/// both branches are solved every frame and the blend position moves one frame per frame in the current direction.
/// The direction turns toward a branch once it has tracked better by `branchHysteresis` (cost ratio) for
/// `branchDwell` frames in a row: crowd_cheer_001__A029 pumps an arm overhead where each branch wins in turn for
/// a few tenths of a second, and switching at every turn swung the shoulder 3 rad back and forth. Reaching the
/// incoming branch swaps the two (the old one stays as the alternative); the alternative is dropped once the
/// output has rested on the current branch for `branchKeep` frames, or when the two have merged.
/// Whether the person touches anything at `frame` (causal contacts): sole corners or other body parts.
bool grounded(uint frame)
{
    for (uint k = 0; k < 8; k++)
        if (rt.soleContact[frame * 8 + k] > 0.0)
            return true;
    for (uint l = 0; l < ik.numLinks; l++)
        if (rt.linkContactRT[frame * ik.numLinks + l] > 0.0)
            return true;
    return false;
}

bool branchSwitch(uint s, uint frame, int history, uint lane, uint nv, uint rowDynamics)
{
    uint n = ik.numDofs;
    bool any = false;
    bool airborne = !grounded(frame);
    float weight[2] = { -1.0, -1.0 };
    bool swap[2] = { false, false };
    // The current branch's arm tracking, from the frame's solve (gR still holds its rows).
    if (lane == 0)
    {
        gArmMain[0] = armTracking(1);
        gArmMain[1] = armTracking(2);
    }
    groupSync();
    for (int arm = 1; arm <= 2; arm++)
    {
        uint b0 = uint(arm - 1) * 8;
        float costMain = gArmMain[arm - 1].x;
        if (gBranch[b0] == 0)
        {
            if (rt.branchInterval == 0 || uint(history) % rt.branchInterval != 0 || gArmMain[arm - 1].y <= rt.branchError)
                continue;
            saveMain(lane);
            for (uint j = lane; j < n; j += GROUP)
                if (rt.dofArm[j] == arm)
                    gQ[j] = 0.0;
            groupSync();
            gaussNewton(s, frame, rt.branchIterations, 0.0, history, lane, nv, rowDynamics);
        }
        else
        {
            // The alternative branch: warm from its last pose, smoothness against it.
            saveMain(lane);
            for (uint j = lane; j < n; j += GROUP)
                if (rt.dofArm[j] == arm)
                {
                    gQ[j] = rt.stateBranchQ[s * n + j];
                    gQPrev[j] = rt.stateBranchQ[s * n + j];
                }
            groupSync();
            gaussNewton(s, frame, ik.iterations, ik.smoothness, history, lane, nv, rowDynamics);
        }
        if (lane == 0)
            gArmCost = armTracking(arm).x;
        groupSync();
        restoreMain(lane, arm);
        float costAlt = gArmCost;
        if (gBranch[b0] == 0)
        {
            // A new branch is taken on only if it more than halves the arm's tracking cost. One that quarters it is
            // eased into at once (reach_jump_R_002__A212: the arm stuck at the shoulder's roll / yaw stops while the
            // person's dropped to the side, 0.5 s behind with the dwell), or from landing when found in the air.
            // Otherwise it waits out the dwell like a reversal.
            if (!(costAlt < 0.5 * costMain))
                continue;
            bool clear = costAlt < 0.25 * costMain;
            if (lane == 0)
            {
                gBranch[b0] = 1;
                gBranch[b0 + 1] = 0;
                gBranch[b0 + 2] = 0;
                gBranch[b0 + 3] = clear ? 1 : 0;
                gBranch[b0 + 4] = clear ? int(rt.branchDwell) : 0;
            }
        }
        for (uint j = lane; j < n; j += GROUP)
            if (rt.dofArm[j] == arm)
                rt.stateBranchQ[s * n + j] = gQBranch[j];
        // Merged: both branches within 0.05 rad on every arm joint.
        if (lane == 0)
        {
            float gap = 0.0;
            for (uint j = 0; j < n; j++)
                if (rt.dofArm[j] == arm)
                    gap = max(gap, abs(gQBranch[j] - gQMain[j]));
            gKeep = gap < 0.05 ? 1 : 0;
        }
        groupSync();
        int p = gBranch[b0 + 1], idle = gBranch[b0 + 2], dir = gBranch[b0 + 3], evidence = gBranch[b0 + 4];
        if (costAlt < rt.branchHysteresis * costMain)
            evidence = max(evidence, 0) + 1;
        else if (costMain < rt.branchHysteresis * costAlt)
            evidence = min(evidence, 0) - 1;
        else
            evidence = 0;
        if (evidence >= int(rt.branchDwell))
            dir = 1;
        else if (evidence <= -int(rt.branchDwell))
            dir = -1;
        // At half speed while the person is airborne: the blend swings the arm through up to 3 rad, and in flight
        // nothing supplies the angular momentum that takes (reach_jump_R_002__A213: residual 1.0 m·g at full speed).
        p = clamp(p + (airborne && history % 2 != 0 ? 0 : dir), 0, int(rt.branchBlend));
        bool swapped = uint(p) >= rt.branchBlend;
        idle = p == 0 && dir <= 0 ? idle + 1 : 0;
        float x = float(p) / float(rt.branchBlend);
        weight[arm - 1] = x * x * (3.0 - 2.0 * x);
        swap[arm - 1] = swapped;
        any = true;
        groupSync();
        if (lane == 0)
        {
            gBranch[b0 + 1] = swapped ? 0 : p;
            gBranch[b0 + 2] = idle;
            gBranch[b0 + 3] = swapped ? -1 : dir;
            gBranch[b0 + 4] = swapped ? 0 : evidence;
            if ((p == 0 && gKeep != 0) || uint(idle) >= rt.branchKeep)
                gBranch[b0] = 0;
        }
        groupSync();
    }
    if (!any)
        return false;
    // Output: each arm with two branches between them; a finished blend swaps them.
    saveMain(lane);
    for (uint j = lane; j < n; j += GROUP)
    {
        int arm = rt.dofArm[j];
        if (arm == 0 || weight[arm - 1] < 0.0)
            continue;
        float b = rt.stateBranchQ[s * n + j];
        gQ[j] = lerp(gQMain[j], b, weight[arm - 1]);
        if (swap[arm - 1])
        {
            rt.stateBranchQ[s * n + j] = gQMain[j];
            gQMain[j] = b;
        }
    }
    groupSync();
    return true;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void solveGroup(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID)
{
    uint s = gid.x;
    uint lane = gtid.x;
    if (s >= ik.numStreams)
        return; // whole group exits together
    uint n = ik.numDofs;
    uint nv = 6 + n;
    uint rowDynamics = 6 * ik.numTargets;
    int offset = ik.streamFrameOffset[s];
    int frames = ik.streamFrameCount[s];
    int history = ik.stateValid[s] != 0 ? rt.stateHistory[s] : 0;

    if (lane == 0)
    {
#ifdef __METAL__
        gJBase = s * JACOBIAN_SIZE;
#endif
        if (ik.stateValid[s] != 0)
        {
            gRootPos = ik.stateRootPos[s].xyz;
            gRootRot = ik.stateRootRot[s];
            for (uint j = 0; j < n; j++)
                gQ[j] = ik.stateQ[s * n + j];
        }
        else
        {
            gRootPos = ik.goalPos[offset * ik.numTargets + ik.rootTarget].xyz;
            gRootRot = ik.goalRot[offset * ik.numTargets + ik.rootTarget];
            for (uint j = 0; j < n; j++)
                gQ[j] = 0.0;
        }
    }
    if (lane < 16)
        gBranch[lane] = rt.branchBlend > 0 && ik.stateValid[s] != 0 ? rt.stateBranch[s * 16 + lane] : 0;
    groupSync();

    setupShared(lane, nv);

    for (int f = 0; f < frames; f++)
    {
        uint frame = uint(offset + f);
        for (uint j = lane; j < n; j += GROUP)
            gQPrev[j] = gQ[j];
        groupSync();

        // A cold stream's first frame starts from q = 0: converge it fully and without the smoothness pull.
        bool cold = f == 0 && ik.stateValid[s] == 0;
        uint frameIterations = cold ? max(ik.iterations, ik.coldIterations) : ik.iterations;
        float frameSmoothness = cold ? 0.0 : ik.smoothness;

        gaussNewton(s, frame, frameIterations, frameSmoothness, cold ? 0 : history, lane, nv, rowDynamics);
        bool blended = rt.branchBlend > 0 && !cold && branchSwitch(s, frame, history, lane, nv, rowDynamics);
        if (blended)
        {
            // gQ holds the output pose now: its link poses and tracking rows for the CoM ring and the cost.
            forwardShared(lane);
            for (uint t = lane; t < ik.numTargets; t += GROUP)
                targetRows(t, frame, nv);
            groupSync();
        }

        // Push this frame's link CoMs into the history ring (gPos / gRot hold the final pose).
        for (uint i = lane; i < ik.numLinks; i += GROUP)
            rt.stateLinkCom[(s * HISTORY + uint(history) % HISTORY) * ik.numLinks + i] = float4(gPos[i] + qrotate(gRot[i], rt.linkCom[i].xyz), 0.0);
        groupSync();
        history++;
        if (lane == 0)
        {
            float cost = 0.0;
            for (uint r = 0; r < rowDynamics; r++)
                cost += gR[r] * gR[r];
            ik.outRootPos[frame] = float4(gRootPos, 0.0);
            ik.outRootRot[frame] = gRootRot;
            for (uint j = 0; j < n; j++)
                ik.outQ[frame * n + j] = gQ[j];
            ik.outCost[frame] = cost;
        }
        groupSync();
        if (blended)
        {
            for (uint j = lane; j < n; j += GROUP)
                gQ[j] = gQMain[j];
            groupSync();
        }
    }

    if (lane == 0)
    {
        rt.stateHistory[s] = history;
        ik.stateRootPos[s] = float4(gRootPos, 0.0);
        ik.stateRootRot[s] = gRootRot;
        for (uint j = 0; j < n; j++)
            ik.stateQ[s * n + j] = gQ[j];
    }
    if (lane < 16 && rt.branchBlend > 0)
        rt.stateBranch[s * 16 + lane] = gBranch[lane];
}
