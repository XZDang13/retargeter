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

uniform StructuredBuffer<float4> linkCom;          // [numLinks] link-frame CoM, w = mass
uniform StructuredBuffer<uint2> linkAncestors;     // [numLinks] bit l: link l is the link or an ancestor
uniform StructuredBuffer<int> soleLink;            // [2]
uniform StructuredBuffer<float4> solePoint;        // [8] link frame
uniform StructuredBuffer<float> soleContact;       // [frame * 8] 1: the human heel / toe under this corner touches
uniform StructuredBuffer<float> linkContactRT;     // [frame * numLinks] 1: mapped body part in contact (not feet)
uniform StructuredBuffer<float> streamFps;         // [stream]
uniform StructuredBuffer<uint> factorStructure;    // see setupShared
uniform RWStructuredBuffer<float4> stateLinkCom;   // [(stream * HISTORY + slot) * numLinks + link]
uniform RWStructuredBuffer<int> stateHistory;      // [stream] frames pushed into the ring
// Arm branch switch: a warm-started arm that has run into a joint-limit corner (jump_right_003__A030: the arm raised
// sideways reached through shoulder abduction, roll and yaw at their stops, 14 cm off for two seconds) cannot leave
// it while the target moves on, and eventually snaps across in a frame or two. Every `branchInterval` frames an arm
// whose elbow or hand is more than `branchError` off is solved again from its joints at 0 (`branchIterations`,
// no smoothness pull). If that more than halves the arm's tracking cost, the new branch is tracked alongside the
// current one (warm, its own smoothness pull) and the output arm eases between them with a smoothstep over
// `branchBlend` frames, toward whichever tracks better (see branchSwitch). Switching in one frame moved joints up
// to 3 rad; dropping the old branch at once switched crowd_cheer_001__A029 into one that went bad a second later.
// branchBlend 0: off.
uniform StructuredBuffer<int> dofArm;              // [numDofs] 0, 1 left arm, 2 right arm
uniform StructuredBuffer<int> targetArm;           // [numTargets] the arm a target belongs to (0: none)
uniform RWStructuredBuffer<float> stateBranchQ;    // [stream * numDofs] the branch being blended in (arm dofs)
uniform RWStructuredBuffer<int> stateBranch;       // [stream * 16] per arm: active, blend position, idle, direction, evidence
uniform uint branchInterval;
uniform float branchError;
uniform uint branchIterations;
uniform uint branchBlend;
uniform float branchHysteresis;
uniform uint branchKeep;
uniform uint branchDwell;
uniform float dynamicsWeight;
uniform uint dynamicsStride;
uniform float friction;
uniform float totalMass;
uniform uint contactIterations;

groupshared float gJ[RT_ROWS * MAX_VARS];   // dense weighted Jacobian rows (6 per target, 6 dynamics)
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
    for (uint i = lane; i < numLinks; i += GROUP)
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
    GroupMemoryBarrierWithGroupSync();
    for (uint level = 1; level <= gMaxDepth; level++)
    {
        for (uint i = lane; i < numLinks; i += GROUP)
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
        GroupMemoryBarrierWithGroupSync();
    }
}

/// Once per dispatch: model cache, link depths, column triple masks and the column pairs to sum.
void setupShared(uint lane, uint nv)
{
    for (uint i = lane; i < numLinks; i += GROUP)
    {
        gJointOriginRot[i] = jointOriginRot[i];
        gJointOriginPos[i] = jointOriginPos[i].xyz;
        gJointToChildRot[i] = jointToChildRot[i];
        gJointToChildPos[i] = jointToChildPos[i].xyz;
        gJointAxis[i] = jointAxis[i].xyz;
        gAxisLocal[i] = qrotate(jointOriginRot[i], jointAxis[i].xyz);
        gLinkParent[i] = linkParent[i];
        gLinkDof[i] = linkDof[i];
        uint depth = 0;
        for (int k = linkParent[i]; k >= 0; k = linkParent[k])
            depth++;
        gDepth[i] = depth;
    }
    // Sparsity structure from the host (RetargetSolver.factorStructure: column masks, the column pairs that share
    // a row triple heaviest first, the minimum-degree elimination order and the factor's row / column lists). It
    // depends only on the model and the weights; derived here on one lane it cost ~3 ms per dispatch, which
    // dominated one-frame (teleop) dispatches.
    uint pairCount = factorStructure[0], offPairs = factorStructure[1], offMask = factorStructure[2];
    uint offInv = factorStructure[3], offColStart = factorStructure[4], offCol = factorStructure[5];
    uint offRowStart = factorStructure[6], offRow = factorStructure[7], factorCount = factorStructure[8];
    for (uint k = lane; k < pairCount; k += GROUP)
        gPairs[k] = factorStructure[offPairs + k];
    for (uint a = lane; a < nv; a += GROUP)
    {
        gColMask[a] = uint2(factorStructure[offMask + 2 * a], factorStructure[offMask + 2 * a + 1]);
        gInv[a] = factorStructure[offInv + a];
    }
    for (uint j = lane; j <= nv; j += GROUP)
    {
        gLColStart[j] = factorStructure[offColStart + j];
        gLRowStart[j] = factorStructure[offRowStart + j];
    }
    for (uint e = lane; e < factorCount; e += GROUP)
    {
        gLCol[e] = uint8_t(factorStructure[offCol + e]);
        gLRow[e] = uint8_t(factorStructure[offRow + e]);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0)
    {
        uint depth = 0;
        for (uint i = 0; i < numLinks; i++)
            depth = max(depth, gDepth[i]);
        gMaxDepth = depth;
        gPairCount = pairCount;
    }
    GroupMemoryBarrierWithGroupSync();
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
            sum += gJ[row * MAX_VARS + a] * gJ[row * MAX_VARS + b]
                 + gJ[(row + 1) * MAX_VARS + a] * gJ[(row + 1) * MAX_VARS + b]
                 + gJ[(row + 2) * MAX_VARS + a] * gJ[(row + 2) * MAX_VARS + b];
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
            sum += gJ[row * MAX_VARS + a] * gR[row] + gJ[(row + 1) * MAX_VARS + a] * gR[row + 1]
                 + gJ[(row + 2) * MAX_VARS + a] * gR[row + 2];
        }
    }
    return sum;
}

/// Fills the six rows of target t (position then rotation) for the current pose.
void targetRows(uint t, uint frame, uint nv)
{
    // Only the entries the column masks select are read (see setupShared), and those are all written here
    // (no clearing): position rows in the root columns, rotation rows in the root rotation columns, the chain.
    int link = targetLink[t];
    float2 w = targetWeight[t];
    float3 p = gPos[link];
    uint chain0 = uint(chainStart[t]), chainN = uint(chainCount[t]);
    uint rowP = 6 * t, rowR = 6 * t + 3;

    float3 rp = w.x * (p - goalPos[frame * numTargets + t].xyz);
    float3 d = p - gRootPos;
    float4 twist = targetTwist[t];
    float3 axis = qrotate(gRot[link], twist.xyz);
    float3 rr = untwist(w.y * qlog(qmul(gRot[link], qconj(goalRot[frame * numTargets + t]))), axis, twist.w);
    for (uint row = 0; row < 3; row++)
    {
        gR[rowP + row] = rp[row];
        gR[rowR + row] = rr[row];
        for (uint c = 0; c < 3; c++)
            gJ[(rowP + row) * MAX_VARS + c] = row == c ? w.x : 0.0;
    }
    for (uint c = 0; c < 3; c++)
    {
        float3 col = untwist(float3(c == 0 ? w.y : 0.0, c == 1 ? w.y : 0.0, c == 2 ? w.y : 0.0), axis, twist.w);
        for (uint row = 0; row < 3; row++)
            gJ[(rowR + row) * MAX_VARS + 3 + c] = col[row];
    }
    float3 c0 = cross(float3(1, 0, 0), d), c1 = cross(float3(0, 1, 0), d), c2 = cross(float3(0, 0, 1), d);
    for (uint row = 0; row < 3; row++)
    {
        gJ[(rowP + row) * MAX_VARS + 3] = w.x * c0[row];
        gJ[(rowP + row) * MAX_VARS + 4] = w.x * c1[row];
        gJ[(rowP + row) * MAX_VARS + 5] = w.x * c2[row];
    }
    for (uint c = 0; c < chainN; c++)
    {
        int l = chainLinks[chain0 + c];
        uint col = 6 + uint(linkDof[l]);
        float3 jp = cross(gAxis[l], p - gOrigin[l]), jr = untwist(w.y * gAxis[l], axis, twist.w);
        for (uint row = 0; row < 3; row++)
        {
            gJ[(rowP + row) * MAX_VARS + col] = w.x * jp[row];
            gJ[(rowR + row) * MAX_VARS + col] = jr[row];
        }
    }
}

float3 rtStoredCom(uint stream, int frameNumber, uint link)
{
    uint slot = uint(frameNumber) % HISTORY;
    return stateLinkCom[(stream * HISTORY + slot) * numLinks + link].xyz;
}

bool rtAncestor(uint ancestor, uint link)
{
    uint2 mask = linkAncestors[link];
    return ancestor < 32 ? ((mask.x >> ancestor) & 1) != 0 : ((mask.y >> (ancestor - 32)) & 1) != 0;
}

/// Dynamics rows (6 from `row0`) of the current frame; zero when the history is too short or disabled.
void dynamicsRows(uint stream, uint frame, int history, uint lane, uint nv, uint row0)
{
    int K = int(dynamicsStride);
    bool valid = dynamicsWeight > 0.0 && history >= 2 * K;
    float rate = streamFps[stream] / float(K);
    float mg = totalMass * 9.81;
    for (uint i = lane; i < numLinks; i += GROUP)
    {
        float3 c = gPos[i] + qrotate(gRot[i], linkCom[i].xyz);
        gC[i] = c;
        gA[i] = valid ? (c - 2.0 * rtStoredCom(stream, history - K, i) + rtStoredCom(stream, history - 2 * K, i)) * (rate * rate) - RT_GRAVITY
                      : float3(0, 0, 0);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0)
    {
        float3 com = float3(0, 0, 0);
        for (uint i = 0; i < numLinks; i++)
            com += linkCom[i].w * gC[i];
        com /= totalMass;
        gCom = com;
        float3 F = float3(0, 0, 0), N = float3(0, 0, 0);
        for (uint i = 0; i < numLinks; i++)
        {
            float3 a = linkCom[i].w * gA[i];
            F += a;
            N += cross(gC[i] - com, a);
        }
        float3 rf = F / mg, rn = N / (mg * RT_MOMENT_SCALE);
        // Contact points of this frame.
        float3 points[RT_CONTACT_POINTS];
        uint count = 0;
        for (uint k = 0; k < 8; k++)
        {
            if (soleContact[frame * 8 + k] <= 0.0 || count >= RT_CONTACT_POINTS)
                continue;
            int l = soleLink[k / 4];
            float3 p = gPos[l] + qrotate(gRot[l], solePoint[k].xyz);
            points[count++] = float3(p.xy, 0.0);
        }
        for (uint l = 0; l < numLinks; l++)
        {
            if (linkContactRT[frame * numLinks + l] <= 0.0 || count >= RT_CONTACT_POINTS)
                continue;
            points[count++] = float3(gPos[l].xy, 0.0);
        }
        // Non-negative least squares by coordinate descent on the friction-pyramid edge coefficients.
        float lambda[RT_CONTACT_POINTS * 4];
        for (uint c = 0; c < RT_CONTACT_POINTS * 4; c++)
            lambda[c] = 0.0;
        for (uint it = 0; valid && it < contactIterations; it++)
        {
            for (uint c = 0; c < count; c++)
            {
                for (uint e = 0; e < 4; e++)
                {
                    float2 t = e == 0 ? float2(1, 0) : (e == 1 ? float2(0, 1) : (e == 2 ? float2(-1, 0) : float2(0, -1)));
                    float3 d = normalize(float3(friction * t, 1.0));
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
            gDyn[r] = valid ? dynamicsWeight * rf[r] : 0.0;
            gDyn[3 + r] = valid ? dynamicsWeight * rn[r] : 0.0;
        }
        gResForce = valid ? rf : float3(0, 0, 0);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint col = lane; col < nv; col += GROUP)
    {
        float3 jF = float3(0, 0, 0), jN = float3(0, 0, 0), jC = float3(0, 0, 0);
        int dofLink = -1;
        if (col >= 6)
            for (uint l = 0; l < numLinks; l++)
                if (linkDof[l] == int(col) - 6)
                    dofLink = int(l);
        for (uint i = 0; valid && i < numLinks; i++)
        {
            float m = linkCom[i].w;
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
        jN -= cross(jC / totalMass, gResForce * mg);
        for (uint r = 0; r < 3; r++)
        {
            gJ[(row0 + r) * MAX_VARS + col] = valid ? dynamicsWeight * jF[r] / mg : 0.0;
            gJ[(row0 + 3 + r) * MAX_VARS + col] = valid ? dynamicsWeight * jN[r] / (mg * RT_MOMENT_SCALE) : 0.0;
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
    uint n = numDofs;
    for (uint it = 0; it <= frameIterations; it++)
    {
        forwardShared(lane);
        for (uint t = lane; t < numTargets; t += GROUP)
            targetRows(t, frame, nv);
        GroupMemoryBarrierWithGroupSync();
        if (it == frameIterations)
            break;
        if (dynamicsWeight > 0.0)
        {
            dynamicsRows(s, frame, dynamicsHistory, lane, nv, rowDynamics);
            GroupMemoryBarrierWithGroupSync();
        }

        // H = J^T J (lower triangle), g = J^T r, over the row triples each column pair shares; plus the
        // joint regularisers and damping on the diagonal.
        for (uint e = lane; e < nv * HS; e += GROUP)
            gH[e] = 0.0;
        GroupMemoryBarrierWithGroupSync();
        for (uint k = lane; k < gPairCount; k += GROUP)
        {
            uint a = gPairs[k] & 0xff, b = gPairs[k] >> 8;
            uint ia = gInv[a], ib = gInv[b];
            gH[max(ia, ib) * HS + min(ia, ib)] = tripleDot(gColMask[a] & gColMask[b], a, b);
        }
        for (uint a = lane; a < nv; a += GROUP)
            gG[gInv[a]] = tripleResidual(gColMask[a], a);
        GroupMemoryBarrierWithGroupSync();
        for (uint j = lane; j < n; j += GROUP)
        {
            uint v = 6 + j;
            float2 lim = dofLimit[j];
            float mid = 0.5 * (lim.x + lim.y), half = 0.5 * (lim.y - lim.x) * limitMargin;
            float ws = frameSmoothness * frameSmoothness, wp = posture * posture;
            float h = ws + wp;
            float gv = ws * (gQ[j] - gQPrev[j]) + wp * gQ[j];
            float excess = abs(gQ[j] - mid) - half;
            if (excess > 0.0)
            {
                float wl = limitWeight * limitWeight;
                h += wl;
                gv += wl * excess * sign(gQ[j] - mid);
            }
            uint pv = gInv[v];
            gH[pv * HS + pv] += h;
            gG[pv] += gv;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint v = lane; v < nv; v += GROUP)
            gH[v * HS + v] += damping * (1.0 + gH[v * HS + v]);
        GroupMemoryBarrierWithGroupSync();

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
            GroupMemoryBarrierWithGroupSync();
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
            GroupMemoryBarrierWithGroupSync();
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
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0)
        {
            gRootPos += float3(gG[gInv[0]], gG[gInv[1]], gG[gInv[2]]);
            gRootRot = normalize(qmul(qexp(float3(gG[gInv[3]], gG[gInv[4]], gG[gInv[5]])), gRootRot));
        }
        for (uint j = lane; j < n; j += GROUP)
        {
            float2 lim = dofLimit[j];
            gQ[j] = clamp(gQ[j] + gG[gInv[6 + j]], lim.x, lim.y);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

/// Tracking cost of one arm's targets (from gR) and the largest position error among them (m); lane 0.
float2 armTracking(int arm)
{
    float cost = 0.0, error = 0.0;
    for (uint t = 0; t < numTargets; t++)
    {
        if (targetArm[t] != arm)
            continue;
        for (uint r = 0; r < 6; r++)
            cost += gR[6 * t + r] * gR[6 * t + r];
        float w = targetWeight[t].x;
        if (w > 0.0)
            error = max(error, length(float3(gR[6 * t], gR[6 * t + 1], gR[6 * t + 2])) / w);
    }
    return float2(cost, error);
}

/// Saves the current solve (joints, previous joints, root) before another branch is solved in its place.
void saveMain(uint lane)
{
    for (uint j = lane; j < numDofs; j += GROUP)
    {
        gQMain[j] = gQ[j];
        gQMainPrev[j] = gQPrev[j];
    }
    if (lane == 0)
    {
        gRootPosMain = gRootPos;
        gRootRotMain = gRootRot;
    }
    GroupMemoryBarrierWithGroupSync();
}

/// Copies the solved `arm`'s joints to gQBranch and puts the saved solve back.
void restoreMain(uint lane, int arm)
{
    for (uint j = lane; j < numDofs; j += GROUP)
    {
        if (dofArm[j] == arm)
            gQBranch[j] = gQ[j];
        gQ[j] = gQMain[j];
        gQPrev[j] = gQMainPrev[j];
    }
    if (lane == 0)
    {
        gRootPos = gRootPosMain;
        gRootRot = gRootRotMain;
    }
    GroupMemoryBarrierWithGroupSync();
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
bool branchSwitch(uint s, uint frame, int history, uint lane, uint nv, uint rowDynamics)
{
    uint n = numDofs;
    bool any = false;
    float weight[2] = { -1.0, -1.0 };
    bool swap[2] = { false, false };
    // The current branch's arm tracking, from the frame's solve (gR still holds its rows).
    if (lane == 0)
    {
        gArmMain[0] = armTracking(1);
        gArmMain[1] = armTracking(2);
    }
    GroupMemoryBarrierWithGroupSync();
    for (int arm = 1; arm <= 2; arm++)
    {
        uint b0 = uint(arm - 1) * 8;
        float costMain = gArmMain[arm - 1].x;
        if (gBranch[b0] == 0)
        {
            if (branchInterval == 0 || uint(history) % branchInterval != 0 || gArmMain[arm - 1].y <= branchError)
                continue;
            saveMain(lane);
            for (uint j = lane; j < n; j += GROUP)
                if (dofArm[j] == arm)
                    gQ[j] = 0.0;
            GroupMemoryBarrierWithGroupSync();
            gaussNewton(s, frame, branchIterations, 0.0, history, lane, nv, rowDynamics);
        }
        else
        {
            // The alternative branch: warm from its last pose, smoothness against it.
            saveMain(lane);
            for (uint j = lane; j < n; j += GROUP)
                if (dofArm[j] == arm)
                {
                    gQ[j] = stateBranchQ[s * n + j];
                    gQPrev[j] = stateBranchQ[s * n + j];
                }
            GroupMemoryBarrierWithGroupSync();
            gaussNewton(s, frame, iterations, smoothness, history, lane, nv, rowDynamics);
        }
        if (lane == 0)
            gArmCost = armTracking(arm).x;
        GroupMemoryBarrierWithGroupSync();
        restoreMain(lane, arm);
        float costAlt = gArmCost;
        if (gBranch[b0] == 0)
        {
            // A new branch is taken on only if it more than halves the arm's tracking cost.
            if (!(costAlt < 0.5 * costMain))
                continue;
            if (lane == 0)
            {
                gBranch[b0] = 1;
                gBranch[b0 + 1] = 0;
                gBranch[b0 + 2] = 0;
                gBranch[b0 + 3] = 0;
                gBranch[b0 + 4] = 0;
            }
        }
        for (uint j = lane; j < n; j += GROUP)
            if (dofArm[j] == arm)
                stateBranchQ[s * n + j] = gQBranch[j];
        // Merged: both branches within 0.05 rad on every arm joint.
        if (lane == 0)
        {
            float gap = 0.0;
            for (uint j = 0; j < n; j++)
                if (dofArm[j] == arm)
                    gap = max(gap, abs(gQBranch[j] - gQMain[j]));
            gKeep = gap < 0.05 ? 1 : 0;
        }
        GroupMemoryBarrierWithGroupSync();
        int p = gBranch[b0 + 1], idle = gBranch[b0 + 2], dir = gBranch[b0 + 3], evidence = gBranch[b0 + 4];
        if (costAlt < branchHysteresis * costMain)
            evidence = max(evidence, 0) + 1;
        else if (costMain < branchHysteresis * costAlt)
            evidence = min(evidence, 0) - 1;
        else
            evidence = 0;
        if (evidence >= int(branchDwell))
            dir = 1;
        else if (evidence <= -int(branchDwell))
            dir = -1;
        p = clamp(p + dir, 0, int(branchBlend));
        bool swapped = uint(p) >= branchBlend;
        idle = p == 0 && dir <= 0 ? idle + 1 : 0;
        float x = float(p) / float(branchBlend);
        weight[arm - 1] = x * x * (3.0 - 2.0 * x);
        swap[arm - 1] = swapped;
        any = true;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0)
        {
            gBranch[b0 + 1] = swapped ? 0 : p;
            gBranch[b0 + 2] = idle;
            gBranch[b0 + 3] = swapped ? -1 : dir;
            gBranch[b0 + 4] = swapped ? 0 : evidence;
            if ((p == 0 && gKeep != 0) || uint(idle) >= branchKeep)
                gBranch[b0] = 0;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!any)
        return false;
    // Output: each arm with two branches between them; a finished blend swaps them.
    saveMain(lane);
    for (uint j = lane; j < n; j += GROUP)
    {
        int arm = dofArm[j];
        if (arm == 0 || weight[arm - 1] < 0.0)
            continue;
        float b = stateBranchQ[s * n + j];
        gQ[j] = lerp(gQMain[j], b, weight[arm - 1]);
        if (swap[arm - 1])
        {
            stateBranchQ[s * n + j] = gQMain[j];
            gQMain[j] = b;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return true;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void solveGroup(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID)
{
    uint s = gid.x;
    uint lane = gtid.x;
    if (s >= numStreams)
        return; // whole group exits together
    uint n = numDofs;
    uint nv = 6 + n;
    uint rowDynamics = 6 * numTargets;
    int offset = streamFrameOffset[s];
    int frames = streamFrameCount[s];
    int history = stateValid[s] != 0 ? stateHistory[s] : 0;

    if (lane == 0)
    {
        if (stateValid[s] != 0)
        {
            gRootPos = stateRootPos[s].xyz;
            gRootRot = stateRootRot[s];
            for (uint j = 0; j < n; j++)
                gQ[j] = stateQ[s * n + j];
        }
        else
        {
            gRootPos = goalPos[offset * numTargets + rootTarget].xyz;
            gRootRot = goalRot[offset * numTargets + rootTarget];
            for (uint j = 0; j < n; j++)
                gQ[j] = 0.0;
        }
    }
    if (lane < 16)
        gBranch[lane] = branchBlend > 0 && stateValid[s] != 0 ? stateBranch[s * 16 + lane] : 0;
    GroupMemoryBarrierWithGroupSync();

    setupShared(lane, nv);

    for (int f = 0; f < frames; f++)
    {
        uint frame = uint(offset + f);
        for (uint j = lane; j < n; j += GROUP)
            gQPrev[j] = gQ[j];
        GroupMemoryBarrierWithGroupSync();

        // A cold stream's first frame starts from q = 0: converge it fully and without the smoothness pull.
        bool cold = f == 0 && stateValid[s] == 0;
        uint frameIterations = cold ? max(iterations, coldIterations) : iterations;
        float frameSmoothness = cold ? 0.0 : smoothness;

        gaussNewton(s, frame, frameIterations, frameSmoothness, cold ? 0 : history, lane, nv, rowDynamics);
        bool blended = branchBlend > 0 && !cold && branchSwitch(s, frame, history, lane, nv, rowDynamics);
        if (blended)
        {
            // gQ holds the output pose now: its link poses and tracking rows for the CoM ring and the cost.
            forwardShared(lane);
            for (uint t = lane; t < numTargets; t += GROUP)
                targetRows(t, frame, nv);
            GroupMemoryBarrierWithGroupSync();
        }

        // Push this frame's link CoMs into the history ring (gPos / gRot hold the final pose).
        for (uint i = lane; i < numLinks; i += GROUP)
            stateLinkCom[(s * HISTORY + uint(history) % HISTORY) * numLinks + i] = float4(gPos[i] + qrotate(gRot[i], linkCom[i].xyz), 0.0);
        GroupMemoryBarrierWithGroupSync();
        history++;
        if (lane == 0)
        {
            float cost = 0.0;
            for (uint r = 0; r < rowDynamics; r++)
                cost += gR[r] * gR[r];
            outRootPos[frame] = float4(gRootPos, 0.0);
            outRootRot[frame] = gRootRot;
            for (uint j = 0; j < n; j++)
                outQ[frame * n + j] = gQ[j];
            outCost[frame] = cost;
        }
        GroupMemoryBarrierWithGroupSync();
        if (blended)
        {
            for (uint j = lane; j < n; j += GROUP)
                gQ[j] = gQMain[j];
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (lane == 0)
    {
        stateHistory[s] = history;
        stateRootPos[s] = float4(gRootPos, 0.0);
        stateRootRot[s] = gRootRot;
        for (uint j = 0; j < n; j++)
            stateQ[s * n + j] = gQ[j];
    }
    if (lane < 16 && branchBlend > 0)
        stateBranch[s * 16 + lane] = gBranch[lane];
}
