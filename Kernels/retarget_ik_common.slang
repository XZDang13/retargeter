// Batched, warm-started Levenberg-Marquardt IK over frames: frames of a stream (clip or teleop stream)
// are solved in order, each initialised from the previous solution (causal, lag = 0).
// Two entry points with identical maths:
//   solveGroup — one 32-thread group per stream, cooperating through groupshared memory (GPU);
//   solve      — one thread per stream (CPU backend, which has no group barriers).
//
// Unknowns per frame: root position (3), root rotation increment in world tangent space (3), joints (n).
// Residuals per target link: wPos * (p - p*) and wRot * log(R R*^T) (world frame). Regularisers:
// smoothness to the previous frame, posture towards q = 0, and a soft barrier beyond the limit margin;
// joints are hard-clamped to their limits after every step. The Jacobian is the analytic geometric one
// (revolute: a x (p - o), a).
// Shared declarations of the retargeting IK (parameters, helpers); entry points live in retarget_ik.slang
// (serial, CPU) and retarget_ik_group.slang (cooperative, GPU). Kept separate because the CPU target
// rejects any module that contains group barriers.
__exported import quat;

static const uint MAX_LINKS = 40;
static const uint MAX_DOFS = 32;
static const uint MAX_VARS = 6 + MAX_DOFS;
static const uint MAX_CHAIN = 16;
static const uint MAX_TARGETS = 24;
static const uint MAX_ROWS = 6 * MAX_TARGETS;
static const uint GROUP = 32;

// Robot model (same layout as robot_fk.slang).
// Global parameters, one parameter block: on Metal an argument buffer (a stage has only 31 buffer slots, the
// solvers bind more); elsewhere a plain uniform struct. The host binds fields by name (rtcompute looks inside).
struct RetargetParams
{
    StructuredBuffer<int> linkParent;
    StructuredBuffer<int> linkDof;
    StructuredBuffer<float4> jointOriginRot;
    StructuredBuffer<float4> jointOriginPos;
    StructuredBuffer<float4> jointToChildRot;
    StructuredBuffer<float4> jointToChildPos;
    StructuredBuffer<float4> jointAxis;
    StructuredBuffer<float2> dofLimit;       // [numDofs] (lower, upper)
    StructuredBuffer<int> targetLink;        // [numTargets]
    StructuredBuffer<float2> targetWeight;   // [numTargets] (pos, rot)
    StructuredBuffer<float4> targetTwist;
    StructuredBuffer<int> chainStart;        // [numTargets] into chainLinks
    StructuredBuffer<int> chainCount;
    StructuredBuffer<int> chainLinks;        // actuated ancestors (incl. self) of each target link
    StructuredBuffer<int> streamFrameOffset; // first frame of each stream in the frame arrays
    StructuredBuffer<int> streamFrameCount;
    StructuredBuffer<float4> goalPos;        // [frame * numTargets]
    StructuredBuffer<float4> goalRot;        // [frame * numTargets] (xyzw)
    RWStructuredBuffer<float4> stateRootPos; // [stream] in/out warm-start state
    RWStructuredBuffer<float4> stateRootRot; // [stream]
    RWStructuredBuffer<float> stateQ;        // [stream * numDofs]
    StructuredBuffer<int> stateValid;        // [stream] 0: initialise from the first frame's root goal
    RWStructuredBuffer<float4> outRootPos;   // [frame]
    RWStructuredBuffer<float4> outRootRot;   // [frame]
    RWStructuredBuffer<float> outQ;          // [frame * numDofs]
    RWStructuredBuffer<float> outCost;       // [frame] final tracking cost
    uint numStreams;
    uint numLinks;
    uint numDofs;
    uint numTargets;
    uint rootTarget;     // index of the target whose goal initialises the root
    uint iterations;
    uint coldIterations;   // iterations for the first frame of a cold stream
    float damping;
    float smoothness;
    float posture;
    float limitMargin;
    float limitWeight;
};
#ifdef __METAL__
ParameterBlock<RetargetParams> ik;
#else
uniform RetargetParams ik;
#endif

// Targets (static description).
// [numTargets] link-frame axis (xyz) and the weight (w, of the rotation weight) of the rotation residual about it:
// a hand target's twist about the forearm, the least reliable part of a captured hand orientation (1: plain).

/// Removes the share 1 - twist.w of a rotation row vector's component along the twist axis `a` (world).
float3 untwist(float3 v, float3 a, float keep) { return v - (1.0 - keep) * dot(a, v) * a; }

// Per stream.


float4 qconj(float4 q) { return float4(-q.xyz, q.w); }

/// Rotation vector (axis * angle) of a unit quaternion.
float3 qlog(float4 q)
{
    if (q.w < 0.0)
        q = -q;
    float s = length(q.xyz);
    if (s < 1e-8)
        return 2.0 * q.xyz;
    return q.xyz * (2.0 * atan2(s, q.w) / s);
}

float4 qexp(float3 w)
{
    float angle = length(w);
    if (angle < 1e-8)
        return normalize(float4(0.5 * w, 1.0));
    return float4(w * (sin(0.5 * angle) / angle), cos(0.5 * angle));
}

struct Pose
{
    float4 rot[MAX_LINKS];
    float3 pos[MAX_LINKS];
    float3 axis[MAX_LINKS];  // world joint axis of actuated links
    float3 origin[MAX_LINKS]; // world joint position of actuated links
};

void forward(inout Pose pose, float3 rootPos, float4 rootRot, float q[MAX_DOFS])
{
    for (uint i = 0; i < ik.numLinks; i++)
    {
        int parent = ik.linkParent[i];
        if (parent < 0)
        {
            pose.rot[i] = rootRot;
            pose.pos[i] = rootPos;
            continue;
        }
        float4 r = pose.rot[parent];
        float3 p = pose.pos[parent] + qrotate(r, ik.jointOriginPos[i].xyz);
        r = qmul(r, ik.jointOriginRot[i]);
        int dof = ik.linkDof[i];
        if (dof >= 0)
        {
            float3 a = ik.jointAxis[i].xyz;
            pose.axis[i] = qrotate(r, a);
            pose.origin[i] = p;
            r = qmul(r, qaxisAngle(a, q[dof]));
        }
        p = p + qrotate(r, ik.jointToChildPos[i].xyz);
        pose.rot[i] = normalize(qmul(r, ik.jointToChildRot[i]));
        pose.pos[i] = p;
    }
}

/// Solves (H) x = -g in place with Cholesky (H symmetric positive definite, n x n, row-major).
void choleskySolve(inout float H[MAX_VARS * MAX_VARS], inout float g[MAX_VARS], uint n)
{
    for (uint j = 0; j < n; j++)
    {
        float d = H[j * MAX_VARS + j];
        for (uint k = 0; k < j; k++)
            d -= H[j * MAX_VARS + k] * H[j * MAX_VARS + k];
        d = sqrt(max(d, 1e-12));
        H[j * MAX_VARS + j] = d;
        for (uint i = j + 1; i < n; i++)
        {
            float s = H[i * MAX_VARS + j];
            for (uint k = 0; k < j; k++)
                s -= H[i * MAX_VARS + k] * H[j * MAX_VARS + k];
            H[i * MAX_VARS + j] = s / d;
        }
    }
    // L y = -g
    for (uint i = 0; i < n; i++)
    {
        float s = -g[i];
        for (uint k = 0; k < i; k++)
            s -= H[i * MAX_VARS + k] * g[k];
        g[i] = s / H[i * MAX_VARS + i];
    }
    // L^T x = y
    for (int i = int(n) - 1; i >= 0; i--)
    {
        float s = g[i];
        for (uint k = uint(i) + 1; k < n; k++)
            s -= H[k * MAX_VARS + uint(i)] * g[k];
        g[i] = s / H[uint(i) * MAX_VARS + uint(i)];
    }
}

