// Milestone-0 spike: differentiable forward kinematics of a serial revolute chain.
// Checks that one [Differentiable] function yields values, forward-mode Jacobians and
// reverse-mode gradients identically on every backend.

static const uint MAX_JOINTS = 32;

uniform StructuredBuffer<float4> jointAxis;    // [numJoints] unit rotation axis in the joint frame (xyz)
uniform StructuredBuffer<float4> jointOffset;  // [numJoints + 1] translation parent -> joint; last entry is the end effector
uniform StructuredBuffer<float> q;             // [batch * numJoints] joint angles (rad)
uniform StructuredBuffer<float4> target;       // [batch] end-effector goal (xyz)
uniform RWStructuredBuffer<float> outPos;      // [batch * 3] end-effector position
uniform RWStructuredBuffer<float> outJac;      // [batch * 3 * numJoints] row-major d(pos)/d(q)
uniform RWStructuredBuffer<float> outGrad;     // [batch * numJoints] d(cost)/d(q), cost = 0.5 |pos - goal|^2
uniform uint batch;
uniform uint numJoints;

[Differentiable]
float3x3 axisAngle(no_diff float3 a, float t)
{
    float c = cos(t);
    float s = sin(t);
    float k = 1.0 - c;
    return float3x3(
        c + a.x * a.x * k,       a.x * a.y * k - a.z * s, a.x * a.z * k + a.y * s,
        a.y * a.x * k + a.z * s, c + a.y * a.y * k,       a.y * a.z * k - a.x * s,
        a.z * a.x * k - a.y * s, a.z * a.y * k + a.x * s, c + a.z * a.z * k);
}

[Differentiable]
float3 endEffector(float angles[MAX_JOINTS], no_diff uint n)
{
    float3x3 R = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    float3 p = float3(0.0);
    [MaxIters(MAX_JOINTS)]
    for (uint i = 0; i < n; i++)
    {
        p = p + mul(R, jointOffset[i].xyz);
        R = mul(R, axisAngle(jointAxis[i].xyz, angles[i]));
    }
    return p + mul(R, jointOffset[n].xyz);
}

[Differentiable]
float trackingCost(float angles[MAX_JOINTS], no_diff uint n, no_diff float3 goal)
{
    float3 d = endEffector(angles, n) - goal;
    return 0.5 * dot(d, d);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void chainFK(uint3 tid: SV_DispatchThreadID)
{
    uint b = tid.x;
    if (b >= batch)
        return;

    float angles[MAX_JOINTS];
    for (uint i = 0; i < MAX_JOINTS; i++)
        angles[i] = i < numJoints ? q[b * numJoints + i] : 0.0;

    float3 p = endEffector(angles, numJoints);
    outPos[b * 3 + 0] = p.x;
    outPos[b * 3 + 1] = p.y;
    outPos[b * 3 + 2] = p.z;

    // Jacobian: one forward-mode pass per joint.
    for (uint j = 0; j < numJoints; j++)
    {
        float seed[MAX_JOINTS];
        for (uint i = 0; i < MAX_JOINTS; i++)
            seed[i] = i == j ? 1.0 : 0.0;
        let column = fwd_diff(endEffector)(diffPair(angles, seed), numJoints).d;
        outJac[(b * 3 + 0) * numJoints + j] = column.x;
        outJac[(b * 3 + 1) * numJoints + j] = column.y;
        outJac[(b * 3 + 2) * numJoints + j] = column.z;
    }

    // Gradient: one reverse-mode pass.
    var pair = diffPair(angles);
    bwd_diff(trackingCost)(pair, numJoints, target[b].xyz, 1.0);
    for (uint j = 0; j < numJoints; j++)
        outGrad[b * numJoints + j] = pair.d[j];
}
