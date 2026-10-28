// Batched forward kinematics of a human skeleton: one thread per frame.
import quat;

uniform StructuredBuffer<int> jointParent; // [numJoints], -1 for the root
uniform StructuredBuffer<float4> localRot; // [frames * numJoints]
uniform StructuredBuffer<float4> localPos; // [frames * numJoints]
uniform RWStructuredBuffer<float4> worldRot;
uniform RWStructuredBuffer<float4> worldPos;
uniform uint frames;
uniform uint numJoints;

[shader("compute")]
[numthreads(64, 1, 1)]
void skeletonFK(uint3 tid: SV_DispatchThreadID)
{
    uint f = tid.x;
    if (f >= frames)
        return;
    uint base = f * numJoints;
    for (uint j = 0; j < numJoints; j++)
    {
        int parent = jointParent[j];
        float4 r = localRot[base + j];
        float3 p = localPos[base + j].xyz;
        if (parent >= 0)
        {
            float4 pr = worldRot[base + parent];
            p = worldPos[base + parent].xyz + qrotate(pr, p);
            r = qmul(pr, r);
        }
        worldRot[base + j] = r;
        worldPos[base + j] = float4(p, 0.0);
    }
}
