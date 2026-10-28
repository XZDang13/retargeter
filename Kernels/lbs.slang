// Batched linear blend skinning: one thread per (frame, vertex).
import quat;

uniform StructuredBuffer<float4> restVertex;   // [numVertices] xyz
uniform StructuredBuffer<int> weightJoint;     // [numVertices * influences]
uniform StructuredBuffer<float> weightValue;   // [numVertices * influences]
uniform StructuredBuffer<float4> skinRot;      // [frames * numJoints]  world * inverse(bind)
uniform StructuredBuffer<float4> skinPos;      // [frames * numJoints]
uniform RWStructuredBuffer<float4> outVertex;  // [frames * numVertices]
uniform uint frames;
uniform uint numVertices;
uniform uint numJoints;
uniform uint influences;

[shader("compute")]
[numthreads(256, 1, 1)]
void skin(uint3 tid: SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= frames * numVertices)
        return;
    uint f = index / numVertices;
    uint v = index - f * numVertices;
    float3 rest = restVertex[v].xyz;
    float3 p = float3(0.0);
    for (uint k = 0; k < influences; k++)
    {
        float w = weightValue[v * influences + k];
        if (w == 0.0)
            continue;
        uint j = f * numJoints + uint(weightJoint[v * influences + k]);
        p += w * (qrotate(skinRot[j], rest) + skinPos[j].xyz);
    }
    outVertex[index] = float4(p, 0.0);
}
