// Batched forward kinematics of a floating-base robot tree (links in topological order).
import quat;

// Static model, uploaded once per robot.
uniform StructuredBuffer<int> linkParent;        // [numLinks], -1 for the root
uniform StructuredBuffer<int> linkDof;           // [numLinks], -1 for fixed links
uniform StructuredBuffer<float4> jointOriginRot; // parent frame -> joint frame at q = 0
uniform StructuredBuffer<float4> jointOriginPos;
uniform StructuredBuffer<float4> jointToChildRot; // joint frame -> child link frame
uniform StructuredBuffer<float4> jointToChildPos;
uniform StructuredBuffer<float4> jointAxis;       // unit axis in the joint frame

// Per sample.
uniform StructuredBuffer<float> q;        // [batch * numDofs]
uniform StructuredBuffer<float4> rootRot; // [batch]
uniform StructuredBuffer<float4> rootPos; // [batch]
uniform RWStructuredBuffer<float4> linkRot; // [batch * numLinks] world
uniform RWStructuredBuffer<float4> linkPos; // [batch * numLinks] world (xyz)
uniform uint batch;
uniform uint numLinks;
uniform uint numDofs;

[shader("compute")]
[numthreads(64, 1, 1)]
void robotFK(uint3 tid: SV_DispatchThreadID)
{
    uint b = tid.x;
    if (b >= batch)
        return;
    uint base = b * numLinks;
    for (uint i = 0; i < numLinks; i++)
    {
        int parent = linkParent[i];
        if (parent < 0)
        {
            linkRot[base + i] = rootRot[b];
            linkPos[base + i] = float4(rootPos[b].xyz, 0.0);
            continue;
        }
        float4 r = linkRot[base + parent];
        float3 p = linkPos[base + parent].xyz;
        // parent * jointOrigin
        p = p + qrotate(r, jointOriginPos[i].xyz);
        r = qmul(r, jointOriginRot[i]);
        // * joint motion
        int dof = linkDof[i];
        if (dof >= 0)
            r = qmul(r, qaxisAngle(jointAxis[i].xyz, q[b * numDofs + dof]));
        // * jointToChild
        p = p + qrotate(r, jointToChildPos[i].xyz);
        r = qmul(r, jointToChildRot[i]);
        linkRot[base + i] = normalize(r);
        linkPos[base + i] = float4(p, 0.0);
    }
}
