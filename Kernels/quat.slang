// Quaternion helpers shared by all kernels. Device layout is float4(x, y, z, w): vector part in xyz,
// scalar in w (host files and streams use w-first; the Swift wrappers convert).
module quat;

[Differentiable]
public float4 qmul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

[Differentiable]
public float3 qrotate(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

[Differentiable]
public float4 qaxisAngle(no_diff float3 axis, float angle)
{
    float h = 0.5 * angle;
    return float4(axis * sin(h), cos(h));
}
