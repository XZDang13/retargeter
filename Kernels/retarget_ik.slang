// Retargeting IK, serial variant: one thread per stream (CPU backend). See retarget_ik_common.slang.
import retarget_ik_common;

[shader("compute")]
[numthreads(32, 1, 1)]
void solve(uint3 tid: SV_DispatchThreadID)
{
    uint s = tid.x;
    if (s >= numStreams)
        return;
    uint n = numDofs;
    uint nv = 6 + n;
    int offset = streamFrameOffset[s];
    int frames = streamFrameCount[s];

    float3 rootPos;
    float4 rootRot;
    float q[MAX_DOFS];
    for (uint j = 0; j < MAX_DOFS; j++)
        q[j] = j < n ? stateQ[s * n + j] : 0.0;
    if (stateValid[s] != 0)
    {
        rootPos = stateRootPos[s].xyz;
        rootRot = stateRootRot[s];
    }
    else
    {
        rootPos = goalPos[offset * numTargets + rootTarget].xyz;
        rootRot = goalRot[offset * numTargets + rootTarget];
        for (uint j = 0; j < n; j++)
            q[j] = 0.0;
    }

    Pose pose;
    float H[MAX_VARS * MAX_VARS];
    float g[MAX_VARS];
    int cols[6 + MAX_CHAIN];
    float J[3 * (6 + MAX_CHAIN)];

    for (int f = 0; f < frames; f++)
    {
        uint frame = uint(offset + f);
        float qPrev[MAX_DOFS];
        for (uint j = 0; j < MAX_DOFS; j++)
            qPrev[j] = q[j];
        float cost = 0.0;
        // A cold stream's first frame starts from q = 0: converge it fully and without the smoothness pull.
        bool cold = f == 0 && stateValid[s] == 0;
        uint frameIterations = cold ? max(iterations, coldIterations) : iterations;
        float frameSmoothness = cold ? 0.0 : smoothness;

        for (uint it = 0; it <= frameIterations; it++)
        {
            forward(pose, rootPos, rootRot, q);
            for (uint a = 0; a < nv * MAX_VARS; a++)
                H[a] = 0.0;
            for (uint a = 0; a < nv; a++)
                g[a] = 0.0;
            cost = 0.0;

            for (uint t = 0; t < numTargets; t++)
            {
                int link = targetLink[t];
                float2 w = targetWeight[t];
                float3 p = pose.pos[link];
                uint chain0 = uint(chainStart[t]);
                uint chainN = uint(chainCount[t]);
                uint nc = 6 + chainN;
                for (uint c = 0; c < 6; c++)
                    cols[c] = int(c);
                for (uint c = 0; c < chainN; c++)
                    cols[6 + c] = 6 + linkDof[chainLinks[chain0 + c]];

                for (uint kind = 0; kind < 2; kind++)
                {
                    float weight = kind == 0 ? w.x : w.y;
                    if (weight <= 0.0)
                        continue;
                    float3 r;
                    if (kind == 0)
                    {
                        r = weight * (p - goalPos[frame * numTargets + t].xyz);
                        // root translation, root rotation (w x (p - p_root)), joints (a x (p - o))
                        float3 d = p - rootPos;
                        for (uint row = 0; row < 3; row++)
                        {
                            float3 e = float3(row == 0 ? 1.0 : 0.0, row == 1 ? 1.0 : 0.0, row == 2 ? 1.0 : 0.0);
                            J[row * nc + 0] = weight * e.x;
                            J[row * nc + 1] = weight * e.y;
                            J[row * nc + 2] = weight * e.z;
                        }
                        float3 c0 = cross(float3(1, 0, 0), d), c1 = cross(float3(0, 1, 0), d), c2 = cross(float3(0, 0, 1), d);
                        for (uint row = 0; row < 3; row++)
                        {
                            J[row * nc + 3] = weight * c0[row];
                            J[row * nc + 4] = weight * c1[row];
                            J[row * nc + 5] = weight * c2[row];
                        }
                        for (uint c = 0; c < chainN; c++)
                        {
                            int l = chainLinks[chain0 + c];
                            float3 col = cross(pose.axis[l], p - pose.origin[l]);
                            for (uint row = 0; row < 3; row++)
                                J[row * nc + 6 + c] = weight * col[row];
                        }
                    }
                    else
                    {
                        float4 twist = targetTwist[t];
                        float3 axis = qrotate(pose.rot[link], twist.xyz);
                        r = untwist(weight * qlog(qmul(pose.rot[link], qconj(goalRot[frame * numTargets + t]))), axis, twist.w);
                        for (uint row = 0; row < 3; row++)
                        {
                            J[row * nc + 0] = 0.0;
                            J[row * nc + 1] = 0.0;
                            J[row * nc + 2] = 0.0;
                        }
                        for (uint c = 0; c < 3; c++)
                        {
                            float3 col = untwist(float3(c == 0 ? weight : 0.0, c == 1 ? weight : 0.0, c == 2 ? weight : 0.0), axis, twist.w);
                            for (uint row = 0; row < 3; row++)
                                J[row * nc + 3 + c] = col[row];
                        }
                        for (uint c = 0; c < chainN; c++)
                        {
                            float3 a = untwist(weight * pose.axis[chainLinks[chain0 + c]], axis, twist.w);
                            for (uint row = 0; row < 3; row++)
                                J[row * nc + 6 + c] = a[row];
                        }
                    }
                    cost += dot(r, r);
                    for (uint a = 0; a < nc; a++)
                    {
                        uint ia = uint(cols[a]);
                        g[ia] += J[0 * nc + a] * r.x + J[1 * nc + a] * r.y + J[2 * nc + a] * r.z;
                        for (uint b = 0; b < nc; b++)
                        {
                            uint ib = uint(cols[b]);
                            H[ia * MAX_VARS + ib] += J[0 * nc + a] * J[0 * nc + b] + J[1 * nc + a] * J[1 * nc + b]
                                                     + J[2 * nc + a] * J[2 * nc + b];
                        }
                    }
                }
            }
            if (it == frameIterations)
                break;

            // Regularisers on the joints.
            for (uint j = 0; j < n; j++)
            {
                uint v = 6 + j;
                float2 lim = dofLimit[j];
                float mid = 0.5 * (lim.x + lim.y), half = 0.5 * (lim.y - lim.x) * limitMargin;
                float ws = frameSmoothness * frameSmoothness, wp = posture * posture;
                H[v * MAX_VARS + v] += ws + wp;
                g[v] += ws * (q[j] - qPrev[j]) + wp * q[j];
                float excess = abs(q[j] - mid) - half;
                if (excess > 0.0)
                {
                    float wl = limitWeight * limitWeight;
                    H[v * MAX_VARS + v] += wl;
                    g[v] += wl * excess * sign(q[j] - mid);
                }
            }
            for (uint v = 0; v < nv; v++)
                H[v * MAX_VARS + v] += damping * (1.0 + H[v * MAX_VARS + v]);

            choleskySolve(H, g, nv);

            rootPos += float3(g[0], g[1], g[2]);
            rootRot = normalize(qmul(qexp(float3(g[3], g[4], g[5])), rootRot));
            for (uint j = 0; j < n; j++)
            {
                float2 lim = dofLimit[j];
                q[j] = clamp(q[j] + g[6 + j], lim.x, lim.y);
            }
        }

        outRootPos[frame] = float4(rootPos, 0.0);
        outRootRot[frame] = rootRot;
        for (uint j = 0; j < n; j++)
            outQ[frame * n + j] = q[j];
        outCost[frame] = cost;
    }

    stateRootPos[s] = float4(rootPos, 0.0);
    stateRootRot[s] = rootRot;
    for (uint j = 0; j < n; j++)
        stateQ[s * n + j] = q[j];
}

