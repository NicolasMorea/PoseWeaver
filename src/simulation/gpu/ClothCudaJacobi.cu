#include "simulation/gpu/ClothCudaJacobi.h"

#include <cuda_runtime.h>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace poseweaver
{
    namespace
    {
        constexpr int kBlockSize = 128;

        struct Edge
        {
            uint2 vertices;
            float rest;
            float lambda;
        };

        struct Node
        {
            float4 minimum;
            float4 maximum;
            int4 meta;
        };

        template <typename T>
        struct DeviceBuffer
        {
            T* data = nullptr;
            size_t count = 0;

            ~DeviceBuffer()
            {
                if (data)
                    cudaFree(data);
            }

            bool allocate(size_t n, std::string* error)
            {
                count = n;
                if (n == 0)
                    return true;
                const cudaError_t status = cudaMalloc((void**)&data, n * sizeof(T));
                if (status == cudaSuccess)
                    return true;
                if (error)
                    *error = std::string("CUDA allocation failed: ") + cudaGetErrorString(status);
                return false;
            }

            bool upload(const std::vector<T>& values, std::string* error)
            {
                if (!allocate(values.size(), error) || values.empty())
                    return data || values.empty();
                const cudaError_t status = cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
                if (status == cudaSuccess)
                    return true;
                if (error)
                    *error = std::string("CUDA upload failed: ") + cudaGetErrorString(status);
                return false;
            }
        };

        static bool cuda_ok(cudaError_t status, const char* operation, std::string* error)
        {
            if (status == cudaSuccess)
                return true;
            if (error)
                *error = std::string(operation) + ": " + cudaGetErrorString(status);
            return false;
        }

        static bool launch_ok(const char* operation, std::string* error)
        {
            return cuda_ok(cudaGetLastError(), operation, error);
        }

        static int blocks(size_t count)
        {
            return (int)((count + kBlockSize - 1u) / kBlockSize);
        }

        __host__ __device__ static float3 f3(float4 v) { return make_float3(v.x, v.y, v.z); }
        __host__ __device__ static float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
        __host__ __device__ static float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
        __host__ __device__ static float3 mul(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
        __host__ __device__ static float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        __host__ __device__ static float3 cross3(float3 a, float3 b)
        {
            return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        }
        __host__ __device__ static float length2(float3 v) { return dot3(v, v); }
        __host__ __device__ static float3 normalized(float3 v)
        {
            const float length = sqrtf(length2(v));
            return length > 1e-12f ? mul(v, 1.0f / length) : make_float3(0.0f, 1.0f, 0.0f);
        }

        __device__ static void atomic_add3(float4* values, unsigned int i, float3 value)
        {
            atomicAdd(&values[i].x, value.x);
            atomicAdd(&values[i].y, value.y);
            atomicAdd(&values[i].z, value.z);
        }

        __device__ static float box_distance2(float3 p, float4 mn4, float4 mx4)
        {
            const float3 mn = f3(mn4);
            const float3 mx = f3(mx4);
            const float dx = fmaxf(fmaxf(mn.x - p.x, 0.0f), p.x - mx.x);
            const float dy = fmaxf(fmaxf(mn.y - p.y, 0.0f), p.y - mx.y);
            const float dz = fmaxf(fmaxf(mn.z - p.z, 0.0f), p.z - mx.z);
            return dx * dx + dy * dy + dz * dz;
        }

        __device__ static float3 closest_triangle(
            float3 p, float3 a, float3 b, float3 c, float3& bary)
        {
            const float3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
            const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
            if (d1 <= 0.0f && d2 <= 0.0f) { bary = make_float3(1, 0, 0); return a; }
            const float3 bp = sub(p, b);
            const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
            if (d3 >= 0.0f && d4 <= d3) { bary = make_float3(0, 1, 0); return b; }
            const float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
            {
                const float v = d1 / (d1 - d3);
                bary = make_float3(1 - v, v, 0);
                return add(a, mul(ab, v));
            }
            const float3 cp = sub(p, c);
            const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
            if (d6 >= 0.0f && d5 <= d6) { bary = make_float3(0, 0, 1); return c; }
            const float vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
            {
                const float w = d2 / (d2 - d6);
                bary = make_float3(1 - w, 0, w);
                return add(a, mul(ac, w));
            }
            const float va = d3 * d6 - d5 * d4;
            if (va <= 0.0f && d4 >= d3 && d5 <= d6)
            {
                const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                bary = make_float3(0, 1 - w, w);
                return add(b, mul(sub(c, b), w));
            }
            const float inverse = 1.0f / (va + vb + vc);
            const float v = vb * inverse, w = vc * inverse;
            bary = make_float3(1 - v - w, v, w);
            return add(a, add(mul(ab, v), mul(ac, w)));
        }

        __global__ static void refit_leaves(
            Node* nodes, unsigned int node_count, const float4* positions, const uint4* faces)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= node_count || nodes[i].meta.z < 0)
                return;
            const uint4 face = faces[nodes[i].meta.z];
            const float3 a = f3(positions[face.x]), b = f3(positions[face.y]), c = f3(positions[face.z]);
            nodes[i].minimum = make_float4(
                fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z)), 0);
            nodes[i].maximum = make_float4(
                fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z)), 0);
        }

        __global__ static void refit_level(Node* nodes, unsigned int node_count, int active_depth)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= node_count || nodes[i].meta.z >= 0 || nodes[i].meta.w != active_depth)
                return;
            const Node a = nodes[nodes[i].meta.x], b = nodes[nodes[i].meta.y];
            nodes[i].minimum = make_float4(
                fminf(a.minimum.x, b.minimum.x), fminf(a.minimum.y, b.minimum.y),
                fminf(a.minimum.z, b.minimum.z), 0);
            nodes[i].maximum = make_float4(
                fmaxf(a.maximum.x, b.maximum.x), fmaxf(a.maximum.y, b.maximum.y),
                fmaxf(a.maximum.z, b.maximum.z), 0);
        }

        __global__ static void clear_substep(
            float4* corrections, unsigned int* weights, float4* contacts,
            Edge* stretch, unsigned int stretch_count, Edge* bend, unsigned int bend_count,
            unsigned int vertex_count)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < vertex_count)
            {
                corrections[i] = make_float4(0, 0, 0, 0);
                weights[i] = 0;
                contacts[i] = make_float4(0, 0, 0, 0);
            }
            if (i < stretch_count) stretch[i].lambda = 0;
            if (i < bend_count) bend[i].lambda = 0;
        }

        __global__ static void clear_corrections(float4* corrections, unsigned int* weights, unsigned int count)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < count)
            {
                corrections[i] = make_float4(0, 0, 0, 0);
                weights[i] = 0;
            }
        }

        __global__ static void predict(
            float4* positions, float4* previous, const float4* velocity, const float4* fixed,
            unsigned int count, float dt, float gravity, float3 wind_direction, float wind_force,
            float random_space, float random_time, float simulation_time)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            previous[i] = positions[i];
            if (fixed[i].w > 0.0f)
            {
                positions[i] = make_float4(fixed[i].x, fixed[i].y, fixed[i].z, 0);
                return;
            }
            const float3 p = f3(positions[i]);
            const float spatial = sinf(dot3(p, make_float3(12.9898f, 78.233f, 37.719f)));
            const float temporal = sinf(simulation_time * 1.7f);
            const float scale = fmaxf(0.0f, 1.0f + 0.25f * random_space * spatial + 0.25f * random_time * temporal);
            const float3 acceleration = add(make_float3(0, gravity, 0), mul(wind_direction, wind_force * scale));
            const float3 next = add(p, add(mul(f3(velocity[i]), dt), mul(acceleration, dt * dt)));
            positions[i] = make_float4(next.x, next.y, next.z, 0);
        }

        __global__ static void solve_distances(
            float4* positions, const float* inverse_mass, Edge* edges, unsigned int count,
            float alpha, float4* corrections, unsigned int* weights)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            Edge& edge = edges[i];
            const unsigned int a = edge.vertices.x, b = edge.vertices.y;
            const float3 delta = sub(f3(positions[a]), f3(positions[b]));
            const float length = sqrtf(length2(delta));
            const float wa = inverse_mass[a], wb = inverse_mass[b];
            const float denominator = wa + wb + alpha;
            if (length < 1e-8f || denominator <= 0.0f)
                return;
            const float dlambda = (-(length - edge.rest) - alpha * edge.lambda) / denominator;
            edge.lambda += dlambda;
            const float3 direction = mul(delta, 1.0f / length);
            if (wa > 0.0f)
            {
                atomic_add3(corrections, a, mul(direction, wa * dlambda));
                atomicAdd(&weights[a], 1u);
            }
            if (wb > 0.0f)
            {
                atomic_add3(corrections, b, mul(direction, -wb * dlambda));
                atomicAdd(&weights[b], 1u);
            }
        }

        __device__ static bool nearest_triangle(
            float3 p, unsigned int excluded_vertex, bool exclude_incident,
            const Node* nodes, const float4* positions, const uint4* faces,
            float maximum_distance2, unsigned int& out_face, float3& out_point, float3& out_bary)
        {
            int stack[64];
            int stack_size = 1;
            stack[0] = 0;
            float best = maximum_distance2;
            bool found = false;
            while (stack_size > 0)
            {
                const Node node = nodes[stack[--stack_size]];
                if (box_distance2(p, node.minimum, node.maximum) > best)
                    continue;
                if (node.meta.z >= 0)
                {
                    const uint4 face = faces[node.meta.z];
                    if (exclude_incident && (face.x == excluded_vertex || face.y == excluded_vertex || face.z == excluded_vertex))
                        continue;
                    float3 bary;
                    const float3 point = closest_triangle(
                        p, f3(positions[face.x]), f3(positions[face.y]), f3(positions[face.z]), bary);
                    const float distance2 = length2(sub(p, point));
                    if (distance2 < best)
                    {
                        best = distance2;
                        out_face = (unsigned int)node.meta.z;
                        out_point = point;
                        out_bary = bary;
                        found = true;
                    }
                }
                else if (stack_size < 62)
                {
                    stack[stack_size++] = node.meta.x;
                    stack[stack_size++] = node.meta.y;
                }
            }
            return found;
        }

        __global__ static void solve_body_collisions(
            const float4* positions, const float4* previous, const float* inverse_mass, unsigned int vertex_count,
            const float4* body_positions, const uint4* body_faces, const Node* body_nodes,
            unsigned int body_node_count, float thickness, float alpha,
            float4* corrections, unsigned int* weights, float4* contacts, int* contact_count)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= vertex_count || inverse_mass[i] <= 0.0f || body_node_count == 0)
                return;
            const float3 p = f3(positions[i]);
            const Node root = body_nodes[0];
            const float margin = fmaxf(0.02f, thickness * 4.0f);
            if (p.x < root.minimum.x - margin || p.x > root.maximum.x + margin
                || p.y < root.minimum.y - margin || p.y > root.maximum.y + margin
                || p.z < root.minimum.z - margin || p.z > root.maximum.z + margin)
                return;
            unsigned int face_index = 0;
            float3 point, bary;
            if (!nearest_triangle(
                p, 0, false, body_nodes, body_positions, body_faces,
                margin * margin, face_index, point, bary))
                return;
            const float3 delta = sub(p, point);
            const float distance = sqrtf(length2(delta));
            float3 reference = sub(f3(previous[i]), point);
            if (length2(reference) <= 1e-12f)
                reference = delta;
            const float3 gradient = normalized(reference);
            const float constraint = dot3(delta, gradient) - thickness;
            if (constraint >= 0.0f)
                return;
            const float old_lambda = contacts[i].w;
            const float candidate = old_lambda + (-constraint - alpha * old_lambda) / (inverse_mass[i] + alpha);
            const float new_lambda = fmaxf(0.0f, candidate);
            const float dlambda = new_lambda - old_lambda;
            if (dlambda == 0.0f)
                return;
            contacts[i] = make_float4(gradient.x, gradient.y, gradient.z, new_lambda);
            atomic_add3(corrections, i, mul(gradient, inverse_mass[i] * dlambda));
            atomicAdd(&weights[i], 1u);
            atomicAdd(contact_count, 1);
        }

        __global__ static void solve_self_collisions(
            const float4* positions, const float* inverse_mass, unsigned int vertex_count,
            const uint4* faces, const Node* nodes, unsigned int node_count,
            float thickness, float alpha, float4* corrections, unsigned int* weights, int* contact_count)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= vertex_count || inverse_mass[i] <= 0.0f || node_count == 0 || thickness <= 0.0f)
                return;
            const float3 p = f3(positions[i]);
            unsigned int face_index = 0;
            float3 point, bary;
            if (!nearest_triangle(p, i, true, nodes, positions, faces, thickness * thickness, face_index, point, bary))
                return;
            const float3 delta = sub(p, point);
            const float distance = sqrtf(length2(delta));
            if (distance >= thickness)
                return;
            const uint4 face = faces[face_index];
            const float3 normal = distance > 1e-7f
                ? mul(delta, 1.0f / distance)
                : normalized(cross3(sub(f3(positions[face.y]), f3(positions[face.x])),
                    sub(f3(positions[face.z]), f3(positions[face.x]))));
            const float wv = inverse_mass[i], w0 = inverse_mass[face.x];
            const float w1 = inverse_mass[face.y], w2 = inverse_mass[face.z];
            const float denominator = wv + bary.x * bary.x * w0 + bary.y * bary.y * w1 + bary.z * bary.z * w2 + alpha;
            if (denominator <= 0.0f)
                return;
            const float3 correction = mul(normal, (thickness - distance) / denominator);
            if (wv > 0) { atomic_add3(corrections, i, mul(correction, wv)); atomicAdd(&weights[i], 1u); }
            if (w0 > 0) { atomic_add3(corrections, face.x, mul(correction, -w0 * bary.x)); atomicAdd(&weights[face.x], 1u); }
            if (w1 > 0) { atomic_add3(corrections, face.y, mul(correction, -w1 * bary.y)); atomicAdd(&weights[face.y], 1u); }
            if (w2 > 0) { atomic_add3(corrections, face.z, mul(correction, -w2 * bary.z)); atomicAdd(&weights[face.z], 1u); }
            atomicAdd(contact_count, 1);
        }

        __global__ static void apply_corrections(
            float4* positions, const float4* fixed, const float4* corrections,
            const unsigned int* weights, unsigned int count)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            if (fixed[i].w > 0.0f)
            {
                positions[i] = make_float4(fixed[i].x, fixed[i].y, fixed[i].z, 0);
                return;
            }
            if (weights[i] > 0)
            {
                const float scale = 1.0f / (float)weights[i];
                positions[i].x += corrections[i].x * scale;
                positions[i].y += corrections[i].y * scale;
                positions[i].z += corrections[i].z * scale;
            }
        }

        __global__ static void update_velocity(
            const float4* positions, const float4* previous, float4* velocity,
            const float4* fixed, const float4* contacts, unsigned int count,
            float dt, float damping, float friction)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            if (fixed[i].w > 0.0f)
            {
                velocity[i] = make_float4(0, 0, 0, 0);
                return;
            }
            const float3 previous_velocity = f3(velocity[i]);
            float3 value = mul(sub(f3(positions[i]), f3(previous[i])), (1.0f - damping) / dt);
            const float3 normal = f3(contacts[i]);
            if (length2(normal) > 0.5f)
            {
                const float corrected_vn = dot3(value, normal);
                const float preserved_vn = fmaxf(0.0f, dot3(previous_velocity, normal));
                const float3 tangent = sub(value, mul(normal, corrected_vn));
                value = add(tangent, mul(normal, preserved_vn));
                value = sub(value, mul(tangent, fminf(1.0f, friction)));
            }
            velocity[i] = make_float4(value.x, value.y, value.z, 0);
        }

        static long long edge_key(int a, int b)
        {
            const int lo = std::min(a, b), hi = std::max(a, b);
            return (static_cast<long long>(lo) << 32) | static_cast<unsigned int>(hi);
        }

        static void build_edges(
            const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces,
            double stretch_multiplier, std::vector<Edge>& stretch, std::vector<Edge>& bend)
        {
            struct EdgeFaces { int opposite[2]{ -1, -1 }; int count = 0; };
            std::unordered_map<long long, EdgeFaces> topology;
            topology.reserve((size_t)faces.rows() * 2u);
            for (int fi = 0; fi < faces.rows(); ++fi)
            {
                const int v[3] = { faces(fi, 0), faces(fi, 1), faces(fi, 2) };
                for (int e = 0; e < 3; ++e)
                {
                    const int a = v[e], b = v[(e + 1) % 3], opposite = v[(e + 2) % 3];
                    if (a < 0 || b < 0 || opposite < 0 || a >= vertices.rows() || b >= vertices.rows() || opposite >= vertices.rows())
                        continue;
                    EdgeFaces& entry = topology[edge_key(a, b)];
                    if (entry.count < 2)
                        entry.opposite[entry.count++] = opposite;
                }
            }
            stretch.reserve(topology.size());
            bend.reserve(topology.size());
            for (const auto& item : topology)
            {
                const int a = (int)(item.first >> 32), b = (int)(item.first & 0xffffffffu);
                stretch.push_back(Edge{ make_uint2((unsigned int)a, (unsigned int)b),
                    (float)((vertices.row(a) - vertices.row(b)).norm() * stretch_multiplier), 0 });
                if (item.second.count == 2)
                {
                    const int c = item.second.opposite[0], d = item.second.opposite[1];
                    bend.push_back(Edge{ make_uint2((unsigned int)c, (unsigned int)d),
                        (float)(vertices.row(c) - vertices.row(d)).norm(), 0 });
                }
            }
        }

        static std::vector<float> inverse_masses(
            const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces,
            const std::vector<int>& fixed_vertices)
        {
            std::vector<double> mass((size_t)vertices.rows(), 0.0);
            for (int fi = 0; fi < faces.rows(); ++fi)
            {
                const int a = faces(fi, 0), b = faces(fi, 1), c = faces(fi, 2);
                if (a < 0 || b < 0 || c < 0 || a >= vertices.rows() || b >= vertices.rows() || c >= vertices.rows())
                    continue;
                const Eigen::Vector3d ab = vertices.row(b) - vertices.row(a);
                const Eigen::Vector3d ac = vertices.row(c) - vertices.row(a);
                const double value = 0.5 * ab.cross(ac).norm() / 3.0;
                if (std::isfinite(value))
                    mass[(size_t)a] += value, mass[(size_t)b] += value, mass[(size_t)c] += value;
            }
            double sum = 0.0;
            int valid = 0;
            for (double value : mass)
                if (value > 0.0) sum += value, ++valid;
            const double fallback = valid ? sum / valid : 1.0;
            std::vector<float> result(mass.size());
            for (size_t i = 0; i < mass.size(); ++i)
                result[i] = (float)(1.0 / (mass[i] > 0.0 ? mass[i] : fallback));
            if (!fixed_vertices.empty())
                for (int i : fixed_vertices)
                    if (i >= 0 && i < (int)result.size()) result[(size_t)i] = 0.0f;
            return result;
        }

        static std::vector<float4> positions4(const Eigen::MatrixXd& vertices)
        {
            std::vector<float4> result((size_t)vertices.rows());
            for (int i = 0; i < vertices.rows(); ++i)
                result[(size_t)i] = make_float4((float)vertices(i, 0), (float)vertices(i, 1), (float)vertices(i, 2), 0);
            return result;
        }

        static std::vector<uint4> faces4(const Eigen::MatrixXi& faces)
        {
            std::vector<uint4> result((size_t)faces.rows());
            for (int i = 0; i < faces.rows(); ++i)
                result[(size_t)i] = make_uint4(
                    (unsigned int)std::max(0, faces(i, 0)), (unsigned int)std::max(0, faces(i, 1)),
                    (unsigned int)std::max(0, faces(i, 2)), 0);
            return result;
        }

        static int build_bvh_recursive(
            const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces,
            std::vector<int>& triangles, int begin, int end, int depth,
            std::vector<Node>& nodes, int& maximum_depth)
        {
            const int node_index = (int)nodes.size();
            nodes.push_back(Node{ make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0), make_int4(-1, -1, -1, depth) });
            maximum_depth = std::max(maximum_depth, depth);
            if (end - begin == 1)
            {
                nodes[(size_t)node_index].meta.z = triangles[(size_t)begin];
                return node_index;
            }
            Eigen::Vector3d minimum = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
            Eigen::Vector3d maximum = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
            for (int i = begin; i < end; ++i)
            {
                const int face = triangles[(size_t)i];
                const Eigen::Vector3d center =
                    (vertices.row(faces(face, 0)) + vertices.row(faces(face, 1)) + vertices.row(faces(face, 2))) / 3.0;
                minimum = minimum.cwiseMin(center);
                maximum = maximum.cwiseMax(center);
            }
            Eigen::Index axis = 0;
            (maximum - minimum).maxCoeff(&axis);
            const int middle = begin + (end - begin) / 2;
            std::nth_element(triangles.begin() + begin, triangles.begin() + middle, triangles.begin() + end,
                [&](int lhs, int rhs)
                {
                    const double a = (vertices(faces(lhs, 0), axis) + vertices(faces(lhs, 1), axis) + vertices(faces(lhs, 2), axis)) / 3.0;
                    const double b = (vertices(faces(rhs, 0), axis) + vertices(faces(rhs, 1), axis) + vertices(faces(rhs, 2), axis)) / 3.0;
                    return a < b;
                });
            const int left = build_bvh_recursive(vertices, faces, triangles, begin, middle, depth + 1, nodes, maximum_depth);
            const int right = build_bvh_recursive(vertices, faces, triangles, middle, end, depth + 1, nodes, maximum_depth);
            nodes[(size_t)node_index].meta.x = left;
            nodes[(size_t)node_index].meta.y = right;
            return node_index;
        }

        static std::vector<Node> build_bvh(
            const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces, int& maximum_depth)
        {
            std::vector<Node> nodes;
            if (faces.rows() <= 0)
                return nodes;
            std::vector<int> triangles((size_t)faces.rows());
            std::iota(triangles.begin(), triangles.end(), 0);
            nodes.reserve((size_t)faces.rows() * 2u - 1u);
            maximum_depth = 0;
            build_bvh_recursive(vertices, faces, triangles, 0, faces.rows(), 0, nodes, maximum_depth);
            return nodes;
        }

        static bool refit(
            DeviceBuffer<Node>& nodes, int maximum_depth,
            const DeviceBuffer<float4>& positions, const DeviceBuffer<uint4>& faces,
            std::string* error)
        {
            if (nodes.count == 0)
                return true;
            refit_leaves<<<blocks(nodes.count), kBlockSize>>>(
                nodes.data, (unsigned int)nodes.count, positions.data, faces.data);
            if (!launch_ok("CUDA BVH leaf refit failed", error))
                return false;
            for (int depth = maximum_depth - 1; depth >= 0; --depth)
            {
                refit_level<<<blocks(nodes.count), kBlockSize>>>(nodes.data, (unsigned int)nodes.count, depth);
                if (!launch_ok("CUDA BVH level refit failed", error))
                    return false;
            }
            return true;
        }
    }

    bool cuda_simulation_available(std::string* reason)
    {
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaSuccess && count > 0)
        {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess && properties.major >= 6)
            {
                if (reason)
                    *reason = properties.name;
                return true;
            }
            if (reason)
                *reason = "CUDA GPU must support compute capability 6.0 or newer";
            return false;
        }
        if (reason)
            *reason = status == cudaSuccess ? "No CUDA GPU was found" : cudaGetErrorString(status);
        cudaGetLastError();
        return false;
    }

    bool run_cloth_cuda_jacobi(
        const Eigen::MatrixXd& cloth_V0,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const ClothXpbdParams& params,
        ClothXpbdResult* out,
        std::string* error,
        std::atomic_bool* cancel,
        std::atomic_int* progress_steps_done,
        ClothXpbdOnFrameFn on_frame,
        void* on_frame_user)
    {
        if (out)
            *out = ClothXpbdResult{};
        std::string availability;
        if (!cuda_simulation_available(&availability))
        {
            if (error) *error = "CUDA simulation unavailable: " + availability;
            return false;
        }
        if (cloth_V0.rows() == 0 || cloth_V0.cols() != 3 || cloth_F.rows() == 0 || cloth_F.cols() != 3
            || body_V.rows() == 0 || body_V.cols() != 3 || body_F.rows() == 0 || body_F.cols() != 3)
        {
            if (error) *error = "CUDA simulation requires non-empty triangle cloth and body meshes";
            return false;
        }
        if (!cuda_ok(cudaSetDevice(0), "Could not select the CUDA GPU", error))
            return false;

        const auto started = std::chrono::steady_clock::now();
        const unsigned int vertex_count = (unsigned int)cloth_V0.rows();
        const int substeps = std::max(1, params.substeps);
        const float sub_dt = (float)(params.dt / substeps);
        const int target_steps = params.duration_seconds <= 0.0 ? 0 : (int)std::ceil(params.duration_seconds / params.dt);
        const int steps = std::min(target_steps, std::max(0, params.max_frames - 1));

        std::vector<Edge> stretch, bend;
        build_edges(cloth_V0, cloth_F, std::max(0.0, params.base_stretch_mult), stretch, bend);
        std::vector<float4> cloth_positions = positions4(cloth_V0);
        std::vector<float4> body_positions = positions4(body_V);
        const std::vector<uint4> cloth_faces = faces4(cloth_F);
        const std::vector<uint4> body_faces = faces4(body_F);
        const std::vector<float> masses = inverse_masses(cloth_V0, cloth_F, params.fixed_vertex_indices);
        std::vector<float4> zeros((size_t)vertex_count, make_float4(0, 0, 0, 0));
        std::vector<unsigned int> zero_weights((size_t)vertex_count, 0);
        std::vector<float4> fixed((size_t)vertex_count, make_float4(0, 0, 0, 0));
        int body_depth = 0, cloth_depth = 0;
        const std::vector<Node> body_nodes = build_bvh(body_V, body_F, body_depth);
        const std::vector<Node> cloth_nodes = build_bvh(cloth_V0, cloth_F, cloth_depth);

        DeviceBuffer<float4> d_positions, d_previous, d_velocity, d_body_positions;
        DeviceBuffer<float4> d_corrections, d_fixed, d_contacts;
        DeviceBuffer<float> d_inverse_mass;
        DeviceBuffer<unsigned int> d_weights;
        DeviceBuffer<Edge> d_stretch, d_bend;
        DeviceBuffer<uint4> d_cloth_faces, d_body_faces;
        DeviceBuffer<Node> d_body_nodes, d_cloth_nodes;
        DeviceBuffer<int> d_contact_count;
        if (!d_positions.upload(cloth_positions, error)
            || !d_previous.upload(cloth_positions, error)
            || !d_velocity.upload(zeros, error)
            || !d_body_positions.upload(body_positions, error)
            || !d_corrections.upload(zeros, error)
            || !d_fixed.upload(fixed, error)
            || !d_contacts.upload(zeros, error)
            || !d_inverse_mass.upload(masses, error)
            || !d_weights.upload(zero_weights, error)
            || !d_stretch.upload(stretch, error)
            || !d_bend.upload(bend, error)
            || !d_cloth_faces.upload(cloth_faces, error)
            || !d_body_faces.upload(body_faces, error)
            || !d_body_nodes.upload(body_nodes, error)
            || !d_cloth_nodes.upload(cloth_nodes, error)
            || !d_contact_count.allocate(1, error))
            return false;

        if (!refit(d_body_nodes, body_depth, d_body_positions, d_body_faces, error)
            || !refit(d_cloth_nodes, cloth_depth, d_positions, d_cloth_faces, error))
            return false;

        ClothXpbdResult result;
        result.frames.reserve((size_t)steps + 1u);
        result.frames.push_back(cloth_V0);
        if (on_frame)
            on_frame(result.frames.back(), &body_V, 0, on_frame_user);

        auto update_fixed = [&](int frame) -> bool
        {
            std::fill(fixed.begin(), fixed.end(), make_float4(0, 0, 0, 0));
            if (!params.fixed_vertex_target_sequence.empty())
            {
                const Eigen::MatrixXd& targets = params.fixed_vertex_target_sequence[
                    (size_t)std::max(0, std::min(frame, (int)params.fixed_vertex_target_sequence.size() - 1))];
                for (int row = 0; row < (int)params.fixed_vertex_indices.size() && row < targets.rows(); ++row)
                {
                    const int vertex = params.fixed_vertex_indices[(size_t)row];
                    if (vertex >= 0 && vertex < (int)vertex_count)
                        fixed[(size_t)vertex] = make_float4(
                            (float)targets(row, 0), (float)targets(row, 1), (float)targets(row, 2), 1);
                }
            }
            return cuda_ok(cudaMemcpy(d_fixed.data, fixed.data(), fixed.size() * sizeof(float4), cudaMemcpyHostToDevice),
                "Could not upload CUDA fixed targets", error);
        };

        const float stretch_alpha = (float)(std::max(0.0, params.stretch_compliance) / (sub_dt * sub_dt));
        const float bend_alpha = (float)(std::max(0.0, params.bend_compliance) / (sub_dt * sub_dt));
        const float collision_alpha = (float)(std::max(0.0, params.collision_compliance) / (sub_dt * sub_dt));
        const float self_alpha = (float)(std::max(0.0, params.self_collision_compliance) / (sub_dt * sub_dt));
        float3 wind = make_float3((float)params.wind_dir.x(), (float)params.wind_dir.y(), (float)params.wind_dir.z());
        wind = normalized(wind);
        int last_contacts = 0;

        for (int step = 0; step < steps; ++step)
        {
            if (cancel && cancel->load())
                break;
            const Eigen::MatrixXd* current_body = &body_V;
            if (!params.body_sequence.empty())
                current_body = &params.body_sequence[(size_t)std::min(step, (int)params.body_sequence.size() - 1)];
            body_positions = positions4(*current_body);
            if (!cuda_ok(cudaMemcpy(d_body_positions.data, body_positions.data(),
                    body_positions.size() * sizeof(float4), cudaMemcpyHostToDevice),
                    "Could not upload CUDA body frame", error)
                || !update_fixed(step)
                || !refit(d_body_nodes, body_depth, d_body_positions, d_body_faces, error))
                return false;

            if (!cuda_ok(cudaMemset(d_contact_count.data, 0, sizeof(int)), "Could not reset CUDA contacts", error))
                return false;
            for (int substep = 0; substep < substeps; ++substep)
            {
                const size_t clear_count = std::max({ (size_t)vertex_count, stretch.size(), bend.size() });
                clear_substep<<<blocks(clear_count), kBlockSize>>>(
                    d_corrections.data, d_weights.data, d_contacts.data,
                    d_stretch.data, (unsigned int)stretch.size(),
                    d_bend.data, (unsigned int)bend.size(), vertex_count);
                predict<<<blocks(vertex_count), kBlockSize>>>(
                    d_positions.data, d_previous.data, d_velocity.data, d_fixed.data, vertex_count,
                    sub_dt, (float)params.gravity_y, wind, (float)std::max(0.0, params.wind_force),
                    (float)std::max(0.0, params.wind_random_space), (float)std::max(0.0, params.wind_random_time),
                    (float)(step * params.dt + substep * sub_dt));
                if (!launch_ok("CUDA prediction failed", error))
                    return false;

                const int iterations = std::max(1, std::min(params.solver_iterations, 500));
                for (int iteration = 0; iteration < iterations; ++iteration)
                {
                    clear_corrections<<<blocks(vertex_count), kBlockSize>>>(d_corrections.data, d_weights.data, vertex_count);
                    if (params.stretch_enabled && !stretch.empty())
                        solve_distances<<<blocks(stretch.size()), kBlockSize>>>(
                            d_positions.data, d_inverse_mass.data, d_stretch.data, (unsigned int)stretch.size(),
                            stretch_alpha, d_corrections.data, d_weights.data);
                    if (params.bend_enabled && !bend.empty())
                        solve_distances<<<blocks(bend.size()), kBlockSize>>>(
                            d_positions.data, d_inverse_mass.data, d_bend.data, (unsigned int)bend.size(),
                            bend_alpha, d_corrections.data, d_weights.data);
                    if (params.collision_enabled)
                        solve_body_collisions<<<blocks(vertex_count), kBlockSize>>>(
                            d_positions.data, d_previous.data, d_inverse_mass.data, vertex_count,
                            d_body_positions.data, d_body_faces.data, d_body_nodes.data, (unsigned int)d_body_nodes.count,
                            (float)std::max(0.0, params.collision_thickness), collision_alpha,
                            d_corrections.data, d_weights.data, d_contacts.data, d_contact_count.data);
                    if (params.self_collision_enabled)
                    {
                        if (!refit(d_cloth_nodes, cloth_depth, d_positions, d_cloth_faces, error))
                            return false;
                        solve_self_collisions<<<blocks(vertex_count), kBlockSize>>>(
                            d_positions.data, d_inverse_mass.data, vertex_count,
                            d_cloth_faces.data, d_cloth_nodes.data, (unsigned int)d_cloth_nodes.count,
                            (float)std::max(0.0, params.self_collision_thickness), self_alpha,
                            d_corrections.data, d_weights.data, d_contact_count.data);
                    }
                    apply_corrections<<<blocks(vertex_count), kBlockSize>>>(
                        d_positions.data, d_fixed.data, d_corrections.data, d_weights.data, vertex_count);
                    if (!launch_ok("CUDA Jacobi constraint solve failed", error))
                        return false;
                }
                update_velocity<<<blocks(vertex_count), kBlockSize>>>(
                    d_positions.data, d_previous.data, d_velocity.data, d_fixed.data, d_contacts.data,
                    vertex_count, sub_dt, (float)params.damping,
                    params.enable_friction ? (float)std::max(0.0, params.friction_mu) : 0.0f);
                if (!launch_ok("CUDA velocity update failed", error))
                    return false;
            }

            if (!cuda_ok(cudaMemcpy(cloth_positions.data(), d_positions.data,
                    cloth_positions.size() * sizeof(float4), cudaMemcpyDeviceToHost),
                    "Could not read CUDA cloth frame", error)
                || !cuda_ok(cudaMemcpy(&last_contacts, d_contact_count.data, sizeof(int), cudaMemcpyDeviceToHost),
                    "Could not read CUDA contact count", error))
                return false;
            Eigen::MatrixXd frame(vertex_count, 3);
            for (unsigned int i = 0; i < vertex_count; ++i)
            {
                frame(i, 0) = cloth_positions[i].x;
                frame(i, 1) = cloth_positions[i].y;
                frame(i, 2) = cloth_positions[i].z;
            }
            result.frames.push_back(std::move(frame));
            result.stats.steps_simulated = step + 1;
            result.stats.frames_recorded = (int)result.frames.size();
            result.stats.last_step_contacts = last_contacts;
            if (progress_steps_done)
                progress_steps_done->store(step + 1);
            if (on_frame)
                on_frame(result.frames.back(), current_body, step + 1, on_frame_user);
        }

        if (!cuda_ok(cudaDeviceSynchronize(), "CUDA simulation synchronization failed", error))
            return false;
        result.stats.frames_recorded = (int)result.frames.size();
        result.stats.ms_total = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        if (out)
            *out = std::move(result);
        return true;
    }
}
