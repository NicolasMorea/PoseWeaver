#include "simulation/SmplSimplifier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace poseweaver
{
    namespace
    {
        struct GridKey
        {
            int region = 0;
            int x = 0;
            int y = 0;
            int z = 0;

            bool operator==(const GridKey& other) const
            {
                return region == other.region && x == other.x && y == other.y && z == other.z;
            }
        };

        struct GridKeyHash
        {
            size_t operator()(const GridKey& k) const
            {
                uint64_t h = 1469598103934665603ull;
                const auto mix = [&](uint64_t v)
                {
                    h ^= v;
                    h *= 1099511628211ull;
                };
                mix((uint64_t)(uint32_t)k.region);
                mix((uint64_t)(uint32_t)k.x);
                mix((uint64_t)(uint32_t)k.y);
                mix((uint64_t)(uint32_t)k.z);
                return (size_t)h;
            }
        };

        static uint64_t face_key(int a, int b, int c)
        {
            int v[3] = { a, b, c };
            std::sort(v, v + 3);
            return ((uint64_t)(uint32_t)v[0] << 42)
                ^ ((uint64_t)(uint32_t)v[1] << 21)
                ^ (uint64_t)(uint32_t)v[2];
        }

        static double safe_extent(double v)
        {
            return (v > 1e-9 && std::isfinite(v)) ? v : 1.0;
        }

        static int vertex_region(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXd& weights,
            int vi,
            const Eigen::RowVector3d& mn,
            const Eigen::RowVector3d& mx)
        {
            const double height = safe_extent(mx[1] - mn[1]);
            const double width = safe_extent(mx[0] - mn[0]);
            const double y01 = (V(vi, 1) - mn[1]) / height;
            const double x_center = 0.5 * (mn[0] + mx[0]);
            const double abs_x01 = std::abs(V(vi, 0) - x_center) / (0.5 * width);

            double hand_w = 0.0;
            double foot_w = 0.0;
            double head_w = 0.0;
            if (weights.rows() == V.rows() && vi < weights.rows())
            {
                const int J = (int)weights.cols();
                const auto w = [&](int j) -> double
                {
                    return (j >= 0 && j < J) ? weights(vi, j) : 0.0;
                };

                foot_w = std::max({ w(7), w(8), w(10), w(11) });
                head_w = std::max(w(12), w(15));
                hand_w = std::max(w(20), w(21));
                for (int j = 22; j < J; ++j)
                    hand_w = std::max(hand_w, w(j));
            }

            if (head_w > 0.35 || y01 > 0.78)
                return 1; // head/face
            if (hand_w > 0.25 || (abs_x01 > 0.82 && y01 > 0.18 && y01 < 0.74))
                return 2; // hands
            if (foot_w > 0.30 || y01 < 0.12)
                return 3; // feet
            return 0;
        }

        static double region_scale(int region)
        {
            switch (region)
            {
            case 1: return 2.75; // head and face can be coarse for cloth collision.
            case 2: return 3.50; // fingers/hands are not important for this workflow.
            case 3: return 2.50; // feet are also aggressively simplified.
            default: return 1.0;
            }
        }

        static int build_clusters_for_cell(
            const Eigen::MatrixXd& V,
            const std::vector<int>& regions,
            const Eigen::RowVector3d& mn,
            double cell,
            std::vector<int>* source_to_cluster = nullptr,
            std::vector<int>* cluster_counts = nullptr)
        {
            std::unordered_map<GridKey, int, GridKeyHash> clusters;
            clusters.reserve((size_t)V.rows());

            if (source_to_cluster)
                source_to_cluster->assign((size_t)V.rows(), -1);
            if (cluster_counts)
                cluster_counts->clear();

            for (int vi = 0; vi < V.rows(); ++vi)
            {
                const int region = regions[(size_t)vi];
                const double local_cell = std::max(1e-9, cell * region_scale(region));
                const GridKey key{
                    region,
                    (int)std::floor((V(vi, 0) - mn[0]) / local_cell),
                    (int)std::floor((V(vi, 1) - mn[1]) / local_cell),
                    (int)std::floor((V(vi, 2) - mn[2]) / local_cell)
                };

                auto it = clusters.find(key);
                int ci = -1;
                if (it == clusters.end())
                {
                    ci = (int)clusters.size();
                    clusters.emplace(key, ci);
                    if (cluster_counts)
                        cluster_counts->push_back(0);
                }
                else
                {
                    ci = it->second;
                }

                if (source_to_cluster)
                    (*source_to_cluster)[(size_t)vi] = ci;
                if (cluster_counts)
                    (*cluster_counts)[(size_t)ci]++;
            }

            return (int)clusters.size();
        }
    }

    bool build_smpl_simplification(
        const Eigen::MatrixXd& reference_V,
        const Eigen::MatrixXi& source_F,
        const Eigen::MatrixXd& skinning_weights,
        double keep_percent,
        SmplSimplification& out,
        std::string* error)
    {
        out = SmplSimplification{};
        if (reference_V.cols() != 3 || reference_V.rows() == 0)
        {
            if (error)
                *error = "Expected SMPL vertices as #V x 3.";
            return false;
        }
        if (source_F.cols() != 3 || source_F.rows() == 0)
        {
            if (error)
                *error = "Expected SMPL faces as #F x 3.";
            return false;
        }

        const int source_n = (int)reference_V.rows();
        keep_percent = std::clamp(keep_percent, 1.0, 100.0);
        const int target_n = std::clamp(
            (int)std::llround((keep_percent / 100.0) * (double)source_n),
            std::min(source_n, 8),
            source_n);

        out.source_vertex_count = source_n;
        out.requested_vertex_count = target_n;

        if (target_n >= source_n || keep_percent >= 99.5)
        {
            out.F = source_F;
            out.source_to_simplified.resize((size_t)source_n);
            out.cluster_counts.assign((size_t)source_n, 1);
            for (int i = 0; i < source_n; ++i)
                out.source_to_simplified[(size_t)i] = i;
            return true;
        }

        const Eigen::RowVector3d mn = reference_V.colwise().minCoeff();
        const Eigen::RowVector3d mx = reference_V.colwise().maxCoeff();
        const double diag = safe_extent((mx - mn).norm());

        std::vector<int> regions((size_t)source_n, 0);
        for (int vi = 0; vi < source_n; ++vi)
            regions[(size_t)vi] = vertex_region(reference_V, skinning_weights, vi, mn, mx);

        double lo = diag * 1e-7;
        double hi = diag;
        for (int i = 0; i < 24; ++i)
        {
            if (build_clusters_for_cell(reference_V, regions, mn, hi) <= target_n)
                break;
            hi *= 2.0;
        }

        double best_cell = hi;
        int best_count = build_clusters_for_cell(reference_V, regions, mn, best_cell);
        for (int iter = 0; iter < 28; ++iter)
        {
            const double mid = 0.5 * (lo + hi);
            const int count = build_clusters_for_cell(reference_V, regions, mn, mid);
            if (count > target_n)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
                best_cell = mid;
                best_count = count;
            }
        }

        (void)best_count;
        build_clusters_for_cell(reference_V, regions, mn, best_cell, &out.source_to_simplified, &out.cluster_counts);

        std::vector<Eigen::Vector3i> faces;
        faces.reserve((size_t)source_F.rows());
        std::unordered_set<uint64_t> seen_faces;
        seen_faces.reserve((size_t)source_F.rows());

        for (int fi = 0; fi < source_F.rows(); ++fi)
        {
            int a = source_F(fi, 0);
            int b = source_F(fi, 1);
            int c = source_F(fi, 2);
            if (a < 0 || b < 0 || c < 0 || a >= source_n || b >= source_n || c >= source_n)
                continue;

            a = out.source_to_simplified[(size_t)a];
            b = out.source_to_simplified[(size_t)b];
            c = out.source_to_simplified[(size_t)c];
            if (a == b || b == c || c == a)
                continue;

            const uint64_t key = face_key(a, b, c);
            if (!seen_faces.insert(key).second)
                continue;
            faces.emplace_back(a, b, c);
        }

        if (faces.empty())
        {
            if (error)
                *error = "SMPL simplification removed all faces.";
            return false;
        }

        out.F.resize((int)faces.size(), 3);
        for (int i = 0; i < (int)faces.size(); ++i)
            out.F.row(i) = faces[(size_t)i];

        return true;
    }

    bool apply_smpl_simplification(
        const Eigen::MatrixXd& source_V,
        const SmplSimplification& simplification,
        Eigen::MatrixXd& out_V,
        std::string* error)
    {
        if (!simplification.valid())
        {
            if (error)
                *error = "Invalid SMPL simplification.";
            return false;
        }
        if (source_V.cols() != 3 || source_V.rows() != simplification.source_vertex_count)
        {
            if (error)
                *error = "SMPL vertex matrix does not match simplification source topology.";
            return false;
        }

        const int out_n = (int)simplification.cluster_counts.size();
        out_V = Eigen::MatrixXd::Zero(out_n, 3);

        for (int vi = 0; vi < source_V.rows(); ++vi)
        {
            const int ci = simplification.source_to_simplified[(size_t)vi];
            if (ci >= 0 && ci < out_n)
                out_V.row(ci) += source_V.row(vi);
        }

        for (int ci = 0; ci < out_n; ++ci)
        {
            const int count = simplification.cluster_counts[(size_t)ci];
            if (count > 0)
                out_V.row(ci) /= (double)count;
        }

        return true;
    }
}
