#include "simulation/collision/TriangleBvh.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stack>

namespace poseweaver::collision
{
    namespace
    {
        constexpr int kLeafSize = 8;

        static inline int longest_axis(const Eigen::Vector3d& e)
        {
            if (e.x() >= e.y() && e.x() >= e.z())
                return 0;
            if (e.y() >= e.x() && e.y() >= e.z())
                return 1;
            return 2;
        }
    }

    void TriangleBvh::build(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
    {
        m_nodes.clear();
        m_faces.clear();
        m_bounds = Aabb{};

        if (V.rows() == 0 || F.rows() == 0)
            return;
        if (V.cols() != 3 || F.cols() != 3)
            return;

        m_faces.resize((size_t)F.rows());
        std::iota(m_faces.begin(), m_faces.end(), 0);
        m_nodes.reserve((size_t)F.rows() * 2u);

        (void)build_node(V, F, 0, (int)m_faces.size());
        if (!m_nodes.empty())
            m_bounds = m_nodes[0].box;
    }

    void TriangleBvh::refit(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
    {
        if (m_nodes.empty())
            return;

        std::vector<int> order;
        order.reserve(m_nodes.size());
        std::stack<int> st;
        st.push(0);
        while (!st.empty())
        {
            const int idx = st.top();
            st.pop();
            order.push_back(idx);
            const Node& n = m_nodes[(size_t)idx];
            if (!n.leaf)
            {
                if (n.left >= 0)
                    st.push(n.left);
                if (n.right >= 0)
                    st.push(n.right);
            }
        }
        for (auto it = order.rbegin(); it != order.rend(); ++it)
        {
            Node& n = m_nodes[(size_t)*it];
            if (n.leaf)
            {
                Aabb b;
                for (int i = 0; i < n.count; ++i)
                {
                    const int face = m_faces[(size_t)(n.start + i)];
                    b.expand(triangle_bounds(V, F, face));
                }
                n.box = b;
            }
            else
            {
                Aabb b;
                if (n.left >= 0)
                    b.expand(m_nodes[(size_t)n.left].box);
                if (n.right >= 0)
                    b.expand(m_nodes[(size_t)n.right].box);
                n.box = b;
            }
        }
        m_bounds = m_nodes[0].box;
    }

    Aabb TriangleBvh::triangle_bounds(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int face) const
    {
        Aabb b;
        if (face < 0 || face >= F.rows())
            return b;
        const int i0 = F(face, 0);
        const int i1 = F(face, 1);
        const int i2 = F(face, 2);
        if (i0 < 0 || i1 < 0 || i2 < 0)
            return b;
        if (i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows())
            return b;
        b.expand(V.row(i0));
        b.expand(V.row(i1));
        b.expand(V.row(i2));
        return b;
    }

    Eigen::Vector3d TriangleBvh::triangle_centroid(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int face) const
    {
        if (face < 0 || face >= F.rows())
            return Eigen::Vector3d(0.0, 0.0, 0.0);
        const int i0 = F(face, 0);
        const int i1 = F(face, 1);
        const int i2 = F(face, 2);
        if (i0 < 0 || i1 < 0 || i2 < 0)
            return Eigen::Vector3d(0.0, 0.0, 0.0);
        if (i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows())
            return Eigen::Vector3d(0.0, 0.0, 0.0);
        return (V.row(i0) + V.row(i1) + V.row(i2)) / 3.0;
    }

    int TriangleBvh::build_node(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int start, int count)
    {
        const int node_index = (int)m_nodes.size();
        m_nodes.emplace_back(); // avoid copy
        Node& node = m_nodes.back();
        node.start = start;
        node.count = count;

        Aabb box;
        Aabb cbox;
        for (int i = 0; i < count; ++i)
        {
            const int face = m_faces[(size_t)(start + i)];
            const Aabb tb = triangle_bounds(V, F, face);
            box.expand(tb);
            const Eigen::Vector3d c = triangle_centroid(V, F, face);
            cbox.expand(c);
        }
        node.box = box;

        if (count <= kLeafSize)
        {
            node.leaf = true;
            return node_index;
        }

        const int axis = longest_axis(cbox.extents());
        auto mid_it = m_faces.begin() + start + count / 2;
        std::nth_element(
            m_faces.begin() + start,
            mid_it,
            m_faces.begin() + start + count,
            [&](int fa, int fb)
            {
                return triangle_centroid(V, F, fa)[axis] < triangle_centroid(V, F, fb)[axis];
            });

        const int left_count = count / 2;
        const int right_count = count - left_count;
        if (left_count <= 0 || right_count <= 0)
        {
            node.leaf = true;
            return node_index;
        }

        node.leaf = false;
        node.left = build_node(V, F, start, left_count);
        node.right = build_node(V, F, start + left_count, right_count);

        Aabb merged;
        merged.expand(m_nodes[(size_t)node.left].box);
        merged.expand(m_nodes[(size_t)node.right].box);
        node.box = merged;

        return node_index;
    }

    bool TriangleBvh::closest_triangle(
        const Eigen::Vector3d& p,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        int& out_face,
        Eigen::Vector3d& out_closest,
        double& out_dist2,
        double max_dist2,
        int ignored_vertex) const
    {
        out_face = -1;
        out_closest.setZero();
        out_dist2 = std::numeric_limits<double>::infinity();

        if (m_nodes.empty())
            return false;
        if (V.rows() == 0 || F.rows() == 0)
            return false;

        double best = max_dist2;

        // Contiguous vector stack; traverse near-to-far to tighten 'best' early for pruning.
        std::vector<int> st;
        st.reserve(64);
        st.push_back(0);
        while (!st.empty())
        {
            const int ni = st.back();
            st.pop_back();
            const Node& n = m_nodes[(size_t)ni];

            const double d2 = core::sqr_distance_point_aabb(p, n.box);
            if (d2 > best)
                continue;

            if (n.leaf)
            {
                for (int i = 0; i < n.count; ++i)
                {
                    const int face = m_faces[(size_t)(n.start + i)];
                    if (face < 0 || face >= F.rows())
                        continue;

                    const int i0 = F(face, 0);
                    const int i1 = F(face, 1);
                    const int i2 = F(face, 2);
                    if (i0 < 0 || i1 < 0 || i2 < 0)
                        continue;
                    if (i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows())
                        continue;
                    if (ignored_vertex >= 0 && (i0 == ignored_vertex || i1 == ignored_vertex || i2 == ignored_vertex))
                        continue;

                    const Eigen::Vector3d a = V.row(i0);
                    const Eigen::Vector3d b = V.row(i1);
                    const Eigen::Vector3d c = V.row(i2);
                    const Eigen::Vector3d q = core::closest_point_on_triangle(p, a, b, c, nullptr);
                    const double cd2 = (p - q).squaredNorm();
                    if (cd2 < best)
                    {
                        best = cd2;
                        out_face = face;
                        out_closest = q;
                        out_dist2 = cd2;
                    }
                }
            }
            else
            {
                const int l = n.left;
                const int r = n.right;

                if (l >= 0 && r >= 0)
                {
                    const double dl = core::sqr_distance_point_aabb(p, m_nodes[(size_t)l].box);
                    const double dr = core::sqr_distance_point_aabb(p, m_nodes[(size_t)r].box);

                    // Push farther first so the nearer node is processed next.
                    if (dl < dr)
                    {
                        if (dr <= best)
                            st.push_back(r);
                        if (dl <= best)
                            st.push_back(l);
                    }
                    else
                    {
                        if (dl <= best)
                            st.push_back(l);
                        if (dr <= best)
                            st.push_back(r);
                    }
                }
                else
                {
                    if (l >= 0)
                        st.push_back(l);
                    if (r >= 0)
                        st.push_back(r);
                }
            }
        }

        return out_face >= 0;
    }
}
