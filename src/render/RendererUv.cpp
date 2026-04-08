#include "render/Renderer.h"
#include "render/RendererInternal.h"
#include "core/Paths.h"

#include <glad/glad.h>
#include <igl/readOBJ.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace poseweaver
{
    bool Renderer::try_load_smpl_uv_layout_once()
    {
        if (m_smpl_uv_layout_loaded)
            return true;

        const std::filesystem::path base = assets_dir() / "smplh";
        const std::filesystem::path g = base / ((m_smplh_loaded_gender == SmplType::Male) ? "male" : "female");

        const std::filesystem::path candidates[] = {
            base / "uv_layout.obj",
            base / "smpl_uv" / "smpl_uv.obj",
            g / "uv_layout.obj",
            g / "model.obj",
            g / "model_with_uv.obj",
        };

        std::filesystem::path uv_obj_path;
        for (const auto& p : candidates)
        {
            if (std::filesystem::exists(p))
            {
                uv_obj_path = p;
                break;
            }
        }
        if (uv_obj_path.empty())
            return false;

        Eigen::MatrixXd V_obj;
        Eigen::MatrixXd UV_obj;
        Eigen::MatrixXd N_obj;
        Eigen::MatrixXi F_obj;
        Eigen::MatrixXi Fuv_obj;
        Eigen::MatrixXi Fn_obj;

        const bool ok = igl::readOBJ(uv_obj_path.string(), V_obj, UV_obj, N_obj, F_obj, Fuv_obj, Fn_obj);
        if (!ok)
            return false;

        if (F_obj.cols() != 3)
            return false;
        if (F_obj.rows() != m_smplh_model.F.rows())
            return false;
        if (Fuv_obj.rows() != m_smplh_model.F.rows() || Fuv_obj.cols() != 3)
            return false;

        if (UV_obj.cols() < 2 || UV_obj.rows() == 0)
            return false;

        const Eigen::MatrixXi& F_model = m_smplh_model.F;
        Eigen::MatrixXi Fuv_mapped(F_model.rows(), 3);

        struct Key
        {
            int a, b, c;
            bool operator==(const Key& o) const { return a == o.a && b == o.b && c == o.c; }
        };
        struct KeyHash
        {
            size_t operator()(const Key& k) const noexcept
            {
                return (size_t)k.a * 73856093u ^ (size_t)k.b * 19349663u ^ (size_t)k.c * 83492791u;
            }
        };

        auto make_key_sorted = [](int i0, int i1, int i2) -> Key
        {
            std::array<int, 3> s = { i0, i1, i2 };
            std::sort(s.begin(), s.end());
            return Key{ s[0], s[1], s[2] };
        };

        std::unordered_map<Key, int, KeyHash> face_map;
        face_map.reserve((size_t)F_obj.rows() * 2u);
        for (int i = 0; i < F_obj.rows(); ++i)
        {
            face_map.emplace(make_key_sorted(F_obj(i, 0), F_obj(i, 1), F_obj(i, 2)), i);
        }

        for (int i = 0; i < F_model.rows(); ++i)
        {
            const int ma = F_model(i, 0);
            const int mb = F_model(i, 1);
            const int mc = F_model(i, 2);

            const auto it = face_map.find(make_key_sorted(ma, mb, mc));
            if (it == face_map.end())
                return false;

            const int oi = it->second;
            const int oa = F_obj(oi, 0);
            const int ob = F_obj(oi, 1);
            const int oc = F_obj(oi, 2);
            const int ta = Fuv_obj(oi, 0);
            const int tb = Fuv_obj(oi, 1);
            const int tc = Fuv_obj(oi, 2);

            auto pick_uv = [&](int mv) -> int
            {
                if (oa == mv) return ta;
                if (ob == mv) return tb;
                if (oc == mv) return tc;
                return -1;
            };

            Fuv_mapped(i, 0) = pick_uv(ma);
            Fuv_mapped(i, 1) = pick_uv(mb);
            Fuv_mapped(i, 2) = pick_uv(mc);
            if (Fuv_mapped(i, 0) < 0 || Fuv_mapped(i, 1) < 0 || Fuv_mapped(i, 2) < 0)
                return false;
        }

        m_smpl_uv_coords = UV_obj.leftCols(2);
        m_smpl_F_uv = std::move(Fuv_mapped);
        m_smpl_uv_layout_loaded = true;
        m_smpl_uv_layout_source = uv_obj_path.string();

        // UV mapping changed -> invalidate cached UV-derived vertex colors.
        m_smpl_uv_vertex_colors_valid = false;
        m_smpl_uv_per_vertex.resize(0, 0);
        m_garment_uv_overlay_valid = false;
        m_garment_uv_vertex_colors_valid = false;
        return true;
    }

    void Renderer::ensure_smpl_uv_vertex_colors()
    {
        if (m_smpl_uv_vertex_colors_valid)
            return;
        if (!m_smplh_loaded)
            return;

        // Ensure we have UV layout if available.
        if (!m_smpl_uv_layout_loaded)
            (void)try_load_smpl_uv_layout_once();

        const int Vcount = m_smplh_model.vertex_count();
        if (Vcount <= 0)
            return;

        Eigen::MatrixXd uv_per_vertex(Vcount, 2);
        uv_per_vertex.setZero();
        Eigen::VectorXi counts(Vcount);
        counts.setZero();

        const bool have_layout = m_smpl_uv_layout_loaded &&
                                (m_smpl_F_uv.rows() == m_smplh_model.F.rows()) &&
                                (m_smpl_uv_coords.rows() > 0);

        if (have_layout)
        {
            const auto& F = m_smplh_model.F;
            for (int i = 0; i < F.rows(); ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    const int vi = F(i, j);
                    const int ti = m_smpl_F_uv(i, j);
                    if (vi < 0 || vi >= Vcount)
                        continue;
                    if (ti < 0 || ti >= m_smpl_uv_coords.rows())
                        continue;

                    uv_per_vertex(vi, 0) += m_smpl_uv_coords(ti, 0);
                    uv_per_vertex(vi, 1) += m_smpl_uv_coords(ti, 1);
                    counts(vi) += 1;
                }
            }
            for (int i = 0; i < Vcount; ++i)
            {
                const int c = counts(i);
                if (c > 0)
                    uv_per_vertex.row(i) /= (double)c;
            }
        }
        else
        {
            // Fallback: planar UVs derived from the template mesh (not a true unwrap).
            const auto& Vref = m_smplh_model.v_template;
            double minx = 0, maxx = 0, minz = 0, maxz = 0;
            if (Vcount > 0)
            {
                minx = maxx = Vref(0, 0);
                minz = maxz = Vref(0, 2);
                for (int i = 1; i < Vcount; ++i)
                {
                    minx = std::min(minx, Vref(i, 0));
                    maxx = std::max(maxx, Vref(i, 0));
                    minz = std::min(minz, Vref(i, 2));
                    maxz = std::max(maxz, Vref(i, 2));
                }
            }
            const double rx = std::max(1e-12, maxx - minx);
            const double rz = std::max(1e-12, maxz - minz);
            for (int i = 0; i < Vcount; ++i)
            {
                uv_per_vertex(i, 0) = (Vref(i, 0) - minx) / rx;
                uv_per_vertex(i, 1) = (Vref(i, 2) - minz) / rz;
            }
        }

        // Cache UVs per (body) vertex too (useful for cloth mapping fallback paths).
        m_smpl_uv_per_vertex = uv_per_vertex;

        m_smpl_uv_vertex_colors.resize(Vcount, 3);
        for (int i = 0; i < Vcount; ++i)
        {
            const double u = uv_per_vertex(i, 0);
            const double v = uv_per_vertex(i, 1);
            m_smpl_uv_vertex_colors.row(i) = render_internal::uv_to_rgb01(u, v);
        }

        m_smpl_uv_vertex_colors_valid = true;
    }

    void Renderer::regenerate_uv_wireframe_segments()
    {
        m_smpl_uv_segments.clear();

        if (!m_smplh_loaded || m_smplh_model.F.rows() == 0)
            return;

        const Eigen::MatrixXi& F = m_smplh_model.F;
        const int face_count = (int)F.rows();
        m_smpl_uv_segments.reserve((size_t)face_count * 3u);

        // If a UV layout isn't present yet, allow it to appear later (e.g. user drops file into assets while app runs).
        if (!m_smpl_uv_layout_loaded)
            (void)try_load_smpl_uv_layout_once();

        const bool have_layout = m_smpl_uv_layout_loaded &&
                                (m_smpl_F_uv.rows() == face_count) &&
                                (m_smpl_uv_coords.rows() > 0);

        // Fallback per-vertex planar UVs (x/z normalized) if we don't have an explicit UV layout.
        Eigen::MatrixXd uv_fallback;
        if (!have_layout)
        {
            const auto& Vref = m_smplh_model.v_template;
            const int V = (int)Vref.rows();
            uv_fallback.resize(V, 2);

            double minx = 0, maxx = 0, minz = 0, maxz = 0;
            if (V > 0)
            {
                minx = maxx = Vref(0, 0);
                minz = maxz = Vref(0, 2);
                for (int i = 1; i < V; ++i)
                {
                    minx = std::min(minx, Vref(i, 0));
                    maxx = std::max(maxx, Vref(i, 0));
                    minz = std::min(minz, Vref(i, 2));
                    maxz = std::max(maxz, Vref(i, 2));
                }
            }

            const double rx = std::max(1e-12, maxx - minx);
            const double rz = std::max(1e-12, maxz - minz);
            for (int i = 0; i < V; ++i)
            {
                const double u = (Vref(i, 0) - minx) / rx;
                const double v = (Vref(i, 2) - minz) / rz;
                uv_fallback(i, 0) = u;
                uv_fallback(i, 1) = v;
            }
        }

        auto add_seg = [&](double u0, double v0, double u1, double v1)
        {
            UvSegment s;
            s.u0 = (float)u0;
            s.v0 = (float)v0;
            s.u1 = (float)u1;
            s.v1 = (float)v1;
            m_smpl_uv_segments.push_back(s);
        };

        if (have_layout)
        {
            for (int i = 0; i < face_count; ++i)
            {
                const int t0 = m_smpl_F_uv(i, 0);
                const int t1 = m_smpl_F_uv(i, 1);
                const int t2 = m_smpl_F_uv(i, 2);
                if (t0 < 0 || t1 < 0 || t2 < 0)
                    continue;
                if (t0 >= m_smpl_uv_coords.rows() || t1 >= m_smpl_uv_coords.rows() || t2 >= m_smpl_uv_coords.rows())
                    continue;

                const double u0 = m_smpl_uv_coords(t0, 0);
                const double v0 = m_smpl_uv_coords(t0, 1);
                const double u1 = m_smpl_uv_coords(t1, 0);
                const double v1 = m_smpl_uv_coords(t1, 1);
                const double u2 = m_smpl_uv_coords(t2, 0);
                const double v2 = m_smpl_uv_coords(t2, 1);

                add_seg(u0, v0, u1, v1);
                add_seg(u1, v1, u2, v2);
                add_seg(u2, v2, u0, v0);
            }
        }
        else
        {
            for (int i = 0; i < face_count; ++i)
            {
                const int v0i = F(i, 0);
                const int v1i = F(i, 1);
                const int v2i = F(i, 2);
                if (v0i < 0 || v1i < 0 || v2i < 0)
                    continue;
                if (v0i >= uv_fallback.rows() || v1i >= uv_fallback.rows() || v2i >= uv_fallback.rows())
                    continue;

                const double u0 = uv_fallback(v0i, 0);
                const double v0 = uv_fallback(v0i, 1);
                const double u1 = uv_fallback(v1i, 0);
                const double v1 = uv_fallback(v1i, 1);
                const double u2 = uv_fallback(v2i, 0);
                const double v2 = uv_fallback(v2i, 1);

                add_seg(u0, v0, u1, v1);
                add_seg(u1, v1, u2, v2);
                add_seg(u2, v2, u0, v0);
            }
        }
    }

}
