#include "render/Renderer.h"
#include "render/RendererInternal.h"
#include "core/Paths.h"
#include "simulation/ClothMeshProcessor.h"

#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace poseweaver
{
    namespace
    {
        static bool triangle_frame(
            const Eigen::RowVector3d& a,
            const Eigen::RowVector3d& b,
            const Eigen::RowVector3d& c,
            Eigen::RowVector3d& T,
            Eigen::RowVector3d& B,
            Eigen::RowVector3d& N)
        {
            const Eigen::RowVector3d e0 = b - a;
            const Eigen::RowVector3d e1 = c - a;

            const double e0n = e0.norm();
            if (!(e0n > 1e-20) || !std::isfinite(e0n))
                return false;

            T = e0 / e0n;
            N = e0.cross(e1);
            const double nn = N.norm();
            if (!(nn > 1e-20) || !std::isfinite(nn))
                return false;
            N /= nn;

            B = N.cross(T);
            const double bn = B.norm();
            if (!(bn > 1e-20) || !std::isfinite(bn))
                return false;
            B /= bn;
            return true;
        }
    }

    void Renderer::ensure_garment_mesh_loaded(AppState& state)
    {
        std::string desired;
        if (state.selection.garment_index >= 0 && state.selection.garment_index < (int)state.selection.garments.size())
            desired = state.selection.garments[(size_t)state.selection.garment_index];

        auto& data = m_viewer.data(m_garment_mesh_index);

        if (desired.empty())
        {
            const bool was_loaded = !m_garment_mesh_name.empty();
            data.clear();
            data.set_visible(false);
            m_garment_mesh_name.clear();
            m_garment_rest_vertices.resize(0, 0);
            m_garment_to_smpl.clear();
            m_garment_mapping_mesh_name.clear();
            m_garment_last_mapping_attempt_mesh_name.clear();
            m_garment_last_mapping_attempt_failed = false;
            m_garment_last_mapping_attempt_error.clear();
            m_garment_uv_segments.clear();
            m_garment_uv_triangles.clear();
            m_garment_uv_overlay_valid = false;
            m_garment_uv_vertex_colors.resize(0, 0);
            m_garment_uv_vertex_colors_valid = false;
            m_last_garment_color_by_smpl_uv = false;
            if (was_loaded)
            {
                m_garment_version++;
                console_add_line(state, "Garment unloaded");
            }
            return;
        }

        const bool mesh_changed = (desired != m_garment_mesh_name);
        if (mesh_changed)
        {
            const std::filesystem::path mesh_path = assets_dir() / "garments" / desired;

            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            const bool ok = igl::read_triangle_mesh(mesh_path.string(), V, F);
            if (!ok)
            {
                data.clear();
                data.set_visible(false);
                m_garment_mesh_name.clear();
                m_garment_to_smpl.clear();
                m_garment_mapping_mesh_name.clear();
                return;
            }

            // Weld garment mesh -> merge duplicate vertices before any mapping or processing
            {
                const int V_before = (int)V.rows();
                const int F_before = (int)F.rows();

                Eigen::MatrixXd V_weld;
                Eigen::MatrixXi F_weld;
                Eigen::VectorXi SVI;
                Eigen::VectorXi SVJ;

                const double epsilon = 0.0025f; // 2.5mm in model units
                igl::remove_duplicate_vertices(V, F, epsilon, V_weld, SVI, SVJ, F_weld);
                V = std::move(V_weld);
                F = std::move(F_weld);

                if (V.rows() != V_before || F.rows() != F_before)
                {
                    console_add_line(state,
                        std::string("Welded garment mesh: #V ") + std::to_string(V_before) + " → " + std::to_string((int)V.rows()) +
                        ", #F " + std::to_string(F_before) + " → " + std::to_string((int)F.rows()));
                }
            }

            data.clear();
            data.set_mesh(V, F);
            data.show_faces = 1;
            data.show_lines = 0;
            data.show_overlay = 0;
            data.set_colors(Eigen::RowVector3d(0.97, 0.74, 0.50)); // Light orange
            data.set_visible(true);

            m_garment_rest_vertices = V;

            m_viewer.core().align_camera_center(V, F);

            m_garment_mesh_name = desired;
            m_garment_version++;
            console_add_line(state,
                std::string("Loaded garment: ") + desired +
                " (#V=" + std::to_string((int)V.rows()) +
                ", #F=" + std::to_string((int)F.rows()) + ")");

            // New garment -> drop any cached mapping/projection.
            m_garment_to_smpl.clear();
            m_garment_mapping_mesh_name.clear();
            m_garment_last_mapping_attempt_mesh_name.clear();
            m_garment_last_mapping_attempt_failed = false;
            m_garment_last_mapping_attempt_error.clear();
            m_garment_uv_segments.clear();
            m_garment_uv_triangles.clear();
            m_garment_uv_overlay_valid = false;
            m_garment_uv_vertex_colors.resize(0, 0);
            m_garment_uv_vertex_colors_valid = false;
            m_last_garment_color_by_smpl_uv = false;
        }
        else
        {
            data.set_visible(true);
        }

        // Ensure garment→SMPL mapping exists (computed in SMPL rest pose) and derived UV overlays/colors are up to date.
        ensure_garment_to_smpl_mapping(state);
        ensure_garment_uv_projection();

        // If a SMPL pose is already selected when the garment loads, the pose will be applied in the main render loop

        // Apply optional SMPL-UV based coloring to the garment.
        if (state.garment_color_by_smpl_uv)
        {
            if (m_garment_uv_vertex_colors_valid && m_garment_uv_vertex_colors.rows() == data.V.rows())
                data.set_colors(m_garment_uv_vertex_colors);
        }
        else if (m_last_garment_color_by_smpl_uv)
        {
            data.set_colors(Eigen::RowVector3d(0.97, 0.74, 0.50)); // Restore default
        }
        m_last_garment_color_by_smpl_uv = state.garment_color_by_smpl_uv;
    }

    void Renderer::ensure_garment_to_smpl_mapping(AppState& state)
    {
        if (m_garment_mesh_name.empty())
        {
            m_garment_to_smpl.clear();
            m_garment_mapping_mesh_name.clear();
            return;
        }

        if (m_garment_rest_vertices.rows() == 0)
            return;

        const SmplType gender = state.selection.gender;
        if (!ensure_smplh_model_loaded(state, gender))
            return;

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
            return;

        // Up to date?
        if (m_garment_to_smpl.valid() &&
            m_garment_mapping_mesh_name == m_garment_mesh_name &&
            m_garment_mapping_gender == gender &&
            m_garment_to_smpl.face_index.size() == garment.V.rows())
        {
            return;
        }

        // don't spam logs if we already failed
        if (m_garment_last_mapping_attempt_failed &&
            m_garment_last_mapping_attempt_mesh_name == m_garment_mesh_name &&
            m_garment_last_mapping_attempt_gender == gender)
        {
            return;
        }

        console_add_line(state,
            std::string("Computing cloth→SMPL mapping for ") + m_garment_mesh_name +
            " (gender=" + ((gender == SmplType::Male) ? std::string("male") : std::string("female")) +
            ", #V=" + std::to_string((int)m_garment_rest_vertices.rows()) + ")");

        ClothToSmplMapping mapping;
        std::string err;

        //* Always map against the SMPL rest shape in T-pose (betas=0, pose=0), not the currently posed/drawn body.
        const bool ok = compute_cloth_to_smpl_mapping(
            m_garment_rest_vertices,
            garment.F,
            m_smplh_model.v_template,
            m_smplh_model.F,
            &mapping,
            &err);

        if (!ok)
        {
            std::fprintf(stderr, "[PoseWeaver] Cloth→SMPL mapping failed: %s\n", err.c_str());
            console_add_line(state, std::string("ERROR: Cloth→SMPL mapping failed: ") + err);
            m_garment_last_mapping_attempt_mesh_name = m_garment_mesh_name;
            m_garment_last_mapping_attempt_gender = gender;
            m_garment_last_mapping_attempt_failed = true;
            m_garment_last_mapping_attempt_error = err;
            m_garment_to_smpl.clear();
            m_garment_mapping_mesh_name.clear();
            m_garment_uv_segments.clear();
            m_garment_uv_triangles.clear();
            m_garment_uv_overlay_valid = false;
            m_garment_uv_vertex_colors.resize(0, 0);
            m_garment_uv_vertex_colors_valid = false;
            return;
        }

        m_garment_to_smpl = std::move(mapping);
        m_garment_mapping_mesh_name = m_garment_mesh_name;
        m_garment_mapping_gender = gender;
        m_garment_last_mapping_attempt_mesh_name = m_garment_mesh_name;
        m_garment_last_mapping_attempt_gender = gender;
        m_garment_last_mapping_attempt_failed = false;
        m_garment_last_mapping_attempt_error.clear();
        m_garment_version++;
        console_add_line(state,
            std::string("Cloth→SMPL mapping ready for ") + m_garment_mesh_name +
            " (#V=" + std::to_string((int)m_garment_to_smpl.face_index.size()) + ")");

        // Mapping changed -> invalidate derived UV data.
        m_garment_uv_segments.clear();
        m_garment_uv_triangles.clear();
        m_garment_uv_overlay_valid = false;
        m_garment_uv_vertex_colors.resize(0, 0);
        m_garment_uv_vertex_colors_valid = false;
    }

    void Renderer::ensure_garment_uv_projection()
    {
        if (!m_garment_to_smpl.valid())
        {
            m_garment_uv_segments.clear();
            m_garment_uv_triangles.clear();
            m_garment_uv_overlay_valid = false;
            m_garment_uv_vertex_colors.resize(0, 0);
            m_garment_uv_vertex_colors_valid = false;
            return;
        }
        if (m_garment_uv_overlay_valid && m_garment_uv_vertex_colors_valid)
            return;

        if (!m_smplh_loaded)
            return;

        // Ensure UV mapping (layout or fallback) is ready.
        ensure_smpl_uv_vertex_colors();
        if (!m_smpl_uv_layout_loaded)
            (void)try_load_smpl_uv_layout_once();

        const bool have_layout = m_smpl_uv_layout_loaded &&
                                (m_smpl_F_uv.rows() == m_smplh_model.F.rows()) &&
                                (m_smpl_uv_coords.rows() > 0);

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
            return;

        const int cloth_V = (int)garment.V.rows();
        m_garment_uv_vertex_colors.resize(cloth_V, 3);

        // Compute UV (u,v) for each cloth vertex by barycentric interpolation over the mapped SMPL face.
        Eigen::MatrixXd cloth_uv(cloth_V, 2);
        cloth_uv.setZero();

        const Eigen::MatrixXi& body_F = m_smplh_model.F;
        for (int vi = 0; vi < cloth_V; ++vi)
        {
            const int fi = m_garment_to_smpl.face_index[vi];
            if (fi < 0 || fi >= body_F.rows())
                continue;

            Eigen::RowVector2d uv0(0.0, 0.0), uv1(0.0, 0.0), uv2(0.0, 0.0);

            if (have_layout)
            {
                const int t0 = m_smpl_F_uv(fi, 0);
                const int t1 = m_smpl_F_uv(fi, 1);
                const int t2 = m_smpl_F_uv(fi, 2);
                if (t0 >= 0 && t1 >= 0 && t2 >= 0 &&
                    t0 < m_smpl_uv_coords.rows() && t1 < m_smpl_uv_coords.rows() && t2 < m_smpl_uv_coords.rows())
                {
                    uv0 = m_smpl_uv_coords.row(t0);
                    uv1 = m_smpl_uv_coords.row(t1);
                    uv2 = m_smpl_uv_coords.row(t2);
                }
            }
            else if (m_smpl_uv_per_vertex.rows() == m_smplh_model.vertex_count())
            {
                const int a = body_F(fi, 0);
                const int b = body_F(fi, 1);
                const int c = body_F(fi, 2);
                if (a >= 0 && b >= 0 && c >= 0 &&
                    a < m_smpl_uv_per_vertex.rows() && b < m_smpl_uv_per_vertex.rows() && c < m_smpl_uv_per_vertex.rows())
                {
                    uv0 = m_smpl_uv_per_vertex.row(a);
                    uv1 = m_smpl_uv_per_vertex.row(b);
                    uv2 = m_smpl_uv_per_vertex.row(c);
                }
            }

            const Eigen::RowVector3d bc = m_garment_to_smpl.barycentric.row(vi);
            const Eigen::RowVector2d uv = bc[0] * uv0 + bc[1] * uv1 + bc[2] * uv2;
            cloth_uv.row(vi) = uv;

            const Eigen::RowVector3d rgb = render_internal::uv_to_rgb01(uv[0], uv[1]);
            m_garment_uv_vertex_colors.row(vi) = rgb;
        }

        m_garment_uv_vertex_colors_valid = true;

        // Build UV-space triangles + wireframe segments for the UV projection view.
        const Eigen::MatrixXi& F = garment.F;
        const int face_count = (int)F.rows();
        m_garment_uv_triangles.clear();
        m_garment_uv_segments.clear();
        m_garment_uv_triangles.reserve((size_t)std::max(0, face_count));
        m_garment_uv_segments.reserve((size_t)std::max(0, face_count) * 3u);

        auto add_seg = [&](double u0, double v0, double u1, double v1)
        {
            UvSegment s;
            s.u0 = (float)u0;
            s.v0 = (float)v0;
            s.u1 = (float)u1;
            s.v1 = (float)v1;
            m_garment_uv_segments.push_back(s);
        };

        for (int fi = 0; fi < face_count; ++fi)
        {
            const int i0 = F(fi, 0);
            const int i1 = F(fi, 1);
            const int i2 = F(fi, 2);
            if (i0 < 0 || i1 < 0 || i2 < 0)
                continue;
            if (i0 >= cloth_uv.rows() || i1 >= cloth_uv.rows() || i2 >= cloth_uv.rows())
                continue;

            const double u0 = cloth_uv(i0, 0);
            const double v0 = cloth_uv(i0, 1);
            const double u1 = cloth_uv(i1, 0);
            const double v1 = cloth_uv(i1, 1);
            const double u2 = cloth_uv(i2, 0);
            const double v2 = cloth_uv(i2, 1);

            UvTriangle t;
            t.u0 = (float)u0; t.v0 = (float)v0;
            t.u1 = (float)u1; t.v1 = (float)v1;
            t.u2 = (float)u2; t.v2 = (float)v2;
            m_garment_uv_triangles.push_back(t);

            add_seg(u0, v0, u1, v1);
            add_seg(u1, v1, u2, v2);
            add_seg(u2, v2, u0, v0);
        }

        m_garment_uv_overlay_valid = true;
    }

    void Renderer::handle_garment_posing(AppState& state, bool force)
    {
        if (!state.show_garment_mesh || m_garment_mesh_name.empty())
            return;
        if (!m_garment_to_smpl.valid())
            return;
        if (m_garment_rest_vertices.rows() == 0)
            return;
        if (!m_smplh_loaded)
            return;

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.F.rows() == 0)
            return;

        const bool pose_selected = (state.selection.pose_index >= 0);
        if (!pose_selected)
        {
            // Restore rest vertices when returning to rest/T-pose selection.
            if (garment.V.rows() == m_garment_rest_vertices.rows() && !garment.V.isApprox(m_garment_rest_vertices, 0.0))
            {
                garment.set_vertices(m_garment_rest_vertices);
                garment.compute_normals();
                m_last_posed_pose_version = m_pose_version;
                m_last_posed_garment_version = m_garment_version;
            }
            return;
        }

        // Only update once per pose/garment change (unless forced by caller).
        if (!force &&
            m_last_posed_pose_version == m_pose_version &&
            m_last_posed_garment_version == m_garment_version)
        {
            return;
        }

        if (m_smplh_vertices.rows() == 0 || m_smplh_vertices.cols() != 3)
            return;
        if (m_smplh_vertices.rows() != m_smplh_model.vertex_count())
            return;
        if (m_garment_to_smpl.face_index.size() != m_garment_rest_vertices.rows())
            return;

        pose_garment_to_smpl(state);
    }

    void Renderer::pose_garment_to_smpl(AppState& state)
    {
        auto& garment = m_viewer.data(m_garment_mesh_index);

        Eigen::MatrixXd V_new(m_garment_rest_vertices.rows(), 3);
        const auto& F_body = m_smplh_model.F;
        const auto& V_rest = m_smplh_model.v_template;
        const auto& V_def = m_smplh_vertices;

        for (int vi = 0; vi < V_new.rows(); ++vi)
        {
            const int fi = m_garment_to_smpl.face_index[vi];
            if (fi < 0 || fi >= F_body.rows())
            {
                V_new.row(vi) = m_garment_rest_vertices.row(vi);
                continue;
            }

            const int ia = F_body(fi, 0);
            const int ib = F_body(fi, 1);
            const int ic = F_body(fi, 2);
            if (ia < 0 || ib < 0 || ic < 0 ||
                ia >= V_def.rows() || ib >= V_def.rows() || ic >= V_def.rows())
            {
                V_new.row(vi) = m_garment_rest_vertices.row(vi);
                continue;
            }

            const Eigen::RowVector3d bc = m_garment_to_smpl.barycentric.row(vi);
            const Eigen::RowVector3d p_def = bc[0] * V_def.row(ia) + bc[1] * V_def.row(ib) + bc[2] * V_def.row(ic);

            // Build local frames in rest and deformed poses
            Eigen::RowVector3d T_def, B_def, N_def;
            if (!triangle_frame(V_def.row(ia), V_def.row(ib), V_def.row(ic), T_def, B_def, N_def))
            {
                // Fallback: use rest frame if deformed triangle is degenerate
                Eigen::RowVector3d T_rest, B_rest, N_rest;
                if (!triangle_frame(V_rest.row(ia), V_rest.row(ib), V_rest.row(ic), T_rest, B_rest, N_rest))
                {
                    V_new.row(vi) = m_garment_rest_vertices.row(vi);
                    continue;
                }
                T_def = T_rest;
                B_def = B_rest;
                N_def = N_rest;
            }

            const Eigen::RowVector3d off_local = m_garment_to_smpl.offset_local.row(vi);
            const Eigen::RowVector3d off_def = off_local[0] * T_def + off_local[1] * B_def + off_local[2] * N_def;
            V_new.row(vi) = p_def + off_def;
        }

        if (garment.V.rows() == V_new.rows())
        {
            garment.set_vertices(V_new);
            garment.compute_normals();
            m_last_posed_pose_version = m_pose_version;
            m_last_posed_garment_version = m_garment_version;

            std::string pose_name;
            if (state.selection.pose_index >= 0 && state.selection.pose_index < (int)state.selection.poses.size())
                pose_name = state.selection.poses[(size_t)state.selection.pose_index];
            else
                pose_name = "(unknown)";
            console_add_line(state,
                std::string("Garment posed from SMPL mapping: ") + m_garment_mesh_name + " ← " + pose_name);
        }
    }

}
