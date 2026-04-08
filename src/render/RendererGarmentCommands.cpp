#include "render/Renderer.h"

#include "simulation/ClothMeshProcessor.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace poseweaver
{
    void Renderer::process_garment_pushout(AppState& state)
    {
        console_add_line(state, "Processing cloth push-out...");

        // Ensure body mesh is evaluated when needed, even if the SMPL UI/panel is hidden.
        if (!m_smplh_loaded || m_smplh_vertices.rows() == 0)
            ensure_smplh_mesh_loaded(state);

        // Need both the body surface and the cloth mesh.
        if (!state.show_garment_mesh)
        {
            console_add_line(state, "Cloth push-out skipped: garment not visible");
            return;
        }
        if (!m_smplh_loaded || m_smplh_vertices.rows() == 0 || m_smplh_model.F.rows() == 0)
        {
            console_add_line(state, "Cloth push-out skipped: SMPL mesh not ready");
            return;
        }
        if (m_garment_mesh_name.empty())
        {
            console_add_line(state, "Cloth push-out skipped: no garment loaded");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
            return;

        Eigen::MatrixXd V = garment.V;

        ClothPushOutResult res;
        std::string err;

        // Minimum cloth-skin distance (in model units).
        const double offset = std::max(0.0, (double)state.garment_min_cloth_skin_distance);
        const bool ok = push_cloth_vertices_outside_body(
            V,
            garment.F,
            m_smplh_vertices,
            m_smplh_model.F,
            offset,
            &res,
            &err);

        if (!ok)
        {
            std::fprintf(stderr, "[PoseWeaver] Cloth push-out failed: %s\n", err.c_str());
            console_add_line(state, std::string("ERROR: Cloth push-out failed: ") + err);
            return;
        }

        if (res.moved_vertices > 0)
        {
            garment.set_vertices(V);
            garment.compute_normals();
            console_add_line(state, std::string("Cloth push-out done: moved ") + std::to_string(res.moved_vertices) + " vertices");
        }
        else
        {
            console_add_line(state, "Cloth push-out done: no vertices moved");
        }
    }

    void Renderer::process_garment_ease(AppState& state)
    {
        console_add_line(state, "Easing cloth...");

        if (!state.show_garment_mesh)
        {
            console_add_line(state, "Ease cloth skipped: garment not visible");
            return;
        }
        if (m_garment_mesh_name.empty())
        {
            console_add_line(state, "Ease cloth skipped: no garment loaded");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
            return;
        if (m_garment_rest_vertices.rows() != garment.V.rows())
        {
            console_add_line(state, "Ease cloth skipped: rest vertices missing/mismatched");
            return;
        }

        Eigen::MatrixXd V = garment.V;
        ClothEaseResult res;
        std::string err;

        const double amount = std::clamp((double)state.garment_ease_amount, 0.0, 1.0);
        const bool ok = ease_cloth_toward_rest_pose(
            V,
            garment.F,
            m_garment_rest_vertices,
            amount,
            &res,
            &err);

        if (!ok)
        {
            std::fprintf(stderr, "[PoseWeaver] Cloth ease failed: %s\n", err.c_str());
            console_add_line(state, std::string("ERROR: Cloth ease failed: ") + err);
            return;
        }

        garment.set_vertices(V);
        garment.compute_normals();
        console_add_line(state,
            std::string("Ease cloth done: edges=") + std::to_string(res.affected_edges) +
            ", bend_edges=" + std::to_string(res.affected_bend_edges) +
            ", max_d=" + std::to_string(res.max_vertex_displacement));
    }

    void Renderer::process_garment_prefit(AppState& state)
    {
        const int steps = std::clamp(state.garment_prefit_steps, 1, 1000);
        console_add_line(state, std::string("Prefit cloth...") + " (steps=" + std::to_string(steps) + ")");

        // Ensure body mesh is evaluated when needed, even if the SMPL UI/panel is hidden.
        if (!m_smplh_loaded || m_smplh_vertices.rows() == 0)
            ensure_smplh_mesh_loaded(state);

        if (!state.show_garment_mesh)
        {
            console_add_line(state, "Prefit skipped: garment not visible");
            return;
        }
        if (!m_smplh_loaded || m_smplh_vertices.rows() == 0 || m_smplh_model.F.rows() == 0)
        {
            console_add_line(state, "Prefit skipped: SMPL mesh not ready");
            return;
        }
        if (m_garment_mesh_name.empty())
        {
            console_add_line(state, "Prefit skipped: no garment loaded");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
            return;
        if (m_garment_rest_vertices.rows() != garment.V.rows())
        {
            console_add_line(state, "Prefit skipped: rest vertices missing/mismatched");
            return;
        }

        Eigen::MatrixXd V = garment.V;
        const double ease_amount = std::clamp((double)state.garment_ease_amount, 0.0, 1.0);
        const double offset = std::max(0.0, (double)state.garment_min_cloth_skin_distance);

        int total_moved = 0;
        double max_d = 0.0;

        for (int si = 0; si < steps; ++si)
        {
            {
                ClothEaseResult eres;
                std::string err;
                const bool ok = ease_cloth_toward_rest_pose(V, garment.F, m_garment_rest_vertices, ease_amount, &eres, &err);
                if (!ok)
                {
                    std::fprintf(stderr, "[PoseWeaver] Cloth prefit ease failed: %s\n", err.c_str());
                    console_add_line(state, std::string("ERROR: Prefit (ease) failed: ") + err);
                    return;
                }
                max_d = std::max(max_d, eres.max_vertex_displacement);
            }

            {
                ClothPushOutResult pres;
                std::string err;
                const bool ok = push_cloth_vertices_outside_body(V, garment.F, m_smplh_vertices, m_smplh_model.F, offset, &pres, &err);
                if (!ok)
                {
                    std::fprintf(stderr, "[PoseWeaver] Cloth prefit push-out failed: %s\n", err.c_str());
                    console_add_line(state, std::string("ERROR: Prefit (push) failed: ") + err);
                    return;
                }
                total_moved += pres.moved_vertices;
            }
        }

        garment.set_vertices(V);
        garment.compute_normals();
        console_add_line(state,
            std::string("Prefit done: total_pushed=") + std::to_string(total_moved) +
            ", max_ease_d=" + std::to_string(max_d));
    }

    void Renderer::process_garment_relax(AppState& state)
    {
        console_add_line(state, "Relaxing cloth (mild)...");

        if (!state.show_garment_mesh)
        {
            console_add_line(state, "Relax skipped: garment not visible");
            return;
        }
        if (m_garment_mesh_name.empty())
        {
            console_add_line(state, "Relax skipped: no garment loaded");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
            return;
        if (m_garment_rest_vertices.rows() != garment.V.rows())
        {
            console_add_line(state, "Relax skipped: rest vertices missing/mismatched");
            return;
        }

        Eigen::MatrixXd V = garment.V;
        ClothRelaxResult res;
        std::string err;

        // Keep this intentionally subtle.
        constexpr int kIters = 6;
        constexpr double kEdgeAlpha = 0.15;
        constexpr double kLambda = 0.02;

        const bool ok = relax_cloth_preserve_stretch(
            V,
            garment.F,
            m_garment_rest_vertices,
            kIters,
            kEdgeAlpha,
            kLambda,
            &res,
            &err);

        if (!ok)
        {
            std::fprintf(stderr, "[PoseWeaver] Cloth mild relax failed: %s\n", err.c_str());
            console_add_line(state, std::string("ERROR: Cloth mild relax failed: ") + err);
            return;
        }

        garment.set_vertices(V);
        garment.compute_normals();
        console_add_line(state,
            std::string("Relax done: iters=") + std::to_string(res.iterations_done) +
            ", edges=" + std::to_string(res.edges_applied) +
            ", max_d=" + std::to_string(res.max_vertex_displacement));
    }
}
