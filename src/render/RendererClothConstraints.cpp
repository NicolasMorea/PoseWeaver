#include "render/Renderer.h"

#include "core/Paths.h"

#include <Eigen/Core>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace poseweaver
{
    namespace
    {
        static std::string constraint_file_name_for_garment(const std::string& garment_name)
        {
            std::string out = garment_name;
            for (char& c : out)
            {
                const unsigned char uc = static_cast<unsigned char>(c);
                if (c == '/' || c == '\\' || c == ':' || !std::isprint(uc))
                    c = '_';
            }
            if (out.empty())
                out = "unnamed";
            return out + ".constraints";
        }

        static std::filesystem::path constraint_path_for_garment(const std::string& garment_name)
        {
            return assets_dir() / "constraints" / constraint_file_name_for_garment(garment_name);
        }

        static int count_constraints(const std::vector<unsigned char>& constraints)
        {
            int n = 0;
            for (unsigned char c : constraints)
                if (c)
                    ++n;
            return n;
        }
    }

    int Renderer::garment_constraint_count() const
    {
        return count_constraints(m_garment_constraints);
    }

    void Renderer::load_garment_constraints(AppState& state, int vertex_count)
    {
        m_garment_constraints.assign((size_t)std::max(0, vertex_count), 0);
        m_garment_constraints_mesh_name = m_garment_mesh_name;
        state.cloth_constraint_count = 0;

        if (m_garment_mesh_name.empty() || vertex_count <= 0)
        {
            state.cloth_constraint_status.clear();
            return;
        }

        const std::filesystem::path path = constraint_path_for_garment(m_garment_mesh_name);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
        {
            state.cloth_constraint_status = "No saved constraints";
            return;
        }

        std::ifstream in(path);
        if (!in)
        {
            state.cloth_constraint_status = "Could not read constraints";
            console_add_line(state, "WARNING: Could not read constraint file: " + path.string());
            return;
        }

        int file_vertex_count = -1;
        int loaded = 0;
        std::string line;
        while (std::getline(in, line))
        {
            const auto hash = line.find('#');
            if (hash != std::string::npos)
                line.erase(hash);

            std::istringstream iss(line);
            std::string first;
            if (!(iss >> first))
                continue;

            if (first == "vertex_count")
            {
                iss >> file_vertex_count;
                continue;
            }

            int idx = -1;
            if (first == "v")
                iss >> idx;
            else
            {
                std::istringstream idx_stream(first);
                idx_stream >> idx;
            }

            if (idx >= 0 && idx < vertex_count && !m_garment_constraints[(size_t)idx])
            {
                m_garment_constraints[(size_t)idx] = 1;
                ++loaded;
            }
        }

        if (file_vertex_count >= 0 && file_vertex_count != vertex_count)
        {
            std::fill(m_garment_constraints.begin(), m_garment_constraints.end(), 0);
            state.cloth_constraint_count = 0;
            state.cloth_constraint_status = "Saved constraints ignored: vertex count mismatch";
            console_add_line(state,
                "WARNING: Ignored constraints for " + m_garment_mesh_name +
                " (file #V=" + std::to_string(file_vertex_count) +
                ", cloth #V=" + std::to_string(vertex_count) + ")");
            return;
        }

        state.cloth_constraint_count = loaded;
        state.cloth_constraint_status = "Loaded " + std::to_string(loaded) + " constraints";
        console_add_line(state,
            "Loaded cloth constraints: " + m_garment_mesh_name +
            " (" + std::to_string(loaded) + " vertices)");
    }

    void Renderer::process_cloth_constraints_save(AppState& state)
    {
        if (m_garment_mesh_name.empty())
        {
            state.cloth_constraint_status = "No garment loaded";
            console_add_line(state, "Save constraints skipped: no garment loaded");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() <= 0)
        {
            state.cloth_constraint_status = "Garment mesh is empty";
            console_add_line(state, "Save constraints skipped: garment mesh is empty");
            return;
        }

        if ((int)m_garment_constraints.size() != garment.V.rows())
            m_garment_constraints.assign((size_t)garment.V.rows(), 0);

        const std::filesystem::path dir = assets_dir() / "constraints";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec)
        {
            state.cloth_constraint_status = "Could not create constraints folder";
            console_add_line(state, "ERROR: Could not create constraints folder: " + ec.message());
            return;
        }

        const std::filesystem::path path = constraint_path_for_garment(m_garment_mesh_name);
        std::ofstream out(path, std::ios::trunc);
        if (!out)
        {
            state.cloth_constraint_status = "Could not write constraints";
            console_add_line(state, "ERROR: Could not write constraint file: " + path.string());
            return;
        }

        out << "# PoseWeaver cloth constraints v1\n";
        out << "vertex_count " << garment.V.rows() << "\n";
        for (int i = 0; i < (int)m_garment_constraints.size(); ++i)
            if (m_garment_constraints[(size_t)i])
                out << i << "\n";

        const int count = count_constraints(m_garment_constraints);
        state.cloth_constraint_count = count;
        state.cloth_constraint_status = "Saved " + std::to_string(count) + " constraints";
        console_add_line(state,
            "Saved cloth constraints: " + path.string() +
            " (" + std::to_string(count) + " vertices)");
    }

    void Renderer::process_cloth_constraints_clear(AppState& state)
    {
        std::fill(m_garment_constraints.begin(), m_garment_constraints.end(), 0);
        state.cloth_constraint_count = 0;
        state.cloth_constraint_status = "Constraints cleared";
        update_garment_constraint_visualization(state);
        console_add_line(state, "Cleared cloth constraints for current garment");
    }

    void Renderer::process_cloth_constraint_click(AppState& state)
    {
        if (!state.cloth_constraints_edit_enabled)
            return;
        if (state.cloth_constraint_click_request_id == 0 ||
            state.cloth_constraint_click_request_id == m_last_cloth_constraint_click_request_id)
            return;
        m_last_cloth_constraint_click_request_id = state.cloth_constraint_click_request_id;

        if (m_garment_mesh_name.empty() || !state.show_garment_mesh)
            return;

        auto& garment = m_viewer.data(m_garment_mesh_index);
        const int N = (int)garment.V.rows();
        if (N <= 0)
            return;

        if ((int)m_garment_constraints.size() != N)
            m_garment_constraints.assign((size_t)N, 0);

        auto& core = m_viewer.core();
        const float vx = core.viewport(0);
        const float vy = core.viewport(1);
        const float vw = core.viewport(2);
        const float vh = core.viewport(3);
        if (!(vw > 0.0f) || !(vh > 0.0f))
            return;

        const double radius = std::max(2.0, (double)state.cloth_constraint_pick_radius);
        const double radius2 = radius * radius;
        const int mode = std::clamp(state.cloth_constraint_edit_mode, 0, 2);

        double x0 = (double)state.cloth_constraint_selection_x0;
        double y0 = (double)state.cloth_constraint_selection_y0;
        double x1 = (double)state.cloth_constraint_selection_x1;
        double y1 = (double)state.cloth_constraint_selection_y1;
        double xmin = std::min(x0, x1);
        double xmax = std::max(x0, x1);
        double ymin = std::min(y0, y1);
        double ymax = std::max(y0, y1);
        const double rect_w = xmax - xmin;
        const double rect_h = ymax - ymin;
        const bool point_pick = (rect_w < 3.0 && rect_h < 3.0);

        if (point_pick || mode != 0)
        {
            const double cx = 0.5 * (x0 + x1);
            const double cy = 0.5 * (y0 + y1);
            if (rect_w < 1.0)
            {
                xmin = cx - radius;
                xmax = cx + radius;
            }
            if (rect_h < 1.0)
            {
                ymin = cy - radius;
                ymax = cy + radius;
            }
        }

        std::vector<int> picked;
        picked.reserve(16);
        int nearest = -1;
        double nearest_d2 = std::numeric_limits<double>::infinity();

        const Eigen::Matrix4f mvp = core.proj * core.view;
        for (int vi = 0; vi < N; ++vi)
        {
            const Eigen::Vector4f p(
                (float)garment.V(vi, 0),
                (float)garment.V(vi, 1),
                (float)garment.V(vi, 2),
                1.0f);
            const Eigen::Vector4f clip = mvp * p;
            if (!(clip.w() > 1e-8f))
                continue;

            const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
            if (ndc.z() < -1.0f || ndc.z() > 1.0f)
                continue;

            const double sx = (double)vx + ((double)ndc.x() * 0.5 + 0.5) * (double)vw;
            const double sy = (double)vy + ((double)ndc.y() * 0.5 + 0.5) * (double)vh;
            const double cx = 0.5 * (x0 + x1);
            const double cy = 0.5 * (y0 + y1);
            const double dx = sx - cx;
            const double dy = sy - cy;
            const double d2 = dx * dx + dy * dy;
            if (d2 < nearest_d2)
            {
                nearest_d2 = d2;
                nearest = vi;
            }

            const bool inside_rect = sx >= xmin && sx <= xmax && sy >= ymin && sy <= ymax;
            if ((mode == 0 && point_pick && d2 <= radius2) ||
                (mode == 0 && !point_pick && inside_rect) ||
                (mode != 0 && inside_rect))
            {
                picked.push_back(vi);
            }
        }

        if (picked.empty() && mode == 0 && point_pick && nearest >= 0 && nearest_d2 <= radius2 * 4.0)
            picked.push_back(nearest);
        if (picked.empty())
            return;

        for (int vi : picked)
        {
            unsigned char& c = m_garment_constraints[(size_t)vi];
            if (mode == 1)
                c = 1;
            else if (mode == 2)
                c = 0;
            else
                c = c ? 0 : 1;
        }

        state.cloth_constraint_count = count_constraints(m_garment_constraints);
        state.cloth_constraint_status = "Edited " + std::to_string((int)picked.size()) + " vertices";
        update_garment_constraint_visualization(state);
    }

    void Renderer::update_garment_constraint_visualization(AppState& state)
    {
        if (m_garment_mesh_name.empty())
            return;

        auto& garment = m_viewer.data(m_garment_mesh_index);
        const int N = (int)garment.V.rows();
        if (N <= 0)
            return;

        if ((int)m_garment_constraints.size() != N ||
            m_garment_constraints_mesh_name != m_garment_mesh_name)
        {
            m_garment_constraints.assign((size_t)N, 0);
            m_garment_constraints_mesh_name = m_garment_mesh_name;
        }

        const int count = count_constraints(m_garment_constraints);
        state.cloth_constraint_count = count;

        Eigen::MatrixXd P(count, 3);
        Eigen::MatrixXd C(count, 3);
        int row = 0;
        for (int vi = 0; vi < N; ++vi)
        {
            if (!m_garment_constraints[(size_t)vi])
                continue;
            P.row(row) = garment.V.row(vi);
            C.row(row) = Eigen::RowVector3d(0.05, 0.95, 1.0);
            ++row;
        }

        garment.clear_points();
        if (count > 0)
        {
            garment.point_size = state.cloth_constraints_edit_enabled ? 14.0f : 10.0f;
            garment.show_overlay = 1;
            garment.add_points(P, C);
        }

        if (state.garment_color_by_smpl_uv &&
            m_garment_uv_vertex_colors_valid &&
            m_garment_uv_vertex_colors.rows() == N)
        {
            Eigen::MatrixXd colors = m_garment_uv_vertex_colors;
            for (int vi = 0; vi < N; ++vi)
                if (m_garment_constraints[(size_t)vi])
                    colors.row(vi) = Eigen::RowVector3d(0.05, 0.95, 1.0);
            garment.set_colors(colors);
        }
        else if (state.garment_use_source_texture && m_garment_has_source_texture)
        {
            if (m_garment_has_image_texture)
                garment.set_colors(Eigen::RowVector3d::Ones());
            else
                garment.set_colors(m_garment_source_colors);
        }
        else
        {
            Eigen::MatrixXd colors(N, 3);
            colors.rowwise() = Eigen::RowVector3d(0.97, 0.74, 0.50);
            for (int vi = 0; vi < N; ++vi)
                if (m_garment_constraints[(size_t)vi])
                    colors.row(vi) = Eigen::RowVector3d(0.05, 0.95, 1.0);
            garment.set_colors(colors);
        }
    }
}
