#include "render/Renderer.h"
#include "render/RendererInternal.h"
#include "core/GlbGarment.h"
#include "core/Geometry.h"
#include "core/ObjMaterialTexture.h"
#include "core/Paths.h"
#include "simulation/ClothMeshProcessor.h"

#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace poseweaver
{
    namespace
    {
        bool is_gltf_path(const std::filesystem::path& path)
        {
            std::string extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return extension == ".glb" || extension == ".gltf";
        }

        void set_rgba_texture(
            igl::opengl::ViewerData& data,
            const std::vector<unsigned char>& rgba,
            int width,
            int height,
            bool flip_y)
        {
            using Channel = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>;
            Channel red(width, height);
            Channel green(width, height);
            Channel blue(width, height);
            Channel alpha(width, height);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const int source_y = flip_y ? height - 1 - y : y;
                    const std::size_t index = (static_cast<std::size_t>(source_y)
                        * static_cast<std::size_t>(width)
                        + static_cast<std::size_t>(x)) * 4u;
                    red(x, y) = rgba[index];
                    green(x, y) = rgba[index + 1u];
                    blue(x, y) = rgba[index + 2u];
                    alpha(x, y) = rgba[index + 3u];
                }
            }
            data.set_texture(red, green, blue, alpha);
        }
    }

    bool Renderer::load_garment_base_pose_axis_angle(
        GarmentBasePose base_pose,
        int joint_count,
        Eigen::VectorXd& pose_axis_angle,
        std::string* error) const
    {
        pose_axis_angle = Eigen::VectorXd::Zero(std::max(0, 3 * joint_count));
        if (base_pose == GarmentBasePose::TPose)
            return true;

        Eigen::VectorXd loaded;
        std::string load_error;
        if (!smplh_load_pose_npz(
                assets_dir() / "poses" / "garment_a_pose.npz",
                loaded,
                nullptr,
                &load_error,
                nullptr)
            || loaded.size() != pose_axis_angle.size())
        {
            if (error)
                *error = load_error.empty()
                    ? "The garment A-pose has an unexpected joint count."
                    : load_error;
            return false;
        }
        pose_axis_angle = std::move(loaded);
        return true;
    }

    void Renderer::ensure_garment_mesh_loaded(AppState& state)
    {
        std::string desired;
        if (state.selection.garment_index >= 0 && state.selection.garment_index < (int)state.selection.garments.size())
            desired = state.selection.garments[(size_t)state.selection.garment_index];
        const std::string desired_material = state.selection.garment_texture.empty()
            ? desired
            : state.selection.garment_texture;
        const bool weld_enabled = state.garment_vertex_merge_enabled;
        const double weld_epsilon = std::max(0.0, (double)state.garment_vertex_merge_distance);

        auto& data = m_viewer.data(m_garment_mesh_index);

        if (desired.empty())
        {
            const bool was_loaded = !m_garment_mesh_name.empty();
            data.clear();
            data.set_visible(false);
            m_garment_mesh_name.clear();
            m_garment_material_name.clear();
            m_garment_has_source_texture = false;
            m_garment_has_image_texture = false;
            m_garment_source_colors.resize(0, 0);
            m_garment_last_show_mesh = false;
            m_garment_vertex_merge_settings_valid = false;
            m_garment_rest_vertices.resize(0, 0);
            m_garment_base_smpl_vertices.resize(0, 0);
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
            m_garment_constraints.clear();
            m_garment_constraints_mesh_name.clear();
            state.cloth_constraint_count = 0;
            state.cloth_constraint_status.clear();
            if (was_loaded)
            {
                m_garment_version++;
                m_garment_material_version++;
                console_add_line(state, "Garment unloaded");
            }
            return;
        }

        const bool settings_changed = m_garment_vertex_merge_settings_valid
            && (weld_enabled != m_garment_vertex_merge_enabled_last
                || std::abs(weld_epsilon - m_garment_vertex_merge_distance_last) > 1e-15
                || state.garment_base_pose != m_garment_base_pose_last);
        const bool mesh_changed = (desired != m_garment_mesh_name) || settings_changed;
        if (mesh_changed)
        {
            const std::filesystem::path mesh_path = assets_dir() / "garments" / desired;
            const std::filesystem::path material_path = assets_dir() / "garments" / desired_material;
            const bool loading_gltf = is_gltf_path(mesh_path);

            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            Eigen::MatrixXd UV;
            Eigen::MatrixXi F_uv;
            GltfGarment gltf;
            ObjMaterialTexture obj_texture;
            std::string load_error;
            const bool ok = loading_gltf
                ? load_gltf_garment(mesh_path, gltf, &load_error)
                : igl::read_triangle_mesh(mesh_path.string(), V, F);
            const bool loaded_gltf_material = loading_gltf && ok && gltf.has_material;
            const bool loaded_gltf_texture = loading_gltf && ok && gltf.has_image_texture();
            const bool loaded_obj_material = !loading_gltf
                && ok
                && load_obj_material_texture(material_path, obj_texture)
                && obj_texture.has_material;
            const bool loaded_obj_texture = loaded_obj_material && obj_texture.has_image_texture();
            if (loading_gltf && ok)
            {
                V = std::move(gltf.vertices);
                F = std::move(gltf.faces);
                UV = std::move(gltf.uv);
                F_uv = std::move(gltf.uv_faces);
            }
            else if (loaded_obj_texture)
            {
                UV = std::move(obj_texture.uv);
                F_uv = std::move(obj_texture.uv_faces);
            }
            if (!ok)
            {
                data.clear();
                data.set_visible(false);
                m_garment_mesh_name.clear();
                m_garment_material_name.clear();
                m_garment_has_source_texture = false;
                m_garment_has_image_texture = false;
                m_garment_source_colors.resize(0, 0);
                m_garment_base_smpl_vertices.resize(0, 0);
                m_garment_to_smpl.clear();
                m_garment_mapping_mesh_name.clear();
                console_add_line(state, "ERROR: Could not load garment " + desired
                    + (load_error.empty() ? std::string() : ": " + load_error));
                return;
            }

            if (weld_enabled && weld_epsilon > 0.0)
            {
                const int V_before = (int)V.rows();
                const int F_before = (int)F.rows();

                Eigen::MatrixXd V_weld;
                Eigen::MatrixXi F_weld;
                Eigen::VectorXi SVI;
                Eigen::VectorXi SVJ;

                igl::remove_duplicate_vertices(V, F, weld_epsilon, V_weld, SVI, SVJ, F_weld);
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
            if (loaded_gltf_texture || loaded_obj_texture)
            {
                data.set_uv(UV, F_uv);
                data.dirty |= igl::opengl::MeshGL::DIRTY_MESH;
                if (loaded_gltf_texture)
                {
                    set_rgba_texture(
                        data,
                        gltf.texture_rgba,
                        gltf.texture_width,
                        gltf.texture_height,
                        false);
                }
                else
                {
                    set_rgba_texture(
                        data,
                        obj_texture.rgba,
                        obj_texture.width,
                        obj_texture.height,
                        true);
                }
            }
            data.show_faces = 1;
            data.show_lines = 0;
            data.show_overlay = 0;
            data.show_texture = 0;
            data.double_sided = loading_gltf && gltf.double_sided;
            data.set_colors(Eigen::RowVector3d(0.97, 0.74, 0.50));
            data.set_visible(true);

            m_viewer.selected_data_index = m_garment_mesh_index;
            m_garment_last_show_mesh = true;

            m_garment_rest_vertices = V;
            m_garment_base_smpl_vertices.resize(0, 0);

            m_viewer.core().align_camera_center(V, F);

            m_garment_mesh_name = desired;
            m_garment_material_name = desired_material;
            m_garment_has_source_texture = loaded_gltf_material || loaded_obj_material;
            m_garment_has_image_texture = loaded_gltf_texture || loaded_obj_texture;
            if (loaded_gltf_material && gltf.face_colors.rows() == F.rows())
                m_garment_source_colors = std::move(gltf.face_colors);
            else if (loaded_obj_material && obj_texture.face_colors.rows() == F.rows())
                m_garment_source_colors = std::move(obj_texture.face_colors);
            else
                m_garment_source_colors = Eigen::MatrixXd::Ones(1, 3);
            m_garment_version++;
            m_garment_material_version++;
            m_garment_vertex_merge_settings_valid = true;
            m_garment_vertex_merge_enabled_last = weld_enabled;
            m_garment_vertex_merge_distance_last = weld_epsilon;
            m_garment_base_pose_last = state.garment_base_pose;
            console_add_line(state,
                std::string("Loaded garment: ") + desired +
                " (#V=" + std::to_string((int)V.rows()) +
                ", #F=" + std::to_string((int)F.rows()) +
                ", base_pose=" + (state.garment_base_pose == GarmentBasePose::APose ? "A" : "T") +
                ", material=" + (m_garment_has_source_texture ? "yes" : "no") +
                ", image_texture=" + (m_garment_has_image_texture ? "yes" : "no") + ")");
            if (!gltf.texture_warning.empty())
                console_add_line(state, "glTF texture warning: " + gltf.texture_warning);
            if (!obj_texture.warning.empty())
                console_add_line(state, "OBJ texture warning: " + obj_texture.warning);

            m_garment_to_smpl.clear();
            m_garment_mapping_mesh_name.clear();
            m_garment_base_smpl_vertices.resize(0, 0);
            m_garment_last_mapping_attempt_mesh_name.clear();
            m_garment_last_mapping_attempt_failed = false;
            m_garment_last_mapping_attempt_error.clear();
            m_garment_uv_segments.clear();
            m_garment_uv_triangles.clear();
            m_garment_uv_overlay_valid = false;
            m_garment_uv_vertex_colors.resize(0, 0);
            m_garment_uv_vertex_colors_valid = false;
            m_last_garment_color_by_smpl_uv = false;
            load_garment_constraints(state, (int)V.rows());
        }
        else
        {
            data.set_visible(true);

            // Coming back from hidden -> re-select garment once so libigl shortcuts apply to it.
            if (!m_garment_last_show_mesh)
                m_viewer.selected_data_index = m_garment_mesh_index;
            m_garment_last_show_mesh = true;
        }

        if (!mesh_changed && desired_material != m_garment_material_name)
        {
            ObjMaterialTexture material;
            const std::filesystem::path material_path = assets_dir() / "garments" / desired_material;
            const bool loaded_material = load_obj_material_texture(material_path, material)
                && material.has_material;
            const bool loaded_texture = loaded_material
                && material.has_image_texture()
                && material.uv_faces.rows() == data.F.rows();
            if (loaded_texture)
            {
                data.set_uv(material.uv, material.uv_faces);
                data.dirty |= igl::opengl::MeshGL::DIRTY_MESH;
                set_rgba_texture(data, material.rgba, material.width, material.height, true);
            }
            m_garment_has_source_texture = loaded_material;
            m_garment_has_image_texture = loaded_texture;
            if (loaded_material && material.face_colors.rows() == data.F.rows())
                m_garment_source_colors = std::move(material.face_colors);
            else
                m_garment_source_colors = Eigen::MatrixXd::Ones(1, 3);
            m_garment_material_name = desired_material;
            m_garment_material_version++;
            console_add_line(state,
                std::string("Changed garment texture: ") + desired_material
                + " (image_texture=" + (loaded_texture ? "yes" : "no") + ")");
            if (!material.warning.empty())
                console_add_line(state, "OBJ texture warning: " + material.warning);
        }

        ensure_garment_to_smpl_mapping(state);
        ensure_garment_uv_projection();
        update_garment_constraint_visualization(state);

        if (state.garment_color_by_smpl_uv)
        {
            data.show_texture = 0;
            if (m_garment_uv_vertex_colors_valid && m_garment_uv_vertex_colors.rows() == data.V.rows())
                data.set_colors(m_garment_uv_vertex_colors);
        }
        else
        {
            const bool use_source_texture = m_garment_has_source_texture
                && state.garment_use_source_texture;
            data.show_texture = use_source_texture && m_garment_has_image_texture ? 1 : 0;
            if (data.show_texture)
                data.set_colors(Eigen::RowVector3d::Ones());
            else if (use_source_texture)
                data.set_colors(m_garment_source_colors);
            else
                data.set_colors(Eigen::RowVector3d(0.97, 0.74, 0.50));
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

        if (m_garment_to_smpl.valid() &&
            m_garment_mapping_mesh_name == m_garment_mesh_name &&
            m_garment_mapping_gender == gender &&
            m_garment_mapping_base_pose == state.garment_base_pose &&
            m_garment_to_smpl.face_index.size() == garment.V.rows())
        {
            return;
        }

        if (m_garment_last_mapping_attempt_failed &&
            m_garment_last_mapping_attempt_mesh_name == m_garment_mesh_name &&
            m_garment_last_mapping_attempt_gender == gender &&
            m_garment_last_mapping_attempt_base_pose == state.garment_base_pose)
        {
            return;
        }

        console_add_line(state,
            std::string("Computing cloth→SMPL mapping for ") + m_garment_mesh_name +
            " (gender=" + ((gender == SmplType::Male) ? std::string("male") : std::string("female")) +
            ", base_pose=" + (state.garment_base_pose == GarmentBasePose::APose ? "A" : "T") +
            ", #V=" + std::to_string((int)m_garment_rest_vertices.rows()) + ")");

        ClothToSmplMapping mapping;
        std::string err;

        Eigen::VectorXd base_pose_axis_angle;
        MatrixX3dRM base_vertices_rm;
        SmplhParams base_params;
        const bool base_pose_loaded = load_garment_base_pose_axis_angle(
            state.garment_base_pose,
            m_smplh_model.joint_count(),
            base_pose_axis_angle,
            &err);
        base_params.pose_axis_angle = base_pose_axis_angle;
        const bool base_pose_evaluated = base_pose_loaded
            && smplh_evaluate_lbs(m_smplh_model, base_params, base_vertices_rm, &err);
        const Eigen::MatrixXd base_vertices = base_vertices_rm;
        const bool ok = base_pose_evaluated && compute_cloth_to_smpl_mapping(
            m_garment_rest_vertices,
            garment.F,
            base_vertices,
            m_smplh_model.F,
            &mapping,
            &err);

        if (!ok)
        {
            std::fprintf(stderr, "[PoseWeaver] Cloth→SMPL mapping failed: %s\n", err.c_str());
            console_add_line(state, std::string("ERROR: Cloth→SMPL mapping failed: ") + err);
            m_garment_last_mapping_attempt_mesh_name = m_garment_mesh_name;
            m_garment_last_mapping_attempt_gender = gender;
            m_garment_last_mapping_attempt_base_pose = state.garment_base_pose;
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
        m_garment_mapping_base_pose = state.garment_base_pose;
        m_garment_base_smpl_vertices = base_vertices;
        m_garment_last_mapping_attempt_mesh_name = m_garment_mesh_name;
        m_garment_last_mapping_attempt_gender = gender;
        m_garment_last_mapping_attempt_base_pose = state.garment_base_pose;
        m_garment_last_mapping_attempt_failed = false;
        m_garment_last_mapping_attempt_error.clear();
        m_garment_version++;
        console_add_line(state,
            std::string("Cloth→SMPL mapping ready for ") + m_garment_mesh_name +
            " (#V=" + std::to_string((int)m_garment_to_smpl.face_index.size()) + ")");

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
        if ((!state.show_garment_mesh && !force) || m_garment_mesh_name.empty())
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
            if (garment.V.rows() == m_garment_rest_vertices.rows() && !garment.V.isApprox(m_garment_rest_vertices, 0.0))
            {
                garment.set_vertices(m_garment_rest_vertices);
                garment.compute_normals();
                m_last_posed_pose_version = m_pose_version;
                m_last_posed_garment_version = m_garment_version;
            }
            return;
        }

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
        if (m_garment_base_smpl_vertices.rows() != m_smplh_model.vertex_count())
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
        const Eigen::MatrixXd& V_rest = m_garment_base_smpl_vertices;
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

            Eigen::RowVector3d T_def, B_def, N_def;
            if (!core::triangle_frame(V_def.row(ia), V_def.row(ib), V_def.row(ic), T_def, B_def, N_def))
            {
                Eigen::RowVector3d T_rest, B_rest, N_rest;
                if (!core::triangle_frame(V_rest.row(ia), V_rest.row(ib), V_rest.row(ic), T_rest, B_rest, N_rest))
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
