#include "render/Renderer.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace poseweaver
{
    namespace
    {
        constexpr int kSketchJointCount = 22;

        static void set_error(std::string* out, const std::string& msg)
        {
            if (out)
                *out = msg;
        }

        static bool parse_json_number_at(const std::string& text, size_t* pos, double* out_value)
        {
            if (!pos || !out_value)
                return false;
            while (*pos < text.size() && std::isspace((unsigned char)text[*pos]))
                ++(*pos);
            const char* begin = text.c_str() + *pos;
            char* end = nullptr;
            const double value = std::strtod(begin, &end);
            if (end == begin)
                return false;
            *out_value = value;
            *pos = (size_t)(end - text.c_str());
            return true;
        }

        static bool parse_json_number_field(const std::string& text, const char* key, double* out_value)
        {
            const std::string needle = std::string("\"") + key + "\":";
            const size_t start = text.find(needle);
            if (start == std::string::npos)
                return false;
            size_t pos = start + needle.size();
            return parse_json_number_at(text, &pos, out_value);
        }

        static bool parse_json_number_array_field(
            const std::string& text,
            const char* key,
            std::vector<double>* out_values,
            std::string* out_error)
        {
            if (!out_values)
                return false;
            out_values->clear();

            const std::string needle = std::string("\"") + key + "\":";
            const size_t start = text.find(needle);
            if (start == std::string::npos)
            {
                set_error(out_error, std::string("Missing numeric array field: ") + key);
                return false;
            }

            size_t pos = text.find('[', start + needle.size());
            if (pos == std::string::npos)
            {
                set_error(out_error, std::string("Malformed numeric array field: ") + key);
                return false;
            }
            ++pos;

            while (pos < text.size())
            {
                while (pos < text.size() && (std::isspace((unsigned char)text[pos]) || text[pos] == ','))
                    ++pos;
                if (pos >= text.size())
                    break;
                if (text[pos] == ']')
                    return true;

                double value = 0.0;
                if (!parse_json_number_at(text, &pos, &value))
                {
                    set_error(out_error, std::string("Failed to parse numeric array field: ") + key);
                    return false;
                }
                out_values->push_back(value);
            }

            set_error(out_error, std::string("Unterminated numeric array field: ") + key);
            return false;
        }

        static bool read_text_file(const std::filesystem::path& path, std::string* out, std::string* out_error)
        {
            std::ifstream in(path);
            if (!in)
            {
                set_error(out_error, "Failed to open " + path.string());
                return false;
            }
            std::ostringstream ss;
            ss << in.rdbuf();
            *out = ss.str();
            return true;
        }

        static bool parse_sketch2rig_joints(
            const std::string& text,
            Eigen::MatrixXd* out_joints,
            std::vector<int>* out_parents,
            int* out_image_size,
            std::string* out_error)
        {
            if (!out_joints || !out_parents)
                return false;

            double image_size_d = 512.0;
            (void)parse_json_number_field(text, "image_size", &image_size_d);
            if (out_image_size)
                *out_image_size = std::clamp((int)std::lround(image_size_d), 2, 8192);

            const size_t joints_key = text.find("\"joints\"");
            if (joints_key == std::string::npos)
            {
                set_error(out_error, "Missing joints array in Sketch2Rig JSON");
                return false;
            }
            size_t pos = text.find('[', joints_key);
            if (pos == std::string::npos)
            {
                set_error(out_error, "Malformed joints array in Sketch2Rig JSON");
                return false;
            }
            ++pos;

            Eigen::MatrixXd joints = Eigen::MatrixXd::Zero(kSketchJointCount, 3);
            std::vector<int> parents((size_t)kSketchJointCount, -1);

            int parsed = 0;
            while (pos < text.size())
            {
                const size_t object_begin = text.find('{', pos);
                const size_t array_end = text.find(']', pos);
                if (array_end != std::string::npos && (object_begin == std::string::npos || array_end < object_begin))
                    break;
                if (object_begin == std::string::npos)
                    break;

                int depth = 0;
                size_t object_end = std::string::npos;
                for (size_t i = object_begin; i < text.size(); ++i)
                {
                    if (text[i] == '{')
                        ++depth;
                    else if (text[i] == '}')
                    {
                        --depth;
                        if (depth == 0)
                        {
                            object_end = i;
                            break;
                        }
                    }
                }
                if (object_end == std::string::npos)
                    break;

                const std::string object = text.substr(object_begin, object_end - object_begin + 1);
                double index_d = -1.0;
                double parent_d = -1.0;
                std::vector<double> values;
                if (parse_json_number_field(object, "index", &index_d)
                    && parse_json_number_field(object, "parent", &parent_d)
                    && parse_json_number_array_field(object, "position", &values, nullptr)
                    && values.size() >= 3)
                {
                    const int index = (int)std::lround(index_d);
                    if (index >= 0 && index < kSketchJointCount)
                    {
                        joints.row(index) = Eigen::RowVector3d(values[0], values[1], values[2]);
                        parents[(size_t)index] = (int)std::lround(parent_d);
                        ++parsed;
                    }
                }

                pos = object_end + 1;
            }

            if (parsed < kSketchJointCount)
            {
                std::ostringstream oss;
                oss << "Expected " << kSketchJointCount << " rig joints, parsed " << parsed;
                set_error(out_error, oss.str());
                return false;
            }

            *out_joints = std::move(joints);
            *out_parents = std::move(parents);
            return true;
        }

        static Eigen::MatrixXd sketch2rig_joints_to_camera_scene_axes(const Eigen::MatrixXd& joints, int image_size)
        {
            const int size = std::max(2, image_size);
            constexpr double kAngleY = 0.7348438501358032;
            constexpr double kAngleX = kAngleY;
            const double cx = 0.5 * (double)(size - 1);
            const double cy = 0.5 * (double)(size - 1);
            const double fx = 0.5 * (double)(size - 1) / std::tan(0.5 * kAngleX);
            const double fy = 0.5 * (double)(size - 1) / std::tan(0.5 * kAngleY);

            Eigen::MatrixXd out(joints.rows(), 3);
            out.setZero();

            for (int i = 0; i < joints.rows(); ++i)
            {
                const double px = joints(i, 0);
                const double py = joints(i, 1);
                const double z = joints(i, 2);
                const double u = (px >= -0.25 && px <= 1.25) ? px * (double)(size - 1) : px;
                const double v = (py >= -0.25 && py <= 1.25) ? py * (double)(size - 1) : py;

                out(i, 0) = (u - cx) / fx * z;
                out(i, 1) = -(v - cy) / fy * z;
                out(i, 2) = -z;
            }

            return out;
        }

        static double rig_height_or_default(const Eigen::MatrixXd& joints)
        {
            if (joints.rows() <= 0)
                return 1.6;
            const double h = joints.col(1).maxCoeff() - joints.col(1).minCoeff();
            return (h > 1e-6 && std::isfinite(h)) ? h : 1.6;
        }

        static bool rig_height_if_valid(const Eigen::MatrixXd& joints, double* out_height)
        {
            if (!out_height || joints.rows() <= 0)
                return false;
            const double h = joints.col(1).maxCoeff() - joints.col(1).minCoeff();
            if (!(h > 1e-6) || !std::isfinite(h))
                return false;
            *out_height = h;
            return true;
        }

        static double smpl_native_height_from_state(const SmplhModel& model, const AppState& state)
        {
            const int V = model.vertex_count();
            const int B = model.beta_count();
            if (V <= 0 || model.J_regressor.cols() != V)
                return 1.6;

            Eigen::VectorXd betas = Eigen::VectorXd::Zero(std::max(0, B));
            const int n = std::min<int>((int)state.smpl_shape.betas.size(), B);
            for (int i = 0; i < n; ++i)
                betas[i] = (double)state.smpl_shape.betas[(size_t)i];

            Eigen::Map<const Eigen::VectorXd> v_template_flat(model.v_template.data(), (size_t)V * 3);
            Eigen::VectorXd v_shaped_flat = v_template_flat;
            if (B > 0)
                v_shaped_flat.noalias() += model.shapedirs * betas;

            Eigen::Map<const MatrixX3dRM> v_shaped(v_shaped_flat.data(), V, 3);
            const Eigen::MatrixXd joints = model.J_regressor * v_shaped;
            if (joints.rows() <= 0 || joints.cols() != 3)
                return 1.6;

            double h = 0.0;
            if (!rig_height_if_valid(joints, &h))
                return 1.6;
            return h;
        }

        static void append_uv_sphere(
            std::vector<Eigen::RowVector3d>& verts,
            std::vector<Eigen::RowVector3i>& faces,
            const Eigen::Vector3d& center,
            double radius)
        {
            constexpr int kRings = 6;
            constexpr int kSlices = 10;

            const int base = (int)verts.size();
            for (int r = 0; r <= kRings; ++r)
            {
                const double v = (double)r / (double)kRings;
                const double theta = v * 3.14159265358979323846;
                const double st = std::sin(theta);
                const double ct = std::cos(theta);
                for (int s = 0; s < kSlices; ++s)
                {
                    const double u = (double)s / (double)kSlices;
                    const double phi = u * 2.0 * 3.14159265358979323846;
                    const Eigen::Vector3d p(
                        center.x() + radius * st * std::cos(phi),
                        center.y() + radius * ct,
                        center.z() + radius * st * std::sin(phi));
                    verts.push_back(p.transpose());
                }
            }

            for (int r = 0; r < kRings; ++r)
            {
                for (int s = 0; s < kSlices; ++s)
                {
                    const int a = base + r * kSlices + s;
                    const int b = base + r * kSlices + ((s + 1) % kSlices);
                    const int c = base + (r + 1) * kSlices + s;
                    const int d = base + (r + 1) * kSlices + ((s + 1) % kSlices);
                    faces.push_back(Eigen::RowVector3i(a, c, b));
                    faces.push_back(Eigen::RowVector3i(b, c, d));
                }
            }
        }

        static void build_rig_geometry(
            const Eigen::MatrixXd& joints,
            const std::vector<int>& parents,
            double rig_height,
            Eigen::MatrixXd& V,
            Eigen::MatrixXi& F,
            Eigen::MatrixXd& P1,
            Eigen::MatrixXd& P2,
            Eigen::MatrixXd& C)
        {
            const double radius = std::clamp(0.010 * rig_height, 0.006, 0.014);
            std::vector<Eigen::RowVector3d> verts;
            std::vector<Eigen::RowVector3i> faces;
            verts.reserve((size_t)joints.rows() * 70u);
            faces.reserve((size_t)joints.rows() * 120u);

            for (int i = 0; i < joints.rows(); ++i)
                append_uv_sphere(verts, faces, joints.row(i).transpose(), radius);

            V.resize((int)verts.size(), 3);
            F.resize((int)faces.size(), 3);
            for (int i = 0; i < (int)verts.size(); ++i)
                V.row(i) = verts[(size_t)i];
            for (int i = 0; i < (int)faces.size(); ++i)
                F.row(i) = faces[(size_t)i];

            std::vector<Eigen::RowVector3d> p1;
            std::vector<Eigen::RowVector3d> p2;
            std::vector<Eigen::RowVector3d> col;
            p1.reserve(parents.size());
            p2.reserve(parents.size());
            col.reserve(parents.size());

            for (int child = 0; child < (int)parents.size() && child < joints.rows(); ++child)
            {
                const int parent = parents[(size_t)child];
                if (parent < 0 || parent >= joints.rows())
                    continue;
                p1.push_back(joints.row(parent));
                p2.push_back(joints.row(child));
                col.push_back(Eigen::RowVector3d(0.08, 0.48, 1.00));
            }

            P1.resize((int)p1.size(), 3);
            P2.resize((int)p2.size(), 3);
            C.resize((int)col.size(), 3);
            for (int i = 0; i < (int)p1.size(); ++i)
            {
                P1.row(i) = p1[(size_t)i];
                P2.row(i) = p2[(size_t)i];
                C.row(i) = col[(size_t)i];
            }
        }
    }

    void Renderer::ensure_sketch2rig_rig_loaded(AppState& state)
    {
        auto& data = m_viewer.data(m_sketch2rig_rig_mesh_index);
        const auto clear_smpl_fit = [&]()
        {
            state.scene_view_smpl_fit_valid = false;
            state.scene_view_smpl_fit_scale = { 1.0f, 1.0f, 1.0f };
            state.scene_view_smpl_fit_translation = { 0.0f, 0.0f, 0.0f };
            state.scene_view_smpl_fit_rms = 0.0f;
        };

        if (state.scene_view_sequence_rig_overlay_active)
        {
            const std::string desired_overlay = state.scene_view_sequence_rig_overlay_path;
            if (!state.show_sketch2rig_rig
                || !state.scene_view_sequence_rig_overlay_visible
                || desired_overlay.empty())
            {
                data.set_visible(false);
                return;
            }

            std::string text;
            std::string err;
            Eigen::MatrixXd source_joints;
            std::vector<int> parents;
            int image_size = 512;
            if (!read_text_file(desired_overlay, &text, &err)
                || !parse_sketch2rig_joints(text, &source_joints, &parents, &image_size, &err))
            {
                data.set_visible(false);
                return;
            }

            const Eigen::MatrixXd camera_joints = sketch2rig_joints_to_camera_scene_axes(
                source_joints,
                image_size);
            Eigen::MatrixXd centered_camera_joints = camera_joints;
            if (camera_joints.rows() > 0)
            {
                const Eigen::Vector3d root_offset = camera_joints.row(0).transpose();
                centered_camera_joints.rowwise() -= root_offset.transpose();

                double predicted_height = 0.0;
                const double target_height = ensure_smplh_model_loaded(state, state.selection.gender)
                    ? smpl_native_height_from_state(m_smplh_model, state)
                    : 1.6;
                if (rig_height_if_valid(centered_camera_joints, &predicted_height)
                    && predicted_height > 1e-6
                    && std::isfinite(predicted_height)
                    && target_height > 1e-6
                    && std::isfinite(target_height))
                {
                    const double scene_scale = std::clamp(target_height / predicted_height, 0.05, 20.0);
                    centered_camera_joints *= scene_scale;
                }
            }

            Eigen::MatrixXd V, P1, P2, C;
            Eigen::MatrixXi F;
            build_rig_geometry(
                centered_camera_joints,
                parents,
                rig_height_or_default(centered_camera_joints),
                V,
                F,
                P1,
                P2,
                C);

            data.clear();
            data.set_mesh(V, F);
            data.show_faces = 1;
            data.show_lines = 0;
            data.show_overlay = 1;
            data.set_colors(Eigen::RowVector3d(0.1, 0.4, 1.0));
            data.add_edges(P1, P2, C);
            data.set_visible(true);
            return;
        }

        const std::string desired = state.sketch2rig.last_output_path;
        if (desired.empty())
        {
            data.set_visible(false);
            if (state.scene_view_smpl_fit_valid)
                m_smplh_deform_dirty = true;
            m_sketch2rig_rig_scene_joints.resize(0, 0);
            state.scene_view_target_valid = false;
            state.scene_view_raw_rig_root_valid = false;
            state.scene_view_raw_rig_height_valid = false;
            clear_smpl_fit();
            return;
        }

        if (desired != m_sketch2rig_rig_source_path)
        {
            m_sketch2rig_rig_source_path = desired;
            m_sketch2rig_rig_source_valid = false;
            m_sketch2rig_rig_last_error.clear();
            m_sketch2rig_rig_source_joints.resize(0, 0);
            m_sketch2rig_rig_scene_joints.resize(0, 0);
            m_sketch2rig_rig_parents.clear();
            m_smplh_deform_dirty = true;
            clear_smpl_fit();

            std::string text;
            std::string err;
            if (read_text_file(desired, &text, &err)
                && parse_sketch2rig_joints(
                    text,
                    &m_sketch2rig_rig_source_joints,
                    &m_sketch2rig_rig_parents,
                    &m_sketch2rig_rig_image_size,
                    &err))
            {
                m_sketch2rig_rig_source_valid = true;
                console_add_line(state, "Loaded Sketch2Rig rig overlay: " + desired);
            }
            else
            {
                m_sketch2rig_rig_last_error = err.empty() ? "Failed to load Sketch2Rig rig overlay" : err;
                console_add_line(state, "ERROR: " + m_sketch2rig_rig_last_error);
            }
        }

        if (!m_sketch2rig_rig_source_valid || m_sketch2rig_rig_source_joints.rows() <= 0)
        {
            data.set_visible(false);
            if (state.scene_view_smpl_fit_valid)
                m_smplh_deform_dirty = true;
            m_sketch2rig_rig_scene_joints.resize(0, 0);
            state.scene_view_target_valid = false;
            state.scene_view_raw_rig_root_valid = false;
            state.scene_view_raw_rig_height_valid = false;
            clear_smpl_fit();
            return;
        }

        const Eigen::MatrixXd camera_joints = sketch2rig_joints_to_camera_scene_axes(
            m_sketch2rig_rig_source_joints,
            m_sketch2rig_rig_image_size);
        Eigen::MatrixXd centered_camera_joints = camera_joints;
        Eigen::Vector3d root_offset = Eigen::Vector3d::Zero();
        double scene_scale = 1.0;
        if (camera_joints.rows() > 0)
        {
            root_offset = camera_joints.row(0).transpose();
            centered_camera_joints.rowwise() -= root_offset.transpose();

            double predicted_height = 0.0;
            const double target_height = ensure_smplh_model_loaded(state, state.selection.gender)
                ? smpl_native_height_from_state(m_smplh_model, state)
                : 1.6;
            if (rig_height_if_valid(centered_camera_joints, &predicted_height)
                && predicted_height > 1e-6
                && std::isfinite(predicted_height)
                && target_height > 1e-6
                && std::isfinite(target_height))
            {
                scene_scale = std::clamp(target_height / predicted_height, 0.05, 20.0);
                centered_camera_joints *= scene_scale;
            }

            state.scene_view_raw_rig_root = {
                0.0f,
                0.0f,
                0.0f
            };
            state.scene_view_raw_rig_root_valid = true;

            // The predicted rig is scaled to native SMPL scale here.  Leave this
            // invalid so RendererSmpl only translates to the extracted root,
            // keeping simulation/body scale constant instead of following the
            // camera-depth scale predicted by Sketch2Rig.
            state.scene_view_raw_rig_height = (float)target_height;
            state.scene_view_raw_rig_height_valid = false;
        }
        else
        {
            state.scene_view_raw_rig_root_valid = false;
            state.scene_view_raw_rig_height_valid = false;
        }

        if (m_sketch2rig_rig_scene_joints.rows() != centered_camera_joints.rows()
            || m_sketch2rig_rig_scene_joints.cols() != centered_camera_joints.cols()
            || !m_sketch2rig_rig_scene_joints.isApprox(centered_camera_joints, 1e-10))
        {
            m_sketch2rig_rig_scene_joints = centered_camera_joints;
            m_smplh_deform_dirty = true;
            clear_smpl_fit();
        }

        state.scene_view_target_eye = {
            (float)(-scene_scale * root_offset.x()),
            (float)(-scene_scale * root_offset.y()),
            (float)(-scene_scale * root_offset.z())
        };
        state.scene_view_target_direction = { 0.0f, 0.0f, -1.0f };
        state.scene_view_target_up = { 0.0f, 1.0f, 0.0f };
        state.scene_view_target_valid = true;

        if (!state.show_sketch2rig_rig)
        {
            data.set_visible(false);
            return;
        }

        Eigen::MatrixXd V, P1, P2, C;
        Eigen::MatrixXi F;
        build_rig_geometry(
            centered_camera_joints,
            m_sketch2rig_rig_parents,
            rig_height_or_default(centered_camera_joints),
            V,
            F,
            P1,
            P2,
            C);

        data.clear();
        data.set_mesh(V, F);
        data.show_faces = 1;
        data.show_lines = 0;
        data.show_overlay = 1;
        data.line_width = 48.0f;
        data.uniform_colors(
            Eigen::Vector4d(1.0, 0.34, 0.14, 1.0),
            Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
            Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
        data.add_edges(P1, P2, C);
        data.set_visible(true);
    }
}
