#include "render/Renderer.h"
#include "core/Paths.h"

#include <Eigen/Core>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cnpy.h>
#include <glad/glad.h>
#include <igl/frustum.h>
#include <igl/look_at.h>
#include <stb_image_write.h>

namespace poseweaver
{
    namespace
    {
        constexpr float kSketch2RigVerticalFovRad = 0.7348438501358032f;
        constexpr float kSketch2RigVerticalFovDeg =
            kSketch2RigVerticalFovRad * 180.0f / 3.14159265358979323846f;

        struct UvFace
        {
            int face = -1;
            double u0 = 0.0;
            double v0 = 0.0;
            double u1 = 0.0;
            double v1 = 0.0;
            double u2 = 0.0;
            double v2 = 0.0;
        };

        static std::string read_text_file_displacement(const std::filesystem::path& path)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f)
                return {};
            std::ostringstream ss;
            ss << f.rdbuf();
            return ss.str();
        }

        static std::string parse_json_string_field_displacement(const std::string& text, const char* key)
        {
            const std::string needle = std::string("\"") + key + "\"";
            const size_t key_pos = text.find(needle);
            if (key_pos == std::string::npos)
                return {};

            size_t start = text.find(':', key_pos + needle.size());
            if (start == std::string::npos)
                return {};
            ++start;
            while (start < text.size() && std::isspace((unsigned char)text[start]))
                ++start;
            if (start >= text.size() || text[start] != '"')
                return {};
            ++start;

            std::string out;
            bool escaped = false;
            for (size_t i = start; i < text.size(); ++i)
            {
                const char c = text[i];
                if (escaped)
                {
                    switch (c)
                    {
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    default: out += c; break;
                    }
                    escaped = false;
                    continue;
                }
                if (c == '\\')
                {
                    escaped = true;
                    continue;
                }
                if (c == '"')
                    break;
                out += c;
            }
            return out;
        }

        static bool barycentric_uv(
            double u,
            double v,
            const UvFace& tri,
            Eigen::Vector3d& bc)
        {
            const double v0x = tri.u1 - tri.u0;
            const double v0y = tri.v1 - tri.v0;
            const double v1x = tri.u2 - tri.u0;
            const double v1y = tri.v2 - tri.v0;
            const double v2x = u - tri.u0;
            const double v2y = v - tri.v0;

            const double d00 = v0x * v0x + v0y * v0y;
            const double d01 = v0x * v1x + v0y * v1y;
            const double d11 = v1x * v1x + v1y * v1y;
            const double d20 = v2x * v0x + v2y * v0y;
            const double d21 = v2x * v1x + v2y * v1y;
            const double denom = d00 * d11 - d01 * d01;
            if (!(std::abs(denom) > 1e-20) || !std::isfinite(denom))
                return false;

            const double b = (d11 * d20 - d01 * d21) / denom;
            const double c = (d00 * d21 - d01 * d20) / denom;
            const double a = 1.0 - b - c;
            constexpr double eps = -1e-5;
            if (a < eps || b < eps || c < eps)
                return false;

            bc = Eigen::Vector3d(a, b, c);
            return bc.allFinite();
        }

        template <typename T>
        static float npy_value_as_float(const cnpy::NpyArray& a, size_t idx)
        {
            return (float)a.data<T>()[idx];
        }

        static bool npy_float_at(const cnpy::NpyArray& a, size_t idx, float* out)
        {
            if (!out)
                return false;
            if (a.word_size == sizeof(float))
            {
                *out = npy_value_as_float<float>(a, idx);
                return true;
            }
            if (a.word_size == sizeof(double))
            {
                *out = npy_value_as_float<double>(a, idx);
                return true;
            }
            return false;
        }

        static bool make_target_camera_matrices(
            const AppState& state,
            float dnear,
            float dfar,
            Eigen::Matrix4f& view,
            Eigen::Matrix4f& proj)
        {
            if (!state.scene_view_target_valid)
                return false;

            Eigen::Vector3f eye(
                state.scene_view_target_eye[0],
                state.scene_view_target_eye[1],
                state.scene_view_target_eye[2]);
            Eigen::Vector3f dir(
                state.scene_view_target_direction[0],
                state.scene_view_target_direction[1],
                state.scene_view_target_direction[2]);
            Eigen::Vector3f up(
                state.scene_view_target_up[0],
                state.scene_view_target_up[1],
                state.scene_view_target_up[2]);

            if (!eye.allFinite() || !dir.allFinite() || !up.allFinite())
                return false;
            if (!(dir.squaredNorm() > 1e-12f))
                return false;
            dir.normalize();
            up -= dir * up.dot(dir);
            if (!(up.squaredNorm() > 1e-12f))
                up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            up.normalize();

            igl::look_at(eye, eye + dir, up, view);

            constexpr float kPi = 3.14159265358979323846f;
            const float f_h = std::tan(kSketch2RigVerticalFovDeg / 360.0f * kPi) * dnear;
            igl::frustum(-f_h, f_h, -f_h, f_h, dnear, dfar, proj);
            return true;
        }

        static bool project_to_image(
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            const Eigen::Vector3d& p,
            int w,
            int h,
            float& x,
            float& y)
        {
            const Eigen::Vector4f hp((float)p.x(), (float)p.y(), (float)p.z(), 1.0f);
            const Eigen::Vector4f clip = (proj * view) * hp;
            if (!(clip.w() > 1e-8f) || !std::isfinite(clip.w()))
                return false;

            const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
            if (!ndc.allFinite())
                return false;
            x = (ndc.x() * 0.5f + 0.5f) * (float)w;
            y = (-ndc.y() * 0.5f + 0.5f) * (float)h;
            return true;
        }

        static bool project_to_image_depth(
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            const Eigen::Vector3d& p,
            int w,
            int h,
            float& x,
            float& y,
            float& depth01)
        {
            const Eigen::Vector4f hp((float)p.x(), (float)p.y(), (float)p.z(), 1.0f);
            const Eigen::Vector4f clip = (proj * view) * hp;
            if (!(std::abs(clip.w()) > 1e-8f) || !std::isfinite(clip.w()))
                return false;

            const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
            if (!ndc.allFinite() || ndc.z() < -1.0f || ndc.z() > 1.0f)
                return false;

            x = (ndc.x() * 0.5f + 0.5f) * (float)w;
            y = (-ndc.y() * 0.5f + 0.5f) * (float)h;
            depth01 = ndc.z() * 0.5f + 0.5f;
            return std::isfinite(x) && std::isfinite(y) && std::isfinite(depth01);
        }

        static void set_uniform_matrix4_if_present(GLuint program, const char* name, const Eigen::Matrix4f& value)
        {
            const GLint loc = glGetUniformLocation(program, name);
            if (loc >= 0)
                glUniformMatrix4fv(loc, 1, GL_FALSE, value.data());
        }

        static void set_uniform_1f_if_present(GLuint program, const char* name, float value)
        {
            const GLint loc = glGetUniformLocation(program, name);
            if (loc >= 0)
                glUniform1f(loc, value);
        }

        static void set_uniform_1i_if_present(GLuint program, const char* name, int value)
        {
            const GLint loc = glGetUniformLocation(program, name);
            if (loc >= 0)
                glUniform1i(loc, value);
        }
    }

    void Renderer::process_canva_displacement_map(AppState& state)
    {
        const std::uint64_t request_id = state.canva_displacement_request_id;
        if (request_id == 0 || request_id == m_last_canva_displacement_request_id)
            return;

        m_last_canva_displacement_request_id = request_id;
        state.canva_displacement_completed_id = request_id;
        state.canva_displacement_path.clear();

        auto fail = [&](const std::string& msg)
        {
            console_add_line(state, "Canva displacement failed: " + msg);
        };

        if (!state.scene_view_target_valid)
        {
            fail("predicted camera view is not available");
            return;
        }
        if (!m_smplh_loaded || m_smplh_vertices.rows() <= 0 || m_smplh_model.F.rows() <= 0)
        {
            fail("SMPL mesh is not ready");
            return;
        }
        if (!m_smpl_uv_layout_loaded && !try_load_smpl_uv_layout_once())
        {
            fail("SMPL UV layout is not available");
            return;
        }
        if (m_smpl_F_uv.rows() != m_smplh_model.F.rows() || m_smpl_uv_coords.rows() <= 0)
        {
            fail("SMPL UV layout does not match the loaded SMPL mesh");
            return;
        }

        std::filesystem::path uv_json_path = state.canva_displacement_uv_json_path;
        if (uv_json_path.empty())
        {
            fail("missing Sketch2UV result path");
            return;
        }

        const std::string uv_json = read_text_file_displacement(uv_json_path);
        if (uv_json.empty())
        {
            fail("could not read " + uv_json_path.string());
            return;
        }

        std::filesystem::path uv_npz_path = parse_json_string_field_displacement(uv_json, "uv_npz_path");
        if (uv_npz_path.empty())
            uv_npz_path = uv_json_path.parent_path() / (uv_json_path.stem().string() + "_uv.npz");
        if (uv_npz_path.is_relative())
            uv_npz_path = uv_json_path.parent_path() / uv_npz_path;

        cnpy::NpyArray uv_arr;
        cnpy::NpyArray mask_arr;
        try
        {
            uv_arr = cnpy::npz_load(uv_npz_path.string(), "uv");
            mask_arr = cnpy::npz_load(uv_npz_path.string(), "mask");
        }
        catch (const std::exception& e)
        {
            fail(std::string("could not load UV NPZ: ") + e.what());
            return;
        }

        if (uv_arr.shape.size() != 3 || uv_arr.shape[0] < 2 || mask_arr.shape.size() != 2)
        {
            fail("unexpected Sketch2UV NPZ shape");
            return;
        }

        const int uv_h = (int)uv_arr.shape[1];
        const int uv_w = (int)uv_arr.shape[2];
        if (uv_w <= 0 || uv_h <= 0 || (int)mask_arr.shape[0] != uv_h || (int)mask_arr.shape[1] != uv_w)
        {
            fail("UV and mask dimensions do not match");
            return;
        }
        if (!((uv_arr.word_size == sizeof(float) || uv_arr.word_size == sizeof(double))
                && (mask_arr.word_size == sizeof(float) || mask_arr.word_size == sizeof(double))))
        {
            fail("UV NPZ arrays are not float arrays");
            return;
        }

        Eigen::Matrix4f view;
        Eigen::Matrix4f proj;
        const auto& core = m_viewer.core();
        if (!make_target_camera_matrices(state, core.camera_dnear, core.camera_dfar, view, proj))
        {
            fail("could not build predicted camera matrices");
            return;
        }

        std::vector<UvFace> faces;
        faces.reserve((size_t)m_smplh_model.F.rows());
        constexpr int kGrid = 64;
        std::vector<std::vector<int>> grid((size_t)kGrid * (size_t)kGrid);

        for (int fi = 0; fi < m_smpl_F_uv.rows(); ++fi)
        {
            const int t0 = m_smpl_F_uv(fi, 0);
            const int t1 = m_smpl_F_uv(fi, 1);
            const int t2 = m_smpl_F_uv(fi, 2);
            if (t0 < 0 || t1 < 0 || t2 < 0)
                continue;
            if (t0 >= m_smpl_uv_coords.rows() || t1 >= m_smpl_uv_coords.rows() || t2 >= m_smpl_uv_coords.rows())
                continue;

            UvFace f;
            f.face = fi;
            f.u0 = m_smpl_uv_coords(t0, 0);
            f.v0 = m_smpl_uv_coords(t0, 1);
            f.u1 = m_smpl_uv_coords(t1, 0);
            f.v1 = m_smpl_uv_coords(t1, 1);
            f.u2 = m_smpl_uv_coords(t2, 0);
            f.v2 = m_smpl_uv_coords(t2, 1);
            const int idx = (int)faces.size();
            faces.push_back(f);

            const double min_u = std::min({ f.u0, f.u1, f.u2 });
            const double max_u = std::max({ f.u0, f.u1, f.u2 });
            const double min_v = std::min({ f.v0, f.v1, f.v2 });
            const double max_v = std::max({ f.v0, f.v1, f.v2 });
            const int gx0 = std::clamp((int)std::floor(min_u * kGrid), 0, kGrid - 1);
            const int gx1 = std::clamp((int)std::floor(max_u * kGrid), 0, kGrid - 1);
            const int gy0 = std::clamp((int)std::floor(min_v * kGrid), 0, kGrid - 1);
            const int gy1 = std::clamp((int)std::floor(max_v * kGrid), 0, kGrid - 1);
            for (int gy = gy0; gy <= gy1; ++gy)
            {
                for (int gx = gx0; gx <= gx1; ++gx)
                    grid[(size_t)gy * (size_t)kGrid + (size_t)gx].push_back(idx);
            }
        }

        if (faces.empty())
        {
            fail("SMPL UV layout has no usable triangles");
            return;
        }

        std::vector<unsigned char> out((size_t)uv_w * (size_t)uv_h * 4u, 0u);
        int projected_count = 0;
        int foreground_count = 0;
        int outside_body_uv_count = 0;

        for (int y = 0; y < uv_h; ++y)
        {
            for (int x = 0; x < uv_w; ++x)
            {
                const size_t pix = (size_t)y * (size_t)uv_w + (size_t)x;
                float mask = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (!npy_float_at(mask_arr, pix, &mask)
                    || !npy_float_at(uv_arr, pix, &u)
                    || !npy_float_at(uv_arr, (size_t)uv_h * (size_t)uv_w + pix, &v))
                    continue;
                if (!(mask > 0.5f) || !std::isfinite(u) || !std::isfinite(v))
                    continue;

                ++foreground_count;
                u = std::clamp(u, 0.0f, 1.0f);
                v = std::clamp(v, 0.0f, 1.0f);

                const int gx = std::clamp((int)std::floor((double)u * kGrid), 0, kGrid - 1);
                const int gy = std::clamp((int)std::floor((double)v * kGrid), 0, kGrid - 1);
                const std::vector<int>& candidates = grid[(size_t)gy * (size_t)kGrid + (size_t)gx];

                Eigen::Vector3d bc;
                int face_idx = -1;
                for (const int ci : candidates)
                {
                    if (ci < 0 || ci >= (int)faces.size())
                        continue;
                    if (barycentric_uv((double)u, (double)v, faces[(size_t)ci], bc))
                    {
                        face_idx = faces[(size_t)ci].face;
                        break;
                    }
                }
                if (face_idx < 0)
                {
                    unsigned char* dst = out.data() + (pix * 4u);
                    dst[0] = 0u;
                    dst[1] = 0u;
                    dst[2] = 255u;
                    dst[3] = 255u;
                    ++outside_body_uv_count;
                    continue;
                }

                const int ia = m_smplh_model.F(face_idx, 0);
                const int ib = m_smplh_model.F(face_idx, 1);
                const int ic = m_smplh_model.F(face_idx, 2);
                if (ia < 0 || ib < 0 || ic < 0
                    || ia >= m_smplh_vertices.rows()
                    || ib >= m_smplh_vertices.rows()
                    || ic >= m_smplh_vertices.rows())
                    continue;

                const Eigen::Vector3d p =
                    bc.x() * m_smplh_vertices.row(ia).transpose()
                    + bc.y() * m_smplh_vertices.row(ib).transpose()
                    + bc.z() * m_smplh_vertices.row(ic).transpose();

                float px = 0.0f;
                float py = 0.0f;
                if (!project_to_image(view, proj, p, uv_w, uv_h, px, py))
                    continue;

                const float dx = px - ((float)x + 0.5f);
                const float dy = py - ((float)y + 0.5f);
                unsigned char* dst = out.data() + (pix * 4u);
                dst[0] = (unsigned char)std::clamp<int>((int)std::lround(std::abs(dx)), 0, 255);
                dst[1] = (unsigned char)std::clamp<int>((int)std::lround(std::abs(dy)), 0, 255);
                dst[2] = 0u;
                dst[3] = 255u;
                ++projected_count;
            }
        }

        std::error_code ec;
        const std::filesystem::path renders_dir = std::filesystem::path("renders");
        std::filesystem::create_directories(renders_dir, ec);
        if (ec)
        {
            fail("could not create renders directory: " + ec.message());
            return;
        }

        const std::filesystem::path out_path = renders_dir / "canva_displacement.png";
        const int ok = stbi_write_png(out_path.string().c_str(), uv_w, uv_h, 4, out.data(), uv_w * 4);
        if (ok == 0)
        {
            fail("could not write " + out_path.string());
            return;
        }

        state.canva_displacement_path = out_path.string();
        console_add_line(
            state,
            "Canva displacement saved: " + out_path.string()
                + " (projected=" + std::to_string(projected_count)
                + ", outside_body_uv=" + std::to_string(outside_body_uv_count)
                + "/" + std::to_string(foreground_count) + ")");
    }

    bool Renderer::fit_cloth_render_to_smpl_visibility(
        AppState& state,
        const std::vector<unsigned char>& cloth_rgba,
        int width,
        int height,
        const std::string& uv_json_path_string,
        std::vector<unsigned char>& out_rgba,
        int& out_tested_count,
        int& out_removed_count,
        std::string* out_error)
    {
        out_rgba.clear();
        out_tested_count = 0;
        out_removed_count = 0;

        auto fail = [&](const std::string& msg)
        {
            if (out_error)
                *out_error = msg;
            return false;
        };

        if (width <= 0 || height <= 0)
            return fail("invalid cloth layer size");
        if (cloth_rgba.size() < (size_t)width * (size_t)height * 4u)
            return fail("cloth layer pixel buffer is incomplete");
        if (!state.scene_view_target_valid)
            return fail("predicted camera view is not available");

        ensure_smplh_mesh_loaded(state);
        ensure_garment_mesh_loaded(state);
        handle_garment_posing(state, false);

        if (!m_smplh_loaded || m_smplh_vertices.rows() <= 0 || m_smplh_model.F.rows() <= 0)
            return fail("SMPL mesh is not ready");
        if (!m_smpl_uv_layout_loaded && !try_load_smpl_uv_layout_once())
            return fail("SMPL UV layout is not available");
        if (m_smpl_F_uv.rows() != m_smplh_model.F.rows() || m_smpl_uv_coords.rows() <= 0)
            return fail("SMPL UV layout does not match the loaded SMPL mesh");

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() <= 0 || garment.F.rows() <= 0)
            return fail("cloth mesh is not ready");

        std::filesystem::path uv_json_path = uv_json_path_string;
        if (uv_json_path.empty())
            return fail("missing displacement UV prediction path");

        const std::string uv_json = read_text_file_displacement(uv_json_path);
        if (uv_json.empty())
            return fail("could not read " + uv_json_path.string());

        std::filesystem::path uv_npz_path = parse_json_string_field_displacement(uv_json, "uv_npz_path");
        if (uv_npz_path.empty())
            uv_npz_path = uv_json_path.parent_path() / (uv_json_path.stem().string() + "_uv.npz");
        if (uv_npz_path.is_relative())
            uv_npz_path = uv_json_path.parent_path() / uv_npz_path;

        cnpy::NpyArray uv_arr;
        cnpy::NpyArray mask_arr;
        try
        {
            uv_arr = cnpy::npz_load(uv_npz_path.string(), "uv");
            mask_arr = cnpy::npz_load(uv_npz_path.string(), "mask");
        }
        catch (const std::exception& e)
        {
            return fail(std::string("could not load displacement UV NPZ: ") + e.what());
        }

        if (uv_arr.shape.size() != 3 || uv_arr.shape[0] < 2 || mask_arr.shape.size() != 2)
            return fail("unexpected displacement UV NPZ shape");

        const int uv_h = (int)uv_arr.shape[1];
        const int uv_w = (int)uv_arr.shape[2];
        if (uv_w <= 0 || uv_h <= 0 || (int)mask_arr.shape[0] != uv_h || (int)mask_arr.shape[1] != uv_w)
            return fail("UV and mask dimensions do not match");
        if (!((uv_arr.word_size == sizeof(float) || uv_arr.word_size == sizeof(double))
                && (mask_arr.word_size == sizeof(float) || mask_arr.word_size == sizeof(double))))
            return fail("UV NPZ arrays are not float arrays");

        Eigen::Matrix4f view;
        Eigen::Matrix4f proj;
        const auto& core_ref = m_viewer.core();
        if (!make_target_camera_matrices(state, core_ref.camera_dnear, core_ref.camera_dfar, view, proj))
            return fail("could not build predicted camera matrices");

        std::vector<UvFace> faces;
        faces.reserve((size_t)m_smplh_model.F.rows());
        constexpr int kGrid = 64;
        std::vector<std::vector<int>> grid((size_t)kGrid * (size_t)kGrid);

        for (int fi = 0; fi < m_smpl_F_uv.rows(); ++fi)
        {
            const int t0 = m_smpl_F_uv(fi, 0);
            const int t1 = m_smpl_F_uv(fi, 1);
            const int t2 = m_smpl_F_uv(fi, 2);
            if (t0 < 0 || t1 < 0 || t2 < 0)
                continue;
            if (t0 >= m_smpl_uv_coords.rows() || t1 >= m_smpl_uv_coords.rows() || t2 >= m_smpl_uv_coords.rows())
                continue;

            UvFace f;
            f.face = fi;
            f.u0 = m_smpl_uv_coords(t0, 0);
            f.v0 = m_smpl_uv_coords(t0, 1);
            f.u1 = m_smpl_uv_coords(t1, 0);
            f.v1 = m_smpl_uv_coords(t1, 1);
            f.u2 = m_smpl_uv_coords(t2, 0);
            f.v2 = m_smpl_uv_coords(t2, 1);
            const int idx = (int)faces.size();
            faces.push_back(f);

            const double min_u = std::min({ f.u0, f.u1, f.u2 });
            const double max_u = std::max({ f.u0, f.u1, f.u2 });
            const double min_v = std::min({ f.v0, f.v1, f.v2 });
            const double max_v = std::max({ f.v0, f.v1, f.v2 });
            const int gx0 = std::clamp((int)std::floor(min_u * kGrid), 0, kGrid - 1);
            const int gx1 = std::clamp((int)std::floor(max_u * kGrid), 0, kGrid - 1);
            const int gy0 = std::clamp((int)std::floor(min_v * kGrid), 0, kGrid - 1);
            const int gy1 = std::clamp((int)std::floor(max_v * kGrid), 0, kGrid - 1);
            for (int gy = gy0; gy <= gy1; ++gy)
            {
                for (int gx = gx0; gx <= gx1; ++gx)
                    grid[(size_t)gy * (size_t)kGrid + (size_t)gx].push_back(idx);
            }
        }

        if (faces.empty())
            return fail("SMPL UV layout has no usable triangles");

        GLint old_framebuffer = 0;
        GLint old_viewport[4] = { 0, 0, 0, 0 };
        GLint old_read_buffer = GL_BACK;
        GLboolean old_scissor = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean old_blend = glIsEnabled(GL_BLEND);
        GLboolean old_depth = glIsEnabled(GL_DEPTH_TEST);
        GLboolean old_cull = glIsEnabled(GL_CULL_FACE);
        GLboolean old_depth_mask = GL_TRUE;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_framebuffer);
        glGetIntegerv(GL_VIEWPORT, old_viewport);
        glGetIntegerv(GL_READ_BUFFER, &old_read_buffer);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &old_depth_mask);

        GLuint fbo = 0;
        GLuint color = 0;
        GLuint depth = 0;
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &color);
        glGenRenderbuffers(1, &depth);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
            glReadBuffer((GLenum)old_read_buffer);
            glDeleteRenderbuffers(1, &depth);
            glDeleteTextures(1, &color);
            glDeleteFramebuffers(1, &fbo);
            return fail("cloth fitting framebuffer is incomplete");
        }

        auto& core = m_viewer.core();
        const Eigen::Vector4f old_core_viewport = core.viewport;
        const Eigen::Matrix4f old_view = core.view;
        const Eigen::Matrix4f old_proj = core.proj;
        const Eigen::Matrix4f old_norm = core.norm;
        const Eigen::Vector4f old_background = core.background_color;
        const bool old_garment_visible = garment.is_visible;

        core.viewport = Eigen::Vector4f(0.0f, 0.0f, (float)width, (float)height);
        core.view = view;
        core.proj = proj;
        core.norm = core.view.inverse().transpose();
        core.background_color = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);

        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (garment.dirty)
        {
            garment.updateGL(garment, garment.invert_normals, garment.meshgl);
            garment.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }

        garment.set_visible(true);
        if (garment.meshgl.is_initialized)
        {
            garment.meshgl.bind_mesh();
            const GLuint prog = garment.meshgl.shader_mesh;
            set_uniform_matrix4_if_present(prog, "view", core.view);
            set_uniform_matrix4_if_present(prog, "proj", core.proj);
            set_uniform_matrix4_if_present(prog, "normal_matrix", core.norm);
            set_uniform_1f_if_present(prog, "texture_factor", 0.0f);
            set_uniform_1f_if_present(prog, "matcap_factor", 0.0f);
            set_uniform_1f_if_present(prog, "double_sided", 1.0f);
            set_uniform_1i_if_present(prog, "toon_outline_pass", 0);
            set_uniform_1f_if_present(prog, "toon_outline_width", 0.0f);
            garment.meshgl.draw_mesh(true);
        }

        std::vector<float> cloth_depth_gl((size_t)width * (size_t)height, 1.0f);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, cloth_depth_gl.data());

        garment.is_visible = old_garment_visible;
        core.viewport = old_core_viewport;
        core.view = old_view;
        core.proj = old_proj;
        core.norm = old_norm;
        core.background_color = old_background;

        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
        glReadBuffer((GLenum)old_read_buffer);
        glViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
        if (old_scissor)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);
        if (old_blend)
            glEnable(GL_BLEND);
        else
            glDisable(GL_BLEND);
        if (old_depth)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        if (old_cull)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);
        glDepthMask(old_depth_mask);
        glDeleteRenderbuffers(1, &depth);
        glDeleteTextures(1, &color);
        glDeleteFramebuffers(1, &fbo);

        out_rgba = cloth_rgba;
        constexpr float kDepthEps = 0.0015f;
        const size_t uv_count = (size_t)uv_w * (size_t)uv_h;

        for (int y = 0; y < height; ++y)
        {
            const int uv_y = std::clamp(
                (int)std::floor((((float)y + 0.5f) / (float)height) * (float)uv_h),
                0,
                uv_h - 1);
            for (int x = 0; x < width; ++x)
            {
                const size_t out_pix = (size_t)y * (size_t)width + (size_t)x;
                if (out_rgba[out_pix * 4u + 3u] == 0u)
                    continue;

                const int uv_x = std::clamp(
                    (int)std::floor((((float)x + 0.5f) / (float)width) * (float)uv_w),
                    0,
                    uv_w - 1);
                const size_t uv_pix = (size_t)uv_y * (size_t)uv_w + (size_t)uv_x;

                float mask = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (!npy_float_at(mask_arr, uv_pix, &mask)
                    || !npy_float_at(uv_arr, uv_pix, &u)
                    || !npy_float_at(uv_arr, uv_count + uv_pix, &v))
                    continue;
                if (!(mask > 0.5f) || !std::isfinite(u) || !std::isfinite(v))
                    continue;

                u = std::clamp(u, 0.0f, 1.0f);
                v = std::clamp(v, 0.0f, 1.0f);

                const int gx = std::clamp((int)std::floor((double)u * kGrid), 0, kGrid - 1);
                const int gy = std::clamp((int)std::floor((double)v * kGrid), 0, kGrid - 1);
                const std::vector<int>& candidates = grid[(size_t)gy * (size_t)kGrid + (size_t)gx];

                Eigen::Vector3d bc;
                int face_idx = -1;
                for (const int ci : candidates)
                {
                    if (ci < 0 || ci >= (int)faces.size())
                        continue;
                    if (barycentric_uv((double)u, (double)v, faces[(size_t)ci], bc))
                    {
                        face_idx = faces[(size_t)ci].face;
                        break;
                    }
                }
                if (face_idx < 0)
                    continue;

                const int ia = m_smplh_model.F(face_idx, 0);
                const int ib = m_smplh_model.F(face_idx, 1);
                const int ic = m_smplh_model.F(face_idx, 2);
                if (ia < 0 || ib < 0 || ic < 0
                    || ia >= m_smplh_vertices.rows()
                    || ib >= m_smplh_vertices.rows()
                    || ic >= m_smplh_vertices.rows())
                    continue;

                const Eigen::Vector3d p =
                    bc.x() * m_smplh_vertices.row(ia).transpose()
                    + bc.y() * m_smplh_vertices.row(ib).transpose()
                    + bc.z() * m_smplh_vertices.row(ic).transpose();

                float px = 0.0f;
                float py = 0.0f;
                float body_depth = 1.0f;
                if (!project_to_image_depth(view, proj, p, width, height, px, py, body_depth))
                    continue;

                const int sx = (int)std::floor(px);
                const int sy = (int)std::floor(py);
                if (sx < 0 || sy < 0 || sx >= width || sy >= height)
                    continue;

                ++out_tested_count;
                const size_t depth_idx = (size_t)(height - 1 - sy) * (size_t)width + (size_t)sx;
                const float cloth_depth = cloth_depth_gl[depth_idx];
                const bool cloth_missing_at_body = !(cloth_depth < 0.999999f);
                const bool body_visible_on_top = cloth_missing_at_body || body_depth <= cloth_depth + kDepthEps;
                if (body_visible_on_top)
                {
                    out_rgba[out_pix * 4u + 3u] = 0u;
                    ++out_removed_count;
                }
            }
        }

        return true;
    }
}
