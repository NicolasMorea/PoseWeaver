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
#include <stb_image.h>
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

        struct UvSourceData
        {
            int width = 0;
            int height = 0;
            std::vector<float> uv;   // [2, height, width]
            std::vector<float> mask; // [height, width]
        };

        static std::string lowercase_extension(const std::filesystem::path& path)
        {
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });
            return ext;
        }

        static bool load_uv_source_png(
            const std::filesystem::path& path,
            UvSourceData& out,
            std::string* out_error)
        {
            int iw = 0, ih = 0, comp = 0;
            stbi_uc* data = stbi_load(path.string().c_str(), &iw, &ih, &comp, 4);
            if (!data)
            {
                if (out_error)
                    *out_error = "could not load UV PNG: " + path.string();
                return false;
            }

            if (iw <= 0 || ih <= 0)
            {
                stbi_image_free(data);
                if (out_error)
                    *out_error = "invalid UV PNG dimensions: " + path.string();
                return false;
            }

            const size_t count = (size_t)iw * (size_t)ih;
            out.width = iw;
            out.height = ih;
            out.uv.assign(count * 2u, 0.0f);
            out.mask.assign(count, 0.0f);

            const bool has_alpha = comp >= 4;
            for (size_t i = 0; i < count; ++i)
            {
                const unsigned char r = data[i * 4u + 0u];
                const unsigned char g = data[i * 4u + 1u];
                const unsigned char b = data[i * 4u + 2u];
                const unsigned char a = data[i * 4u + 3u];
                out.uv[i] = (float)r / 255.0f;
                out.uv[count + i] = (float)g / 255.0f;
                out.mask[i] = has_alpha
                    ? (float)a / 255.0f
                    : ((r != 0u || g != 0u || b != 0u) ? 1.0f : 0.0f);
            }

            stbi_image_free(data);
            return true;
        }

        static bool load_uv_source_json(
            const std::filesystem::path& path,
            UvSourceData& out,
            std::string* out_error)
        {
            const std::string uv_json = read_text_file_displacement(path);
            if (uv_json.empty())
            {
                if (out_error)
                    *out_error = "could not read " + path.string();
                return false;
            }

            std::filesystem::path uv_npz_path = parse_json_string_field_displacement(uv_json, "uv_npz_path");
            if (uv_npz_path.empty())
                uv_npz_path = path.parent_path() / (path.stem().string() + "_uv.npz");
            if (uv_npz_path.is_relative())
                uv_npz_path = path.parent_path() / uv_npz_path;

            cnpy::NpyArray uv_arr;
            cnpy::NpyArray mask_arr;
            try
            {
                uv_arr = cnpy::npz_load(uv_npz_path.string(), "uv");
                mask_arr = cnpy::npz_load(uv_npz_path.string(), "mask");
            }
            catch (const std::exception& e)
            {
                if (out_error)
                    *out_error = std::string("could not load UV NPZ: ") + e.what();
                return false;
            }

            if (uv_arr.shape.size() != 3 || uv_arr.shape[0] < 2 || mask_arr.shape.size() != 2)
            {
                if (out_error)
                    *out_error = "unexpected UV NPZ shape";
                return false;
            }

            const int uv_h = (int)uv_arr.shape[1];
            const int uv_w = (int)uv_arr.shape[2];
            if (uv_w <= 0 || uv_h <= 0 || (int)mask_arr.shape[0] != uv_h || (int)mask_arr.shape[1] != uv_w)
            {
                if (out_error)
                    *out_error = "UV and mask dimensions do not match";
                return false;
            }
            if (!((uv_arr.word_size == sizeof(float) || uv_arr.word_size == sizeof(double))
                    && (mask_arr.word_size == sizeof(float) || mask_arr.word_size == sizeof(double))))
            {
                if (out_error)
                    *out_error = "UV NPZ arrays are not float arrays";
                return false;
            }

            const size_t count = (size_t)uv_w * (size_t)uv_h;
            out.width = uv_w;
            out.height = uv_h;
            out.uv.assign(count * 2u, 0.0f);
            out.mask.assign(count, 0.0f);
            for (size_t i = 0; i < count; ++i)
            {
                if (!npy_float_at(mask_arr, i, &out.mask[i])
                    || !npy_float_at(uv_arr, i, &out.uv[i])
                    || !npy_float_at(uv_arr, count + i, &out.uv[count + i]))
                {
                    if (out_error)
                        *out_error = "failed to read UV NPZ arrays";
                    return false;
                }
            }
            return true;
        }

        static bool load_uv_source(
            const std::filesystem::path& path,
            UvSourceData& out,
            std::string* out_error)
        {
            if (path.empty())
            {
                if (out_error)
                    *out_error = "missing UV source path";
                return false;
            }

            const std::string ext = lowercase_extension(path);
            if (ext == ".png")
                return load_uv_source_png(path, out, out_error);
            return load_uv_source_json(path, out, out_error);
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

        static void insert_depth_layer(float depth, float& first, float& second)
        {
            if (!std::isfinite(depth) || depth < 0.0f || depth > 1.0f)
                return;

            constexpr float kSameLayerEps = 1e-5f;
            if (std::abs(depth - first) <= kSameLayerEps || std::abs(depth - second) <= kSameLayerEps)
                return;

            if (depth < first)
            {
                second = first;
                first = depth;
                return;
            }
            if (depth < second)
                second = depth;
        }

        static void rasterize_depth_layers(
            const Eigen::MatrixXd& vertices,
            const Eigen::MatrixXi& faces,
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            int width,
            int height,
            std::vector<float>& first_depth,
            std::vector<float>& second_depth)
        {
            first_depth.assign((size_t)width * (size_t)height, 1.0f);
            second_depth.assign((size_t)width * (size_t)height, 1.0f);

            if (width <= 0 || height <= 0 || vertices.rows() <= 0 || faces.rows() <= 0)
                return;

            for (int fi = 0; fi < faces.rows(); ++fi)
            {
                const int i0 = faces(fi, 0);
                const int i1 = faces(fi, 1);
                const int i2 = faces(fi, 2);
                if (i0 < 0 || i1 < 0 || i2 < 0
                    || i0 >= vertices.rows()
                    || i1 >= vertices.rows()
                    || i2 >= vertices.rows())
                    continue;

                float x0 = 0.0f;
                float y0 = 0.0f;
                float z0 = 1.0f;
                float x1 = 0.0f;
                float y1 = 0.0f;
                float z1 = 1.0f;
                float x2 = 0.0f;
                float y2 = 0.0f;
                float z2 = 1.0f;
                if (!project_to_image_depth(view, proj, vertices.row(i0).transpose(), width, height, x0, y0, z0)
                    || !project_to_image_depth(view, proj, vertices.row(i1).transpose(), width, height, x1, y1, z1)
                    || !project_to_image_depth(view, proj, vertices.row(i2).transpose(), width, height, x2, y2, z2))
                    continue;

                const float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                if (!(std::abs(area) > 1e-8f) || !std::isfinite(area))
                    continue;

                const int min_x = std::clamp((int)std::floor(std::min({ x0, x1, x2 })), 0, width - 1);
                const int max_x = std::clamp((int)std::ceil(std::max({ x0, x1, x2 })), 0, width - 1);
                const int min_y = std::clamp((int)std::floor(std::min({ y0, y1, y2 })), 0, height - 1);
                const int max_y = std::clamp((int)std::ceil(std::max({ y0, y1, y2 })), 0, height - 1);

                for (int py = min_y; py <= max_y; ++py)
                {
                    const float sy = (float)py + 0.5f;
                    for (int px = min_x; px <= max_x; ++px)
                    {
                        const float sx = (float)px + 0.5f;
                        const float w0 = ((x1 - sx) * (y2 - sy) - (y1 - sy) * (x2 - sx)) / area;
                        const float w1 = ((x2 - sx) * (y0 - sy) - (y2 - sy) * (x0 - sx)) / area;
                        const float w2 = 1.0f - w0 - w1;
                        constexpr float kEdgeEps = -1e-5f;
                        if (w0 < kEdgeEps || w1 < kEdgeEps || w2 < kEdgeEps)
                            continue;

                        const float depth = w0 * z0 + w1 * z1 + w2 * z2;
                        const size_t idx = (size_t)(height - 1 - py) * (size_t)width + (size_t)px;
                        insert_depth_layer(depth, first_depth[idx], second_depth[idx]);
                    }
                }
            }
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

        std::filesystem::path uv_source_path = state.canva_displacement_uv_json_path;
        if (uv_source_path.empty())
        {
            fail("missing UV source path");
            return;
        }

        UvSourceData uv_source;
        std::string uv_source_error;
        if (!load_uv_source(uv_source_path, uv_source, &uv_source_error))
        {
            fail(uv_source_error.empty() ? "could not load UV source" : uv_source_error);
            return;
        }

        const int uv_h = uv_source.height;
        const int uv_w = uv_source.width;

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
                mask = uv_source.mask[pix];
                u = uv_source.uv[pix];
                v = uv_source.uv[(size_t)uv_h * (size_t)uv_w + pix];
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
        float uv_from_cloth_scale,
        float uv_from_cloth_offset_x,
        float uv_from_cloth_offset_y,
        std::vector<unsigned char>& out_rgba,
        std::vector<unsigned char>& out_debug_rgba,
        int& out_tested_count,
        int& out_removed_count,
        std::string* out_error)
    {
        out_rgba.clear();
        out_debug_rgba.clear();
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
        if (!(uv_from_cloth_scale > 1e-6f)
            || !std::isfinite(uv_from_cloth_scale)
            || !std::isfinite(uv_from_cloth_offset_x)
            || !std::isfinite(uv_from_cloth_offset_y))
            return fail("invalid UV-to-cloth layer transform");
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
        auto& smpl = m_viewer.data(m_smplh_mesh_index);

        std::filesystem::path uv_source_path = uv_json_path_string;
        if (uv_source_path.empty())
            return fail("missing UV source path");

        UvSourceData uv_source;
        std::string uv_source_error;
        if (!load_uv_source(uv_source_path, uv_source, &uv_source_error))
            return fail(uv_source_error.empty() ? "could not load UV source" : uv_source_error);

        const int uv_h = uv_source.height;
        const int uv_w = uv_source.width;

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
        const bool old_smpl_visible = smpl.is_visible;

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

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (smpl.dirty)
        {
            smpl.updateGL(smpl, smpl.invert_normals, smpl.meshgl);
            smpl.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }

        smpl.set_visible(true);
        if (smpl.meshgl.is_initialized)
        {
            smpl.meshgl.bind_mesh();
            const GLuint prog = smpl.meshgl.shader_mesh;
            set_uniform_matrix4_if_present(prog, "view", core.view);
            set_uniform_matrix4_if_present(prog, "proj", core.proj);
            set_uniform_matrix4_if_present(prog, "normal_matrix", core.norm);
            set_uniform_1f_if_present(prog, "texture_factor", 0.0f);
            set_uniform_1f_if_present(prog, "matcap_factor", 0.0f);
            set_uniform_1f_if_present(prog, "double_sided", 1.0f);
            set_uniform_1i_if_present(prog, "toon_outline_pass", 0);
            set_uniform_1f_if_present(prog, "toon_outline_width", 0.0f);
            smpl.meshgl.draw_mesh(true);
        }

        std::vector<float> smpl_depth_gl((size_t)width * (size_t)height, 1.0f);
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, smpl_depth_gl.data());

        garment.is_visible = old_garment_visible;
        smpl.is_visible = old_smpl_visible;
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
        out_debug_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        const float depth_margin = std::clamp(state.cloth_fitting_depth_margin, 0.0f, 0.05f);
        const size_t uv_count = (size_t)uv_w * (size_t)uv_h;
        std::vector<float> cloth_depth_cpu;
        std::vector<float> cloth_second_depth_cpu;
        rasterize_depth_layers(
            garment.V,
            garment.F,
            view,
            proj,
            width,
            height,
            cloth_depth_cpu,
            cloth_second_depth_cpu);

        auto set_debug_pixel = [&](size_t pix, unsigned char r, unsigned char g, unsigned char b)
        {
            unsigned char* dst = out_debug_rgba.data() + pix * 4u;
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
            dst[3] = 255u;
        };

        // Cloth fitting debug colors:
        // grey=no UV, blue=UV miss, cyan=projection miss,
        // black=UV pixel outside source cloth with no cloth over match,
        // dark grey=UV pixel outside source cloth with cloth over match,
        // red=no cloth, orange=matching SMPL occludes cloth,
        // crimson=other SMPL occludes cloth but source cloth remains visible,
        // green=cloth on match, yellow=cloth over other SMPL, magenta=two cloth layers,
        // purple=two cloth layers over another SMPL layer.
        for (int y = 0; y < height; ++y)
        {
            const float uv_layer_y =
                uv_from_cloth_scale * ((float)y + 0.5f) + uv_from_cloth_offset_y;
            const bool uv_y_valid = uv_layer_y >= 0.0f && uv_layer_y < (float)height;
            const int uv_y = uv_y_valid
                ? std::clamp(
                    (int)std::floor((uv_layer_y / (float)height) * (float)uv_h),
                    0,
                    uv_h - 1)
                : 0;
            for (int x = 0; x < width; ++x)
            {
                const size_t out_pix = (size_t)y * (size_t)width + (size_t)x;
                const bool source_cloth_pixel = out_rgba[out_pix * 4u + 3u] != 0u;

                const float uv_layer_x =
                    uv_from_cloth_scale * ((float)x + 0.5f) + uv_from_cloth_offset_x;
                const bool uv_x_valid = uv_layer_x >= 0.0f && uv_layer_x < (float)width;
                const int uv_x = uv_x_valid
                    ? std::clamp(
                        (int)std::floor((uv_layer_x / (float)width) * (float)uv_w),
                        0,
                        uv_w - 1)
                    : 0;
                const size_t uv_pix = (size_t)uv_y * (size_t)uv_w + (size_t)uv_x;

                float mask = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (uv_x_valid && uv_y_valid)
                {
                    mask = uv_source.mask[uv_pix];
                    u = uv_source.uv[uv_pix];
                    v = uv_source.uv[uv_count + uv_pix];
                }
                if (!(mask > 0.5f) || !std::isfinite(u) || !std::isfinite(v))
                {
                    if (source_cloth_pixel)
                        set_debug_pixel(out_pix, 128u, 128u, 128u);
                    continue;
                }

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
                    set_debug_pixel(out_pix, 0u, 96u, 255u);
                    continue;
                }

                const int ia = m_smplh_model.F(face_idx, 0);
                const int ib = m_smplh_model.F(face_idx, 1);
                const int ic = m_smplh_model.F(face_idx, 2);
                if (ia < 0 || ib < 0 || ic < 0
                    || ia >= m_smplh_vertices.rows()
                    || ib >= m_smplh_vertices.rows()
                    || ic >= m_smplh_vertices.rows())
                {
                    set_debug_pixel(out_pix, 0u, 96u, 255u);
                    continue;
                }

                const Eigen::Vector3d p =
                    bc.x() * m_smplh_vertices.row(ia).transpose()
                    + bc.y() * m_smplh_vertices.row(ib).transpose()
                    + bc.z() * m_smplh_vertices.row(ic).transpose();

                float px = 0.0f;
                float py = 0.0f;
                float body_depth = 1.0f;
                if (!project_to_image_depth(view, proj, p, width, height, px, py, body_depth))
                {
                    set_debug_pixel(out_pix, 0u, 200u, 220u);
                    continue;
                }

                const int sx = (int)std::floor(px);
                const int sy = (int)std::floor(py);
                if (sx < 0 || sy < 0 || sx >= width || sy >= height)
                {
                    set_debug_pixel(out_pix, 0u, 200u, 220u);
                    continue;
                }

                if (source_cloth_pixel)
                    ++out_tested_count;
                const size_t depth_idx = (size_t)(height - 1 - sy) * (size_t)width + (size_t)sx;
                const float cloth_depth = cloth_depth_gl[depth_idx];
                const float cloth_second_depth = cloth_second_depth_cpu.size() == (size_t)width * (size_t)height
                    ? cloth_second_depth_cpu[depth_idx]
                    : 1.0f;
                const bool cloth_present_at_body = cloth_depth < 0.999999f;
                const bool cloth_on_top_of_match =
                    cloth_present_at_body
                    && cloth_depth + depth_margin < body_depth;
                const float nearest_smpl_depth = smpl_depth_gl[depth_idx] < 0.999999f
                    ? smpl_depth_gl[depth_idx]
                    : body_depth;
                const bool other_smpl_layer_on_top = nearest_smpl_depth + depth_margin < body_depth;
                const bool two_cloth_layers_on_top =
                    cloth_second_depth < 0.999999f
                    && cloth_second_depth + depth_margin < body_depth;
                const bool smpl_visible_on_top = !cloth_present_at_body
                    || nearest_smpl_depth + depth_margin < cloth_depth;
                if (!source_cloth_pixel)
                {
                    if (cloth_on_top_of_match)
                        set_debug_pixel(out_pix, 48u, 48u, 48u);
                    else
                        set_debug_pixel(out_pix, 0u, 0u, 0u);
                }
                else if (smpl_visible_on_top)
                {
                    if (other_smpl_layer_on_top)
                    {
                        set_debug_pixel(out_pix, 190u, 0u, 32u);
                    }
                    else
                    {
                        out_rgba[out_pix * 4u + 3u] = 0u;
                        if (!cloth_present_at_body)
                            set_debug_pixel(out_pix, 255u, 32u, 32u);
                        else
                            set_debug_pixel(out_pix, 255u, 144u, 0u);
                        ++out_removed_count;
                    }
                }
                else if (other_smpl_layer_on_top && two_cloth_layers_on_top)
                {
                    set_debug_pixel(out_pix, 160u, 64u, 255u);
                }
                else if (other_smpl_layer_on_top)
                {
                    set_debug_pixel(out_pix, 255u, 224u, 0u);
                }
                else if (two_cloth_layers_on_top)
                {
                    set_debug_pixel(out_pix, 255u, 0u, 255u);
                }
                else
                {
                    set_debug_pixel(out_pix, 32u, 220u, 80u);
                }
            }
        }

        return true;
    }
}
