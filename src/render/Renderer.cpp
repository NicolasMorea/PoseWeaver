#include "render/Renderer.h"

#include <cmath>
#include <vector>
#include <glad/glad.h>

namespace poseweaver
{
    namespace
    {
        struct Vec3
        {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
        };

        struct Mat4
        {
            float m[16] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            };
        };

        static Mat4 mul(const Mat4& a, const Mat4& b)
        {
            Mat4 out;
            for (int c = 0; c < 4; ++c)
            {
                for (int r = 0; r < 4; ++r)
                {
                    out.m[c * 4 + r] =
                        a.m[0 * 4 + r] * b.m[c * 4 + 0] +
                        a.m[1 * 4 + r] * b.m[c * 4 + 1] +
                        a.m[2 * 4 + r] * b.m[c * 4 + 2] +
                        a.m[3 * 4 + r] * b.m[c * 4 + 3];
                }
            }
            return out;
        }

        static Mat4 translate(float x, float y, float z)
        {
            Mat4 t;
            t.m[12] = x;
            t.m[13] = y;
            t.m[14] = z;
            return t;
        }

        static Mat4 rotate_x(float deg)
        {
            const float a = deg * 3.14159265f / 180.0f;
            const float c = std::cos(a);
            const float s = std::sin(a);
            Mat4 r;
            r.m[5] = c;
            r.m[6] = s;
            r.m[9] = -s;
            r.m[10] = c;
            return r;
        }

        static Mat4 rotate_y(float deg)
        {
            const float a = deg * 3.14159265f / 180.0f;
            const float c = std::cos(a);
            const float s = std::sin(a);
            Mat4 r;
            r.m[0] = c;
            r.m[2] = -s;
            r.m[8] = s;
            r.m[10] = c;
            return r;
        }

        static Mat4 perspective(float fov_y_deg, float aspect, float z_near, float z_far)
        {
            const float fov = fov_y_deg * 3.14159265f / 180.0f;
            const float f = 1.0f / std::tan(fov * 0.5f);
            Mat4 p{};
            for (float& v : p.m)
                v = 0.0f;
            p.m[0] = f / aspect;
            p.m[5] = f;
            p.m[10] = (z_far + z_near) / (z_near - z_far);
            p.m[11] = -1.0f;
            p.m[14] = (2.0f * z_far * z_near) / (z_near - z_far);
            return p;
        }

        struct Vertex
        {
            float px, py, pz;
            float cr, cg, cb;
        };

        static unsigned int compile_shader(unsigned int type, const char* src)
        {
            const unsigned int s = glCreateShader(type);
            glShaderSource(s, 1, &src, nullptr);
            glCompileShader(s);

            int ok = 0;
            glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
            if (!ok)
            {
                glDeleteShader(s);
                return 0;
            }
            return s;
        }

        static unsigned int link_program(unsigned int vs, unsigned int fs)
        {
            const unsigned int p = glCreateProgram();
            glAttachShader(p, vs);
            glAttachShader(p, fs);
            glLinkProgram(p);

            int ok = 0;
            glGetProgramiv(p, GL_LINK_STATUS, &ok);
            if (!ok)
            {
                glDeleteProgram(p);
                return 0;
            }
            return p;
        }

        static Mat4 build_mvp(int width, int height)
        {
            const float aspect = (height > 0) ? (float)width / (float)height : 1.0f;
            const Mat4 p = perspective(45.0f, aspect, 0.05f, 100.0f);
            Mat4 v = mul(rotate_y(-35.0f), rotate_x(25.0f));
            v = mul(translate(0.0f, -0.7f, -4.0f), v);
            return mul(p, v);
        }

        static void push_line(std::vector<Vertex>& v, Vec3 a, Vec3 b, Vec3 c)
        {
            v.push_back(Vertex{ a.x, a.y, a.z, c.x, c.y, c.z });
            v.push_back(Vertex{ b.x, b.y, b.z, c.x, c.y, c.z });
        }

        static void build_scene_lines(std::vector<Vertex>& out)
        {
            out.clear();
            out.reserve(8192);

            push_line(out, { 0, 0, 0 }, { 0.75f, 0, 0 }, { 1.0f, 0.2f, 0.2f });
            push_line(out, { 0, 0, 0 }, { 0, 0.75f, 0 }, { 0.2f, 1.0f, 0.2f });
            push_line(out, { 0, 0, 0 }, { 0, 0, 0.75f }, { 0.2f, 0.4f, 1.0f });

            const float extent = 2.0f;
            const int steps = 20;
            const Vec3 grid_col{ 0.35f, 0.35f, 0.35f };
            for (int i = 0; i <= steps; ++i)
            {
                const float t = (float)i / (float)steps;
                const float x = -extent + 2.0f * extent * t;
                push_line(out, { x, 0, -extent }, { x, 0, extent }, grid_col);

                const float z = -extent + 2.0f * extent * t;
                push_line(out, { -extent, 0, z }, { extent, 0, z }, grid_col);
            }

            const Vec3 cloth_col{ 0.95f, 0.85f, 0.2f };
            const float size = 0.8f;
            const int n = 18;

            for (int j = 0; j <= n; ++j)
            {
                const float v = (float)j / (float)n;
                const float z = -size + 2.0f * size * v;
                for (int i = 0; i < n; ++i)
                {
                    const float u0 = (float)i / (float)n;
                    const float u1 = (float)(i + 1) / (float)n;
                    const float x0 = -size + 2.0f * size * u0;
                    const float x1 = -size + 2.0f * size * u1;
                    push_line(out, { x0, 1.0f, z }, { x1, 1.0f, z }, cloth_col);
                }
            }

            for (int i = 0; i <= n; ++i)
            {
                const float u = (float)i / (float)n;
                const float x = -size + 2.0f * size * u;
                for (int j = 0; j < n; ++j)
                {
                    const float v0 = (float)j / (float)n;
                    const float v1 = (float)(j + 1) / (float)n;
                    const float z0 = -size + 2.0f * size * v0;
                    const float z1 = -size + 2.0f * size * v1;
                    push_line(out, { x, 1.0f, z0 }, { x, 1.0f, z1 }, cloth_col);
                }
            }
        }
    }

    bool Renderer::ensure_initialized()
    {
        if (m_program != 0)
            return true;

        const char* vs_src = R"(
    #version 330 core
    layout(location = 0) in vec3 a_pos;
    layout(location = 1) in vec3 a_col;
    uniform mat4 u_mvp;
    out vec3 v_col;
    void main() {
    v_col = a_col;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    }
    )";

        const char* fs_src = R"(
    #version 330 core
    in vec3 v_col;
    out vec4 out_col;
    void main() {
    out_col = vec4(v_col, 1.0);
    }
    )";

        const unsigned int vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        const unsigned int fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
        if (!vs || !fs)
            return false;

        m_program = link_program(vs, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);
        if (!m_program)
            return false;

        m_u_mvp = glGetUniformLocation(m_program, "u_mvp");
        if (m_u_mvp < 0)
            return false;

        glGenVertexArrays(1, &m_vao);
        glGenBuffers(1, &m_vbo);
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        return true;
    }

    void Renderer::shutdown()
    {
        if (m_vbo)
        {
            glDeleteBuffers(1, &m_vbo);
            m_vbo = 0;
        }
        if (m_vao)
        {
            glDeleteVertexArrays(1, &m_vao);
            m_vao = 0;
        }
        if (m_program)
        {
            glDeleteProgram(m_program);
            m_program = 0;
        }
        m_u_mvp = -1;
    }

    void Renderer::render_cloth_view(const AppState& state)
    {
        if (!ensure_initialized())
            return;

        const Mat4 mvp = build_mvp(state.cloth_view.width, state.cloth_view.height);

        std::vector<Vertex> lines;
        build_scene_lines(lines);

        glUseProgram(m_program);
        glUniformMatrix4fv(m_u_mvp, 1, GL_FALSE, mvp.m);
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(lines.size() * sizeof(Vertex)), lines.data(), GL_DYNAMIC_DRAW);

        glEnable(GL_DEPTH_TEST);
        glDrawArrays(GL_LINES, 0, (GLsizei)lines.size());

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    void Renderer::render(const AppState& state, int fb_width, int fb_height)
    {
        glViewport(0, 0, fb_width, fb_height);
        glDisable(GL_SCISSOR_TEST);
        glClearColor(0.12f, 0.12f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        if (!state.cloth_view.valid)
            return;

        const auto& r = state.cloth_view;
        glEnable(GL_SCISSOR_TEST);
        glScissor(r.x, r.y, r.width, r.height);
        glViewport(r.x, r.y, r.width, r.height);

        glClearColor(0.06f, 0.06f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render_cloth_view(state);

        glDisable(GL_SCISSOR_TEST);
    }
}
