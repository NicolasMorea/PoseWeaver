#include "render/Renderer.h"
#include "render/RendererInternal.h"

#include <glad/glad.h>

#include <algorithm>
#include <vector>

namespace poseweaver
{
    namespace
    {
        static std::vector<unsigned char> make_uv_background_rgba8(int w, int h)
        {
            std::vector<unsigned char> px;
            if (w <= 0 || h <= 0)
                return px;

            px.resize((size_t)w * (size_t)h * 4u);
            for (int y = 0; y < h; ++y)
            {
                const float v = (h > 1) ? (float)(h - 1 - y) / (float)(h - 1) : 0.0f; // v increases upward
                for (int x = 0; x < w; ++x)
                {
                    const float u = (w > 1) ? (float)x / (float)(w - 1) : 0.0f;

                    unsigned char r = (unsigned char)std::clamp<int>((int)(u * 255.0f + 0.5f), 0, 255);
                    unsigned char g = (unsigned char)std::clamp<int>((int)(v * 255.0f + 0.5f), 0, 255);
                    unsigned char b = render_internal::kUvBlueByte;
                    unsigned char a = 255;

                    // Subtle UV grid lines (every 0.1)
                    const int gx = (w >= 10) ? (w / 10) : w;
                    const int gy = (h >= 10) ? (h / 10) : h;
                    if ((gx > 0 && (x % gx) == 0) || (gy > 0 && (y % gy) == 0))
                    {
                        r = (unsigned char)(r * 3 / 5);
                        g = (unsigned char)(g * 3 / 5);
                        b = (unsigned char)(b * 3 / 5);
                    }

                    const size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 4u;
                    px[idx + 0] = r;
                    px[idx + 1] = g;
                    px[idx + 2] = b;
                    px[idx + 3] = a;
                }
            }
            return px;
        }
    }

    void Renderer::generate_uv_debug_image(AppState& state)
    {
        console_add_line(state, "Generating UV debug image...");

        // Generate a square UV debug background.
        constexpr int kTexSize = 1024;
        const int w = kTexSize;
        const int h = kTexSize;

        const auto pixels = make_uv_background_rgba8(w, h);
        if (pixels.empty())
            return;

        if (m_uv_debug_texture_id == 0)
            glGenTextures(1, &m_uv_debug_texture_id);

        glBindTexture(GL_TEXTURE_2D, m_uv_debug_texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // (Re)allocate each time for simplicity.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        m_uv_debug_texture_w = w;
        m_uv_debug_texture_h = h;

        // Build/refresh UV wireframe segments.
        regenerate_uv_wireframe_segments();
        console_add_line(state,
            std::string("UV debug image ready: ") + std::to_string(w) + "x" + std::to_string(h) +
            " (segments=" + std::to_string((int)m_smpl_uv_segments.size()) + ")");

        // UV view assets refreshed -> recompute garment UV overlay if present.
        m_garment_uv_overlay_valid = false;
    }
}
