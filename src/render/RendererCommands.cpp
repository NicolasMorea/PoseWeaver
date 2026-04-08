#include "render/Renderer.h"

namespace poseweaver
{
    void Renderer::call_commands_from_state(AppState& state)
    {
        const auto consume_request = [&](auto current, auto& last, auto&& fn)
        {
            if (current == last)
                return;
            last = current;
            fn();
        };

        consume_request(state.garment_pushout_request_id, m_last_garment_pushout_request_id, [&]()
            { process_garment_pushout(state); });
        consume_request(state.garment_ease_request_id, m_last_garment_ease_request_id, [&]()
            { process_garment_ease(state); });
        consume_request(state.garment_prefit_request_id, m_last_garment_prefit_request_id, [&]()
            { process_garment_prefit(state); });
        consume_request(state.garment_relax_request_id, m_last_garment_relax_request_id, [&]()
            { process_garment_relax(state); });
        consume_request(state.uv_image_request_id, m_last_uv_image_request_id, [&]()
            { generate_uv_debug_image(state); });
    }
}
