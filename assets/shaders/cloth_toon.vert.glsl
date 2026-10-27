#version 150

uniform mat4 view;
uniform mat4 proj;
uniform mat4 normal_matrix;

uniform bool toon_outline_pass;
uniform float toon_outline_width;
uniform vec2 viewport_size;

in vec3 position;
in vec3 normal;

out vec3 position_eye;
out vec3 normal_eye;

in vec4 Ka;
in vec4 Kd;
in vec4 Ks;
in vec2 texcoord;

out vec4 Kai;
out vec4 Kdi;
out vec4 Ksi;
out vec2 texcoordi;

void main()
{
  normal_eye = normalize(vec3(normal_matrix * vec4(normal, 0.0)));
  position_eye = vec3(view * vec4(position, 1.0));

  Kai = Ka;
  Kdi = Kd;
  Ksi = Ks;
  texcoordi = texcoord;

  vec4 clip_position = proj * vec4(position_eye, 1.0);

  if(toon_outline_pass)
  {
    vec4 clip_offset = proj * vec4(position_eye + normal_eye, 1.0);
    vec2 base_ndc = clip_position.xy / clip_position.w;
    vec2 offset_ndc = clip_offset.xy / clip_offset.w;
    vec2 outline_dir = offset_ndc - base_ndc;
    float outline_len = length(outline_dir);

    if(outline_len > 1e-6)
    {
      vec2 pixel_ndc = vec2(2.0) / max(viewport_size, vec2(1.0));
      clip_position.xy += (outline_dir / outline_len) * pixel_ndc * toon_outline_width * clip_position.w;
    }
  }

  gl_Position = clip_position;
}
