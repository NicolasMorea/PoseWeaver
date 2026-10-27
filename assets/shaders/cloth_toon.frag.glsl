#version 150

uniform vec4 fixed_color;

in vec3 position_eye;
in vec3 normal_eye;

uniform bool is_directional_light;
uniform vec3 light_position_eye;
uniform vec3 light_position_eye2;
uniform float light2_factor;
uniform float lighting_factor;
uniform float double_sided;
uniform bool toon_outline_pass;
uniform int toon_style;
uniform vec3 cloth_color;
uniform float toon_scene_diag;
uniform sampler2D tex;
uniform float texture_factor;
uniform float material_color_factor;
uniform float surface_ink_factor;

in vec4 Kai;
in vec4 Kdi;
in vec4 Ksi;
in vec2 texcoordi;

out vec4 outColor;

void main()
{
  if(toon_outline_pass)
  {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  if(fixed_color != vec4(0.0))
  {
    outColor = fixed_color;
    return;
  }

  vec3 n = normalize(normal_eye);
  if(double_sided > 0.5 && !gl_FrontFacing)
  {
    n = -n;
  }

  vec3 dir1 = light_position_eye;
  vec3 dir2 = light_position_eye2;
  if(!is_directional_light)
  {
    dir1 = normalize(light_position_eye - position_eye);
    dir2 = normalize(light_position_eye2 - position_eye);
  }

  vec3 view_dir = normalize(-position_eye);
  float ndl1 = max(dot(n, dir1), 0.0);
  float ndl2 = max(dot(n, dir2), 0.0);
  float lit = clamp(0.38 + lighting_factor * (0.56 * ndl1 + 0.28 * light2_factor * ndl2), 0.0, 1.0);

  bool clean_style = toon_style >= 2;
  float pixel_footprint = length(fwidth(position_eye)) / max(toon_scene_diag, 1e-4);
  float distance_smoothing = smoothstep(0.0018, 0.0060, pixel_footprint);

  float band_aa = max(0.012, 1.5 * fwidth(lit));
  band_aa = max(band_aa, mix(0.0, 0.08, distance_smoothing));
  float mid_band = smoothstep(0.48 - band_aa, 0.48 + band_aa, lit);
  float light_band = smoothstep(0.72 - band_aa, 0.72 + band_aa, lit);
  float toon = mix(0.78, 0.91, mid_band);
  toon = mix(toon, 1.0, light_band);

  float normal_change = length(fwidth(n));
  float coarse_detail = smoothstep(0.12, 0.38, normal_change);
  float smooth_tone = mix(0.78, 1.0, smoothstep(0.34, 0.88, lit));
  toon = mix(toon, smooth_tone, max(coarse_detail, distance_smoothing));
  vec4 material_color = mix(vec4(cloth_color, 1.0), Kdi, material_color_factor);
  vec4 texture_color = texture(tex, texcoordi);
  vec4 material = mix(material_color, texture_color, texture_factor);
  vec3 base = material.rgb * toon;

  float rim = 1.0 - abs(dot(n, view_dir));
  float rim_aa = max(0.018, 1.5 * fwidth(rim));
  float rim_ink = clean_style
    ? smoothstep(0.72 - rim_aa, 0.91 + rim_aa, rim)
    : smoothstep(0.64 - rim_aa, 0.86 + rim_aa, rim);

  float detailed_crease = smoothstep(0.10, 0.24, normal_change)
    * (1.0 - smoothstep(0.24, 0.48, normal_change));
  float clean_crease = smoothstep(0.035, 0.12, normal_change)
    * (1.0 - smoothstep(0.22, 0.50, normal_change));
  clean_crease *= clean_crease;
  float crease_visibility = clean_style
    ? 1.0 - smoothstep(0.004, 0.012, pixel_footprint)
    : 1.0 - distance_smoothing;
  float crease_ink = (clean_style ? clean_crease : detailed_crease) * crease_visibility;

  float ink = surface_ink_factor * clamp(max(rim_ink, crease_ink), 0.0, 1.0);
  float alpha = material.a * clamp((Kai.a + Kdi.a + Ksi.a) / 3.0, 0.0, 1.0);
  outColor = vec4(mix(base, vec3(0.0), ink), alpha);
}
