#version 150

uniform mat4 view;
uniform mat4 proj;
uniform vec4 fixed_color;

in vec3 position_eye;
in vec3 normal_eye;

uniform bool is_directional_light;
uniform bool is_shadow_mapping;
uniform bool shadow_pass;

uniform vec3 light_position_eye;
uniform vec3 light_position_eye2;
uniform float light2_factor;

vec3 Ls = vec3(1,1,1);
vec3 Ld = vec3(1,1,1);
vec3 La = vec3(1,1,1);

in vec4 Ksi;
in vec4 Kdi;
in vec4 Kai;

in vec2 texcoordi;
uniform sampler2D tex;

uniform float specular_exponent;
uniform float lighting_factor;
uniform float texture_factor;
uniform float matcap_factor;
uniform float double_sided;
uniform bool scene_shaded;

uniform sampler2D shadow_tex;
in vec4 position_shadow;

out vec4 outColor;

void main()
{
  if(shadow_pass)
  {
    outColor = vec4(0.56,0.85,0.77,1.);
    return;
  }

  vec3 dir1 = light_position_eye;
  vec3 dir2 = light_position_eye2;
  if(!is_directional_light)
  {
    dir1 = normalize(light_position_eye - position_eye);
    dir2 = normalize(light_position_eye2 - position_eye);
  }

  float shadow = 1.0;
  if(is_shadow_mapping)
  {
    vec3 shadow_pos = (position_shadow.xyz / position_shadow.w) * 0.5 + 0.5;
    float currentDepth = shadow_pos.z;
    float ddd = max(dot(normalize(normal_eye), dir1),0);
    float bias = max(0.02 * (1.0 - ddd), 0.005);
    if(shadow_pos.z < 1.0)
    {
      float closestDepth = texture(shadow_tex, shadow_pos.xy).r;
      shadow = currentDepth - bias >= closestDepth ? 0.0 : 1.0;
      vec2 texelSize = 1.0 / textureSize(shadow_tex, 0);
      for(int x=-1; x<=1; x+=2)
      {
        for(int y=-1; y<=1; y+=2)
        {
          float pcfDepth = texture(shadow_tex, shadow_pos.xy + vec2(x,y) * texelSize).r;
          shadow += currentDepth - bias >= pcfDepth ? 0.0 : 1.0;
        }
      }
      shadow /= 5.0;
    }
  }

  if(matcap_factor == 1.0f)
  {
    vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
    outColor = mix(Kai, texture(tex, uv), shadow);
  }
  else
  {
    vec3 Ia = La * vec3(Kai);

    // Reduce "shininess" by dampening specular highlights and capping the exponent.
    const float kSpecularStrength = 0.15;
    float shininess = clamp(specular_exponent, 1.0, 32.0);

    float dp1 = dot(dir1, normalize(normal_eye));
    float cdp1 = abs(max(dp1, -double_sided));
    vec3 Id1 = Ld * vec3(Kdi) * cdp1;
    vec3 r1 = reflect(-dir1, normalize(normal_eye));
    vec3 v = normalize(-position_eye);
    float sp1 = dot(r1, v);
    sp1 = float(abs(dp1)==dp1) * abs(max(sp1, -double_sided));
    float sf1 = pow(sp1, shininess);
    vec3 Is1 = kSpecularStrength * Ls * vec3(Ksi) * sf1;

    float dp2 = dot(dir2, normalize(normal_eye));
    float cdp2 = abs(max(dp2, -double_sided));
    vec3 Id2 = Ld * vec3(Kdi) * cdp2;
    vec3 r2 = reflect(-dir2, normalize(normal_eye));
    float sp2 = dot(r2, v);
    sp2 = float(abs(dp2)==dp2) * abs(max(sp2, -double_sided));
    float sf2 = pow(sp2, shininess);
    vec3 Is2 = kSpecularStrength * Ls * vec3(Ksi) * sf2;

    vec3 Id = Id1 + light2_factor * Id2;
    vec3 Is = Is1 + light2_factor * Is2;

    vec4 color = vec4(
      Ia + shadow*(lighting_factor*(Is + Id) + (1.0-lighting_factor)*vec3(Kdi)),
      (Kai.a+Ksi.a+Kdi.a)/3);

    if (scene_shaded)
    {
      vec4 base_color = mix(Kdi, texture(tex, texcoordi), texture_factor);
      float diffuse = (cdp1 + light2_factor * cdp2) / (1.0 + light2_factor);
      float illumination = 0.22 + 0.78 * shadow * diffuse;
      outColor = vec4(base_color.rgb * illumination + shadow * lighting_factor * Is, base_color.a);
    }
    else
    {
      outColor = mix(color, texture(tex, texcoordi), texture_factor);
    }
    if (fixed_color != vec4(0.0)) outColor = fixed_color;
  }
}
