#define STB_IMAGE_IMPLEMENTATION

// Keep stb_image lean: we only need standard 8-bit images (PNG).
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM

#include <stb_image.h>
