#include "core/Assets.h"
#include "core/GlbGarment.h"
#include "core/ObjMaterialTexture.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    bool write_text(const std::filesystem::path& path, const std::string& text);

    void append_u32(std::vector<unsigned char>& bytes, std::uint32_t value)
    {
        for (int i = 0; i < 4; ++i)
            bytes.push_back(static_cast<unsigned char>((value >> (i * 8)) & 0xffu));
    }

    template <typename T>
    void append_value(std::vector<unsigned char>& bytes, const T& value)
    {
        const auto* raw = reinterpret_cast<const unsigned char*>(&value);
        bytes.insert(bytes.end(), raw, raw + sizeof(T));
    }

    bool write_test_glb(const std::filesystem::path& path)
    {
        std::vector<unsigned char> bin;
        const float positions[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        const float uv[] = { 0, 0, 1, 0, 0, 1 };
        const std::uint16_t indices[] = { 0, 1, 2 };
        for (float value : positions)
            append_value(bin, value);
        for (float value : uv)
            append_value(bin, value);
        for (std::uint16_t value : indices)
            append_value(bin, value);
        while (bin.size() < 68u)
            bin.push_back(0u);
        const unsigned char image[] = {
            0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            1u, 0u, 1u, 0u, 24u, 0u,
            20u, 40u, 230u,
        };
        bin.insert(bin.end(), std::begin(image), std::end(image));
        const std::size_t image_size = sizeof(image);
        while ((bin.size() & 3u) != 0u)
            bin.push_back(0u);

        std::string json =
            "{\"asset\":{\"version\":\"2.0\"},"
            "\"buffers\":[{\"byteLength\":" + std::to_string(bin.size()) + "}],"
            "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
            "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":24},"
            "{\"buffer\":0,\"byteOffset\":60,\"byteLength\":6},"
            "{\"buffer\":0,\"byteOffset\":68,\"byteLength\":" + std::to_string(image_size) + "}],"
            "\"accessors\":["
            "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
            "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
            "{\"bufferView\":2,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}],"
            "\"images\":[{\"bufferView\":3,\"mimeType\":\"image/png\"}],"
            "\"textures\":[{\"source\":0}],"
            "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0}}}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"TEXCOORD_0\":1},"
            "\"indices\":2,\"material\":0}]}],"
            "\"nodes\":[{\"mesh\":0,\"translation\":[2,3,4]}],"
            "\"scenes\":[{\"nodes\":[0]}],\"scene\":0}";
        while ((json.size() & 3u) != 0u)
            json.push_back(' ');

        std::vector<unsigned char> glb;
        append_u32(glb, 0x46546c67u);
        append_u32(glb, 2u);
        append_u32(glb, static_cast<std::uint32_t>(12u + 8u + json.size() + 8u + bin.size()));
        append_u32(glb, static_cast<std::uint32_t>(json.size()));
        append_u32(glb, 0x4e4f534au);
        glb.insert(glb.end(), json.begin(), json.end());
        append_u32(glb, static_cast<std::uint32_t>(bin.size()));
        append_u32(glb, 0x004e4942u);
        glb.insert(glb.end(), bin.begin(), bin.end());

        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(glb.data()), static_cast<std::streamsize>(glb.size()));
        return output.good();
    }

    bool write_test_gltf(const std::filesystem::path& path)
    {
        std::vector<unsigned char> bin;
        const float positions[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        const float uv[] = { 0, 0, 1, 0, 0, 1 };
        const std::uint16_t indices[] = { 0, 1, 2 };
        for (float value : positions)
            append_value(bin, value);
        for (float value : uv)
            append_value(bin, value);
        for (std::uint16_t value : indices)
            append_value(bin, value);

        std::ofstream buffer(path.parent_path() / "dress.bin", std::ios::binary);
        buffer.write(reinterpret_cast<const char*>(bin.data()), static_cast<std::streamsize>(bin.size()));
        if (!buffer.good())
            return false;

        return write_text(path,
            "{\"asset\":{\"version\":\"2.0\"},"
            "\"buffers\":[{\"uri\":\"dress.bin\",\"byteLength\":66}],"
            "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
            "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":24},"
            "{\"buffer\":0,\"byteOffset\":60,\"byteLength\":6}],"
            "\"accessors\":["
            "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
            "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
            "{\"bufferView\":2,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}],"
            "\"images\":[{\"uri\":\"cloth.tga\"}],\"textures\":[{\"source\":0}],"
            "\"materials\":[{\"pbrMetallicRoughness\":{"
            "\"baseColorFactor\":[0.5,0.25,0.75,1],\"baseColorTexture\":{\"index\":0}}}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"TEXCOORD_0\":1},"
            "\"indices\":2,\"material\":0}]}],\"nodes\":[{\"mesh\":0}],"
            "\"scenes\":[{\"nodes\":[0]}],\"scene\":0}");
    }

    bool write_text(const std::filesystem::path& path, const std::string& text)
    {
        std::ofstream output(path, std::ios::binary);
        output << text;
        return output.good();
    }

    bool write_test_tga(const std::filesystem::path& path)
    {
        const unsigned char image[] = {
            0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            1u, 0u, 1u, 0u, 24u, 0u,
            240u, 128u, 16u,
        };
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(image), static_cast<std::streamsize>(sizeof(image)));
        return output.good();
    }
}

int main()
{
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::temp_directory_path()
        / ("poseweaver_garment_asset_smoke_" + std::to_string(stamp));
    const std::filesystem::path garment_dir = root / "garments" / "nested";
    std::error_code ec;
    std::filesystem::create_directories(garment_dir, ec);
    if (ec)
        return 1;

    const std::filesystem::path obj_path = garment_dir / "shirt.obj";
    const std::filesystem::path mtl_path = garment_dir / "shirt.mtl";
    const std::filesystem::path image_path = garment_dir / "cloth.tga";
    const std::filesystem::path glb_path = garment_dir / "dress.glb";
    const std::filesystem::path gltf_path = garment_dir / "dress.gltf";
    const std::filesystem::path color_gltf_path = garment_dir / "color-only.gltf";
    const std::filesystem::path color_obj_path = garment_dir / "color-only.obj";
    const std::filesystem::path color_mtl_path = garment_dir / "color-only.mtl";
    const std::filesystem::path polygon_obj_path = garment_dir / "polygon.obj";
    const std::filesystem::path polygon_mtl_path = garment_dir / "polygon.mtl";
    if (!write_text(obj_path,
            "mtllib shirt.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "usemtl cloth\nf 1/1 2/2 3/3\n")
        || !write_text(mtl_path, "newmtl cloth\nmap_Kd cloth.tga\n")
        || !write_text(color_obj_path,
            "mtllib color-only.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "usemtl blue\nf 1 2 3\n")
        || !write_text(color_mtl_path, "newmtl blue\nKd 0.1 0.2 0.8\n")
        || !write_text(polygon_obj_path,
            "mtllib polygon.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "v 2 0 0\nv 3 0 0\nv 3 1 0\nv 2 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "usemtl blue\nf 1/1 2/2 3/3 4/4\n"
            "usemtl red\nf 5/1 6/2 7/3 8/4\n")
        || !write_text(polygon_mtl_path,
            "newmtl blue\nKd 0.1 0.2 0.8\n"
            "newmtl red\nKd 0.8 0.2 0.1\n")
        || !write_test_tga(image_path)
        || !write_test_glb(glb_path)
        || !write_test_gltf(gltf_path)
        || !write_text(color_gltf_path,
            "{\"asset\":{\"version\":\"2.0\"},"
            "\"buffers\":[{\"uri\":\"dress.bin\",\"byteLength\":66}],"
            "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
            "{\"buffer\":0,\"byteOffset\":60,\"byteLength\":6}],"
            "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
            "{\"bufferView\":1,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}],"
            "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.2,0.6,0.3,1]}}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"indices\":1,\"material\":0}]}],"
            "\"nodes\":[{\"mesh\":0}],\"scenes\":[{\"nodes\":[0]}],\"scene\":0}"))
    {
        std::filesystem::remove_all(root, ec);
        return 2;
    }

    poseweaver::AssetIndex assets(root);
    const auto& garments = assets.garments();
    if (std::find(garments.begin(), garments.end(), "nested/shirt.obj") == garments.end()
        || std::find(garments.begin(), garments.end(), "nested/dress.glb") == garments.end()
        || std::find(garments.begin(), garments.end(), "nested/dress.gltf") == garments.end()
        || std::find(garments.begin(), garments.end(), "nested/color-only.gltf") == garments.end())
    {
        std::cerr << "Recursive garment discovery failed\n";
        std::filesystem::remove_all(root, ec);
        return 3;
    }
    const auto& shirt_textures = assets.garment_textures("nested/shirt.obj");
    if (std::find(garments.begin(), garments.end(), "nested/color-only.obj") != garments.end()
        || shirt_textures.size() != 2
        || shirt_textures[0].file != "nested/shirt.obj"
        || shirt_textures[0].name != "Default"
        || shirt_textures[1].file != "nested/color-only.obj")
    {
        std::cerr << "Garment texture variant grouping failed\n";
        std::filesystem::remove_all(root, ec);
        return 11;
    }

    poseweaver::ObjMaterialTexture obj;
    if (!poseweaver::load_obj_material_texture(obj_path, obj)
        || !obj.has_material
        || !obj.has_image_texture()
        || obj.uv.rows() != 3
        || obj.uv_faces.rows() != 1)
    {
        std::cerr << "OBJ/MTL texture loading failed: " << obj.warning << '\n';
        std::filesystem::remove_all(root, ec);
        return 4;
    }

    poseweaver::ObjMaterialTexture color_obj;
    if (!poseweaver::load_obj_material_texture(color_obj_path, color_obj)
        || !color_obj.has_material
        || color_obj.has_image_texture()
        || color_obj.face_colors.rows() != 1
        || (color_obj.face_colors.row(0) - Eigen::RowVector3d(0.1, 0.2, 0.8)).norm() > 1e-9)
    {
        std::cerr << "OBJ/MTL material color loading failed: " << color_obj.warning << '\n';
        std::filesystem::remove_all(root, ec);
        return 5;
    }

    poseweaver::ObjMaterialTexture polygon_obj;
    if (!poseweaver::load_obj_material_texture(polygon_obj_path, polygon_obj)
        || !polygon_obj.has_material
        || polygon_obj.face_colors.rows() != 4
        || polygon_obj.uv_faces.rows() != 4
        || (polygon_obj.face_colors.row(0) - Eigen::RowVector3d(0.1, 0.2, 0.8)).norm() > 1e-9
        || (polygon_obj.face_colors.row(1) - Eigen::RowVector3d(0.1, 0.2, 0.8)).norm() > 1e-9
        || (polygon_obj.face_colors.row(2) - Eigen::RowVector3d(0.8, 0.2, 0.1)).norm() > 1e-9
        || (polygon_obj.face_colors.row(3) - Eigen::RowVector3d(0.8, 0.2, 0.1)).norm() > 1e-9)
    {
        std::cerr << "OBJ polygon material triangulation failed: " << polygon_obj.warning << '\n';
        std::filesystem::remove_all(root, ec);
        return 6;
    }

    poseweaver::GltfGarment glb;
    std::string error;
    if (!poseweaver::load_gltf_garment(glb_path, glb, &error)
        || !glb.has_material
        || !glb.has_image_texture()
        || glb.vertices.rows() != 3
        || glb.faces.rows() != 1
        || (glb.vertices.row(0) - Eigen::RowVector3d(2, 3, 4)).norm() > 1e-9)
    {
        std::cerr << "GLB texture loading failed: " << error << ' ' << glb.texture_warning << '\n';
        std::filesystem::remove_all(root, ec);
        return 7;
    }

    poseweaver::GltfGarment gltf;
    if (!poseweaver::load_gltf_garment(gltf_path, gltf, &error)
        || !gltf.has_material
        || !gltf.has_image_texture()
        || gltf.vertices.rows() != 3
        || gltf.faces.rows() != 1
        || (gltf.face_colors.row(0) - Eigen::RowVector3d(0.5, 0.25, 0.75)).norm() > 1e-9)
    {
        std::cerr << "glTF external resource loading failed: " << error << ' ' << gltf.texture_warning << '\n';
        std::filesystem::remove_all(root, ec);
        return 8;
    }

    poseweaver::GltfGarment color_gltf;
    if (!poseweaver::load_gltf_garment(color_gltf_path, color_gltf, &error)
        || !color_gltf.has_material
        || color_gltf.has_image_texture()
        || color_gltf.face_colors.rows() != 1
        || (color_gltf.face_colors.row(0) - Eigen::RowVector3d(0.2, 0.6, 0.3)).norm() > 1e-6)
    {
        std::cerr << "glTF material color loading failed: " << error << '\n';
        std::filesystem::remove_all(root, ec);
        return 9;
    }

    std::filesystem::remove_all(root, ec);
    std::cout << "Garment asset smoke test passed\n";
    return ec ? 10 : 0;
}
