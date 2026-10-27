#include "core/Assets.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <system_error>

namespace poseweaver
{
    AssetIndex::AssetIndex(std::filesystem::path assets_root) : m_assets_root(std::move(assets_root))
    {
        refresh();
    }

    namespace
    {
        std::string lower_extension(const std::filesystem::path& path)
        {
            std::string extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return extension;
        }

        std::string obj_geometry_signature(const std::filesystem::path& path)
        {
            std::ifstream input(path);
            std::ostringstream signature;
            signature << std::setprecision(17);
            std::string line;
            int vertex_count = 0;
            int face_count = 0;
            while (std::getline(input, line))
            {
                std::istringstream values(line);
                std::string directive;
                values >> directive;
                if (directive == "v")
                {
                    double x = 0.0;
                    double y = 0.0;
                    double z = 0.0;
                    if (values >> x >> y >> z)
                    {
                        signature << 'v' << x << ',' << y << ',' << z << ';';
                        ++vertex_count;
                    }
                }
                else if (directive == "f")
                {
                    std::string corner;
                    signature << 'f';
                    while (values >> corner)
                        signature << corner.substr(0, corner.find('/')) << ',';
                    signature << ';';
                    ++face_count;
                }
            }
            if (vertex_count == 0 || face_count == 0)
                return {};
            return signature.str();
        }

        std::string texture_name(const std::string& canonical, const std::string& variant)
        {
            std::string base = std::filesystem::path(canonical).stem().string();
            std::string name = std::filesystem::path(variant).stem().string();
            if (name == base)
                return "Default";
            while (!base.empty() && (base.back() == '_' || base.back() == '-' || base.back() == '.'))
                base.pop_back();
            if (name.compare(0, base.size(), base) == 0)
                name.erase(0, base.size());
            while (!name.empty() && (name.front() == '_' || name.front() == '-' || name.front() == '.'))
                name.erase(name.begin());
            for (char& c : name)
            {
                if (c == '_' || c == '-')
                    c = ' ';
            }
            bool capitalize = true;
            for (char& c : name)
            {
                if (capitalize && std::isalpha(static_cast<unsigned char>(c)))
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                capitalize = std::isspace(static_cast<unsigned char>(c)) != 0;
            }
            return name.empty() ? std::filesystem::path(variant).stem().string() : name;
        }
    }

    void AssetIndex::index_garments(const std::filesystem::path& dir)
    {
        m_garments.clear();
        m_garment_textures.clear();
        std::map<std::string, std::vector<std::string>> groups;

        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
            return;

        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec))
        {
            if (ec)
                break;
            if (!entry.is_regular_file(ec))
                continue;

            const auto& p = entry.path();
            const std::string name = std::filesystem::relative(p, dir, ec).generic_string();
            if (ec)
                continue;
            if (name.find(":Zone.Identifier") != std::string::npos)
                continue;

            const std::string ext = lower_extension(p);
            if (ext != ".obj" && ext != ".glb" && ext != ".gltf")
                continue;

            std::string key;
            if (ext == ".obj")
                key = obj_geometry_signature(p);
            if (key.empty())
                key = "file:" + name;
            groups[key].push_back(name);
        }

        for (auto& item : groups)
        {
            auto& files = item.second;
            std::sort(files.begin(), files.end());
            const auto canonical = std::min_element(files.begin(), files.end(), [](const std::string& a, const std::string& b)
            {
                const std::size_t a_length = std::filesystem::path(a).stem().string().size();
                const std::size_t b_length = std::filesystem::path(b).stem().string().size();
                return a_length != b_length ? a_length < b_length : a < b;
            });
            std::rotate(files.begin(), canonical, canonical + 1);
            m_garments.push_back(files.front());
            std::vector<GarmentTextureVariant> textures;
            textures.reserve(files.size());
            for (const std::string& file : files)
                textures.push_back({ texture_name(files.front(), file), file });
            m_garment_textures.push_back(std::move(textures));
        }

        std::vector<std::size_t> order(m_garments.size());
        for (std::size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b)
            { return m_garments[a] < m_garments[b]; });
        std::vector<std::string> sorted_garments;
        std::vector<std::vector<GarmentTextureVariant>> sorted_textures;
        sorted_garments.reserve(order.size());
        sorted_textures.reserve(order.size());
        for (std::size_t index : order)
        {
            sorted_garments.push_back(std::move(m_garments[index]));
            sorted_textures.push_back(std::move(m_garment_textures[index]));
        }
        m_garments = std::move(sorted_garments);
        m_garment_textures = std::move(sorted_textures);
    }

    const std::vector<GarmentTextureVariant>& AssetIndex::garment_textures(const std::string& garment) const
    {
        static const std::vector<GarmentTextureVariant> empty;
        const auto found = std::lower_bound(m_garments.begin(), m_garments.end(), garment);
        if (found == m_garments.end() || *found != garment)
            return empty;
        return m_garment_textures[static_cast<std::size_t>(found - m_garments.begin())];
    }

    std::vector<std::string> AssetIndex::list_pose_files(const std::filesystem::path& dir)
    {
        std::vector<std::string> out;

        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
            return out;

        for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        {
            if (ec)
                break;
            if (!entry.is_regular_file(ec))
                continue;

            const auto& p = entry.path();
            const std::string name = p.filename().string();
            if (name.find(":Zone.Identifier") != std::string::npos)
                continue;

            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            if (ext == ".npz")
                out.push_back(name);
        }

        std::sort(out.begin(), out.end());
        return out;
    }

    std::vector<std::string> AssetIndex::list_mesh_files(const std::filesystem::path& dir)
    {
        std::vector<std::string> out;

        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
            return out;

        for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        {
            if (ec)
                break;

            if (!entry.is_regular_file(ec))
                continue;

            const auto& p = entry.path();
            const std::string name = p.filename().string();
            if (name.find(":Zone.Identifier") != std::string::npos)
                continue;

            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });

            if (ext == ".obj")
                out.push_back(name);
        }

        std::sort(out.begin(), out.end());
        return out;
    }

    void AssetIndex::refresh()
    {
        m_poses = list_pose_files(m_assets_root / "poses");
        index_garments(m_assets_root / "garments");
        m_meshes = list_mesh_files(m_assets_root / "mesh");
    }
}
