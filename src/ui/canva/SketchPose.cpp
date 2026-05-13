#include "ui/canva/SketchPose.h"

#include "core/Paths.h"
#include "smpl/Smplh.h"

#include <cnpy.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

namespace poseweaver::ui_canva
{
    namespace
    {
        constexpr int kSketchJointCount = 22;
        constexpr int kSketchParents[kSketchJointCount] = {
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
        };

        struct RigJoint
        {
            int index = -1;
            int parent = -1;
            Eigen::Vector3d position = Eigen::Vector3d::Zero();
            bool valid = false;
        };

        static void set_error(std::string* out, const std::string& msg)
        {
            if (out)
                *out = msg;
        }

        static bool read_text_file(const std::filesystem::path& path, std::string* out, std::string* out_error)
        {
            std::ifstream in(path);
            if (!in)
            {
                set_error(out_error, "Failed to open " + path.string());
                return false;
            }
            std::ostringstream ss;
            ss << in.rdbuf();
            *out = ss.str();
            return true;
        }

        static std::string json_escape(const std::string& s)
        {
            std::string out;
            out.reserve(s.size() + 8);
            for (char c : s)
            {
                switch (c)
                {
                case '\\': out += "\\\\"; break;
                case '"': out += "\\\""; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += c; break;
                }
            }
            return out;
        }

        static bool parse_json_number_at(const std::string& text, size_t* pos, double* out_value)
        {
            if (!pos || !out_value)
                return false;
            size_t i = *pos;
            while (i < text.size() && std::isspace((unsigned char)text[i]))
                ++i;

            const char* start = text.c_str() + i;
            char* end = nullptr;
            errno = 0;
            const double value = std::strtod(start, &end);
            if (end == start || errno == ERANGE)
                return false;

            *out_value = value;
            *pos = (size_t)(end - text.c_str());
            return true;
        }

        static bool parse_json_number_field(const std::string& text, const char* key, double* out_value)
        {
            const std::string needle = std::string("\"") + key + "\"";
            size_t pos = text.find(needle);
            if (pos == std::string::npos)
                return false;
            pos = text.find(':', pos + needle.size());
            if (pos == std::string::npos)
                return false;
            ++pos;
            return parse_json_number_at(text, &pos, out_value);
        }

        static std::string parse_json_string_field(const std::string& text, const char* key)
        {
            const std::string needle = std::string("\"") + key + "\"";
            size_t pos = text.find(needle);
            if (pos == std::string::npos)
                return {};
            pos = text.find(':', pos + needle.size());
            if (pos == std::string::npos)
                return {};
            ++pos;
            while (pos < text.size() && std::isspace((unsigned char)text[pos]))
                ++pos;
            if (pos >= text.size() || text[pos] != '"')
                return {};
            ++pos;

            std::string out;
            bool escaped = false;
            for (size_t i = pos; i < text.size(); ++i)
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

        static bool parse_json_number_array_field(
            const std::string& text,
            const char* key,
            std::vector<double>* out_values,
            std::string* out_error)
        {
            if (!out_values)
                return false;
            out_values->clear();

            const std::string needle = std::string("\"") + key + "\"";
            size_t pos = text.find(needle);
            if (pos == std::string::npos)
            {
                set_error(out_error, std::string("Missing JSON field: ") + key);
                return false;
            }
            pos = text.find('[', pos + needle.size());
            if (pos == std::string::npos)
            {
                set_error(out_error, std::string("JSON field is not an array: ") + key);
                return false;
            }
            ++pos;

            while (pos < text.size())
            {
                while (pos < text.size() && (std::isspace((unsigned char)text[pos]) || text[pos] == ','))
                    ++pos;
                if (pos >= text.size())
                    break;
                if (text[pos] == ']')
                    return true;

                double value = 0.0;
                if (!parse_json_number_at(text, &pos, &value))
                {
                    set_error(out_error, std::string("Failed to parse numeric array field: ") + key);
                    return false;
                }
                out_values->push_back(value);
            }

            set_error(out_error, std::string("Unterminated numeric array field: ") + key);
            return false;
        }

        static bool parse_rig_joints(
            const std::string& text,
            std::vector<RigJoint>* out_joints,
            std::string* out_error)
        {
            if (!out_joints)
                return false;
            out_joints->assign(kSketchJointCount, RigJoint{});

            const size_t joints_key = text.find("\"joints\"");
            if (joints_key == std::string::npos)
            {
                set_error(out_error, "Missing joints array in Sketch2Rig JSON");
                return false;
            }
            size_t pos = text.find('[', joints_key);
            if (pos == std::string::npos)
            {
                set_error(out_error, "Malformed joints array in Sketch2Rig JSON");
                return false;
            }
            ++pos;

            int parsed = 0;
            while (pos < text.size())
            {
                const size_t object_begin = text.find('{', pos);
                const size_t array_end = text.find(']', pos);
                if (array_end != std::string::npos && (object_begin == std::string::npos || array_end < object_begin))
                    break;
                if (object_begin == std::string::npos)
                    break;

                int depth = 0;
                size_t object_end = std::string::npos;
                for (size_t i = object_begin; i < text.size(); ++i)
                {
                    if (text[i] == '{')
                        ++depth;
                    else if (text[i] == '}')
                    {
                        --depth;
                        if (depth == 0)
                        {
                            object_end = i;
                            break;
                        }
                    }
                }
                if (object_end == std::string::npos)
                    break;

                const std::string object = text.substr(object_begin, object_end - object_begin + 1);
                double index_d = -1.0;
                double parent_d = -1.0;
                std::vector<double> pos_values;
                if (parse_json_number_field(object, "index", &index_d)
                    && parse_json_number_field(object, "parent", &parent_d)
                    && parse_json_number_array_field(object, "position", &pos_values, nullptr)
                    && pos_values.size() >= 3)
                {
                    const int index = (int)std::lround(index_d);
                    if (index >= 0 && index < kSketchJointCount)
                    {
                        RigJoint joint;
                        joint.index = index;
                        joint.parent = (int)std::lround(parent_d);
                        joint.position = Eigen::Vector3d(pos_values[0], pos_values[1], pos_values[2]);
                        joint.valid = true;
                        (*out_joints)[(size_t)index] = joint;
                        ++parsed;
                    }
                }

                pos = object_end + 1;
            }

            if (parsed < kSketchJointCount)
            {
                std::ostringstream oss;
                oss << "Expected " << kSketchJointCount << " rig joints, parsed " << parsed;
                set_error(out_error, oss.str());
                return false;
            }
            return true;
        }

        static std::vector<std::vector<int>> children_from_parents(const int* parents, int count)
        {
            std::vector<std::vector<int>> children((size_t)count);
            for (int j = 0; j < count; ++j)
            {
                const int p = parents[j];
                if (p >= 0 && p < count)
                    children[(size_t)p].push_back(j);
            }
            return children;
        }

        static Eigen::Vector3d normalize_or_zero(const Eigen::Vector3d& v)
        {
            const double n = v.norm();
            if (n < 1e-9)
                return Eigen::Vector3d::Zero();
            return v / n;
        }

        static Eigen::Matrix3d project_to_rotation(const Eigen::Matrix3d& input)
        {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(input, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            const Eigen::Matrix3d V = svd.matrixV();
            Eigen::Matrix3d R = U * V.transpose();
            if (R.determinant() < 0.0)
            {
                U.col(2) *= -1.0;
                R = U * V.transpose();
            }
            return R;
        }

        static Eigen::Matrix3d rotation_between(Eigen::Vector3d a, Eigen::Vector3d b)
        {
            a = normalize_or_zero(a);
            b = normalize_or_zero(b);
            if (a.squaredNorm() == 0.0 || b.squaredNorm() == 0.0)
                return Eigen::Matrix3d::Identity();

            const Eigen::Vector3d v = a.cross(b);
            const double c = std::clamp(a.dot(b), -1.0, 1.0);
            if (c > 1.0 - 1e-8)
                return Eigen::Matrix3d::Identity();
            if (c < -1.0 + 1e-8)
            {
                Eigen::Vector3d helper(1.0, 0.0, 0.0);
                if (std::abs(a.dot(helper)) > 0.9)
                    helper = Eigen::Vector3d(0.0, 1.0, 0.0);
                Eigen::Vector3d axis = normalize_or_zero(a.cross(helper));
                if (axis.squaredNorm() == 0.0)
                    axis = Eigen::Vector3d::UnitZ();
                return Eigen::AngleAxisd(std::acos(-1.0), axis).toRotationMatrix();
            }

            const double s = std::max(1e-12, v.norm());
            Eigen::Matrix3d K;
            K << 0.0, -v.z(), v.y(),
                v.z(), 0.0, -v.x(),
                -v.y(), v.x(), 0.0;
            return Eigen::Matrix3d::Identity() + K + K * K * ((1.0 - c) / (s * s));
        }

        static Eigen::Matrix3d kabsch_rotation(
            const std::vector<Eigen::Vector3d>& src_in,
            const std::vector<Eigen::Vector3d>& dst_in)
        {
            std::vector<Eigen::Vector3d> src;
            std::vector<Eigen::Vector3d> dst;
            const size_t n = std::min(src_in.size(), dst_in.size());
            src.reserve(n);
            dst.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                Eigen::Vector3d a = normalize_or_zero(src_in[i]);
                Eigen::Vector3d b = normalize_or_zero(dst_in[i]);
                if (a.squaredNorm() == 0.0 || b.squaredNorm() == 0.0)
                    continue;
                src.push_back(a);
                dst.push_back(b);
            }

            if (src.empty())
                return Eigen::Matrix3d::Identity();
            if (src.size() == 1)
                return rotation_between(src[0], dst[0]);

            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < src.size(); ++i)
                H.noalias() += src[i] * dst[i].transpose();

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d V = svd.matrixV();
            const Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d R = V * U.transpose();
            if (R.determinant() < 0.0)
            {
                V.col(2) *= -1.0;
                R = V * U.transpose();
            }
            return R;
        }

        static Eigen::Vector3d matrix_to_axis_angle(const Eigen::Matrix3d& R)
        {
            Eigen::AngleAxisd aa(project_to_rotation(R));
            if (std::abs(aa.angle()) < 1e-8)
                return Eigen::Vector3d::Zero();
            return aa.axis() * aa.angle();
        }

        static Eigen::MatrixXd rest_joints_blender_from_model(
            const SmplhModel& model,
            const std::vector<double>& betas_in)
        {
            const int V = model.vertex_count();
            const int B = model.beta_count();
            Eigen::VectorXd betas = Eigen::VectorXd::Zero(B);
            const int n = std::min<int>(B, (int)betas_in.size());
            for (int i = 0; i < n; ++i)
                betas[i] = betas_in[(size_t)i];

            Eigen::Map<const Eigen::VectorXd> v_template_flat(model.v_template.data(), (size_t)V * 3);
            Eigen::VectorXd v_shaped_flat = v_template_flat;
            if (B > 0)
                v_shaped_flat.noalias() += model.shapedirs * betas;

            Eigen::Map<const MatrixX3dRM> v_shaped(v_shaped_flat.data(), V, 3);
            Eigen::MatrixXd joints = model.J_regressor * v_shaped;
            joints.col(1) *= -1.0;
            return joints;
        }

        static std::vector<double> estimate_pose_from_bone_angles(
            const Eigen::MatrixXd& target_joints_blender,
            const Eigen::MatrixXd& rest_joints_blender,
            int full_joint_count)
        {
            const int keep = std::min<int>(
                kSketchJointCount,
                std::min<int>((int)target_joints_blender.rows(), (int)rest_joints_blender.rows()));
            std::vector<Eigen::Matrix3d> global_rot((size_t)full_joint_count, Eigen::Matrix3d::Identity());
            std::vector<Eigen::Matrix3d> local_rot((size_t)full_joint_count, Eigen::Matrix3d::Identity());
            const auto children = children_from_parents(kSketchParents, keep);

            for (int j = 0; j < keep; ++j)
            {
                const int p = kSketchParents[j];
                const Eigen::Matrix3d parent_global =
                    (p >= 0 && p < keep) ? global_rot[(size_t)p] : Eigen::Matrix3d::Identity();

                std::vector<Eigen::Vector3d> src;
                std::vector<Eigen::Vector3d> dst_parent_local;
                for (int child : children[(size_t)j])
                {
                    const Eigen::Vector3d rest_dir =
                        normalize_or_zero(rest_joints_blender.row(child).transpose() - rest_joints_blender.row(j).transpose());
                    const Eigen::Vector3d target_dir =
                        normalize_or_zero(target_joints_blender.row(child).transpose() - target_joints_blender.row(j).transpose());
                    if (rest_dir.squaredNorm() == 0.0 || target_dir.squaredNorm() == 0.0)
                        continue;
                    src.push_back(rest_dir);
                    dst_parent_local.push_back(parent_global.transpose() * target_dir);
                }

                Eigen::Matrix3d local = Eigen::Matrix3d::Identity();
                if (src.size() == 1)
                    local = rotation_between(src[0], dst_parent_local[0]);
                else if (src.size() > 1)
                    local = kabsch_rotation(src, dst_parent_local);

                local = project_to_rotation(local);
                local_rot[(size_t)j] = local;
                global_rot[(size_t)j] = project_to_rotation(parent_global * local);
            }

            const Eigen::Matrix3d C = (Eigen::Vector3d(1.0, -1.0, 1.0)).asDiagonal();
            std::vector<double> pose((size_t)std::max(0, full_joint_count) * 3u, 0.0);
            for (int j = 0; j < keep; ++j)
            {
                const Eigen::Vector3d aa = matrix_to_axis_angle(C * local_rot[(size_t)j] * C);
                pose[(size_t)j * 3u + 0u] = aa.x();
                pose[(size_t)j * 3u + 1u] = aa.y();
                pose[(size_t)j * 3u + 2u] = aa.z();
            }
            return pose;
        }

        static Eigen::MatrixXd rig_joints_to_camera_blender(
            const std::vector<RigJoint>& joints,
            int image_size)
        {
            const int size = std::max(2, image_size);
            constexpr double kAngleX = 1.0471975803375244;
            constexpr double kAngleY = 0.7348438501358032;
            const double cx = 0.5 * (double)(size - 1);
            const double cy = 0.5 * (double)(size - 1);
            const double fx = 0.5 * (double)(size - 1) / std::tan(0.5 * kAngleX);
            const double fy = 0.5 * (double)(size - 1) / std::tan(0.5 * kAngleY);

            Eigen::MatrixXd out(kSketchJointCount, 3);
            out.setZero();
            for (int i = 0; i < kSketchJointCount; ++i)
            {
                const Eigen::Vector3d p = joints[(size_t)i].position;
                const double u = (p.x() >= -0.25 && p.x() <= 1.25) ? p.x() * (double)(size - 1) : p.x();
                const double v = (p.y() >= -0.25 && p.y() <= 1.25) ? p.y() * (double)(size - 1) : p.y();
                const double z = p.z();
                out(i, 0) = (u - cx) / fx * z;
                out(i, 1) = (v - cy) / fy * z;
                out(i, 2) = -z;
            }
            return out;
        }

        static int gender_index_from_string(const std::string& gender)
        {
            if (gender == "male")
                return 0;
            if (gender == "female")
                return 1;
            return 2;
        }

        static std::string gender_model_dir(const std::string& gender)
        {
            if (gender == "female")
                return "female";
            if (gender == "neutral")
                return "neutral";
            return "male";
        }

        static bool save_json(
            const std::filesystem::path& path,
            const std::filesystem::path& rig_json_path,
            const std::filesystem::path& betas_json_path,
            const std::filesystem::path& npz_path,
            const std::string& gender,
            const std::vector<double>& betas,
            const std::vector<double>& pose,
            const Eigen::MatrixXd& target_joints,
            int image_size,
            std::string* out_error)
        {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec)
            {
                set_error(out_error, "Failed to create pose output directory: " + ec.message());
                return false;
            }

            std::ofstream out(path);
            if (!out)
            {
                set_error(out_error, "Failed to write " + path.string());
                return false;
            }

            out << std::setprecision(17);
            out << "{\n"
                << "  \"schema\": \"poseweaver.sketch_pose.v1\",\n"
                << "  \"source_rig\": \"" << json_escape(rig_json_path.string()) << "\",\n"
                << "  \"source_betas\": \"" << json_escape(betas_json_path.string()) << "\",\n"
                << "  \"npz\": \"" << json_escape(npz_path.string()) << "\",\n"
                << "  \"gender\": \"" << json_escape(gender) << "\",\n"
                << "  \"gender_index\": " << gender_index_from_string(gender) << ",\n"
                << "  \"image_size\": " << image_size << ",\n"
                << "  \"parents\": [";
            for (int i = 0; i < kSketchJointCount; ++i)
                out << (i ? ", " : "") << kSketchParents[i];
            out << "],\n  \"betas\": [";
            for (size_t i = 0; i < betas.size(); ++i)
                out << (i ? ", " : "") << betas[i];
            out << "],\n  \"pose_axis_angle\": [";
            for (size_t i = 0; i < pose.size(); ++i)
                out << (i ? ", " : "") << pose[i];
            out << "],\n  \"target_joints_camera_blender\": [\n";
            for (int r = 0; r < target_joints.rows(); ++r)
            {
                out << "    [" << target_joints(r, 0) << ", " << target_joints(r, 1) << ", " << target_joints(r, 2) << "]";
                out << (r + 1 < target_joints.rows() ? ",\n" : "\n");
            }
            out << "  ]\n}\n";
            return true;
        }

        static bool save_npz(
            const std::filesystem::path& path,
            const std::string& gender,
            const std::vector<double>& betas,
            const std::vector<double>& pose,
            std::string* out_error)
        {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec)
            {
                set_error(out_error, "Failed to create pose output directory: " + ec.message());
                return false;
            }
            std::filesystem::remove(path, ec);

            try
            {
                cnpy::npz_save(path.string(), "poses", pose.data(), { 1u, pose.size() }, "w");
                cnpy::npz_save(path.string(), "betas", betas.data(), { betas.size() }, "a");

                const int gender_index = gender_index_from_string(gender);
                cnpy::npz_save(path.string(), "gender_index", &gender_index, { 1u }, "a");

                const int no_root_fix = 1;
                cnpy::npz_save(path.string(), "poseweaver_no_root_fix", &no_root_fix, { 1u }, "a");

                std::vector<unsigned char> gender_bytes(gender.begin(), gender.end());
                if (gender_bytes.empty())
                    gender_bytes.push_back((unsigned char)'m');
                cnpy::npz_save(path.string(), "gender", gender_bytes.data(), { gender_bytes.size() }, "a");

                std::vector<int> parents(kSketchParents, kSketchParents + kSketchJointCount);
                cnpy::npz_save(path.string(), "parents", parents.data(), { parents.size() }, "a");
            }
            catch (const std::exception& e)
            {
                set_error(out_error, std::string("Failed to write sketch pose NPZ: ") + e.what());
                return false;
            }

            return true;
        }
    }

    bool save_sketch_pose_from_outputs(
        const std::filesystem::path& rig_json_path,
        const std::filesystem::path& betas_json_path,
        const std::filesystem::path& out_json_path,
        const std::filesystem::path& out_npz_path,
        SketchPoseBuildResult* out_result,
        std::string* out_error)
    {
        std::string rig_text;
        std::string betas_text;
        if (!read_text_file(rig_json_path, &rig_text, out_error)
            || !read_text_file(betas_json_path, &betas_text, out_error))
            return false;

        std::vector<RigJoint> rig_joints;
        if (!parse_rig_joints(rig_text, &rig_joints, out_error))
            return false;

        std::vector<double> betas;
        if (!parse_json_number_array_field(betas_text, "betas", &betas, out_error))
            return false;

        std::string gender = parse_json_string_field(betas_text, "gender");
        if (gender != "female" && gender != "male" && gender != "neutral")
            gender = "male";

        double image_size_d = 512.0;
        (void)parse_json_number_field(rig_text, "image_size", &image_size_d);
        const int image_size = std::max(2, (int)std::lround(image_size_d));

        SmplhModel model;
        const std::filesystem::path model_path = assets_dir() / "smplh" / gender_model_dir(gender) / "model.npz";
        std::string model_error;
        if (!model.load_from_npz(model_path, &model_error))
        {
            set_error(out_error, "Failed to load SMPLH model for sketch pose: " + model_error);
            return false;
        }

        const Eigen::MatrixXd target_joints = rig_joints_to_camera_blender(rig_joints, image_size);
        const Eigen::MatrixXd rest_joints = rest_joints_blender_from_model(model, betas);
        const std::vector<double> pose =
            estimate_pose_from_bone_angles(target_joints, rest_joints, model.joint_count());

        if (!save_json(out_json_path, rig_json_path, betas_json_path, out_npz_path, gender, betas, pose, target_joints, image_size, out_error)
            || !save_npz(out_npz_path, gender, betas, pose, out_error))
            return false;

        if (out_result)
        {
            out_result->json_path = out_json_path;
            out_result->npz_path = out_npz_path;
            out_result->gender = gender;
            out_result->betas = betas;
            out_result->pose_axis_angle = pose;
        }
        return true;
    }
}
