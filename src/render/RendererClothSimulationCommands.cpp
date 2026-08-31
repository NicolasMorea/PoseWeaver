#include "render/Renderer.h"

#include "core/Geometry.h"
#include "simulation/ClothMeshProcessor.h"
#include "simulation/SmplCollisionProxy.h"
#include "simulation/SmplSimplifier.h"
#include "simulation/collision/TriangleBvh.h"
#include "simulation/xpbd/ClothXpbd.h"
#include "simulation/gpu/ClothCudaJacobi.h"
#include "smpl/Smplh.h"

#include <igl/signed_distance.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace poseweaver
{
    namespace
    {
        template <typename T>
        static bool future_ready(std::future<T>& f)
        {
            return f.valid() && (f.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
        }

        static inline int clamp_frame(int i, int count)
        {
            if (count <= 0)
                return 0;
            return std::clamp(i, 0, count - 1);
        }

        struct SmplSceneTransform
        {
            Eigen::RowVector3d scale = Eigen::RowVector3d::Ones();
            Eigen::RowVector3d translation = Eigen::RowVector3d::Zero();
            bool valid = false;
        };

        static SmplSceneTransform lerp_scene_transform(
            const SmplSceneTransform& a,
            const SmplSceneTransform& b,
            double t)
        {
            const double u = std::clamp(t, 0.0, 1.0);
            SmplSceneTransform out;
            out.scale = (1.0 - u) * a.scale + u * b.scale;
            out.translation = (1.0 - u) * a.translation + u * b.translation;
            out.valid = a.valid || b.valid;
            return out;
        }

        static void apply_scene_transform(Eigen::MatrixXd& vertices, const SmplSceneTransform& transform)
        {
            if (!transform.valid || vertices.cols() != 3)
                return;
            vertices.array().rowwise() *= transform.scale.array();
            vertices.rowwise() += transform.translation;
        }

        static bool parse_json_number_at_cloth(const std::string& text, size_t* pos, double* out_value)
        {
            if (!pos || !out_value)
                return false;
            while (*pos < text.size() && std::isspace((unsigned char)text[*pos]))
                ++(*pos);
            const char* begin = text.c_str() + *pos;
            char* end = nullptr;
            const double value = std::strtod(begin, &end);
            if (end == begin)
                return false;
            *out_value = value;
            *pos = (size_t)(end - text.c_str());
            return true;
        }

        static bool parse_json_number_field_cloth(const std::string& text, const char* key, double* out_value)
        {
            const std::string needle = std::string("\"") + key + "\":";
            const size_t start = text.find(needle);
            if (start == std::string::npos)
                return false;
            size_t pos = start + needle.size();
            return parse_json_number_at_cloth(text, &pos, out_value);
        }

        static bool parse_json_number_array_field_cloth(
            const std::string& text,
            const char* key,
            std::vector<double>* out_values)
        {
            if (!out_values)
                return false;
            out_values->clear();

            const std::string needle = std::string("\"") + key + "\":";
            const size_t start = text.find(needle);
            if (start == std::string::npos)
                return false;

            size_t pos = text.find('[', start + needle.size());
            if (pos == std::string::npos)
                return false;
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
                if (!parse_json_number_at_cloth(text, &pos, &value))
                    return false;
                out_values->push_back(value);
            }
            return false;
        }

        static bool read_text_file_cloth(const std::filesystem::path& path, std::string* out)
        {
            std::ifstream in(path);
            if (!in)
                return false;
            std::ostringstream ss;
            ss << in.rdbuf();
            *out = ss.str();
            return true;
        }

        static bool parse_sketch2rig_joints_cloth(
            const std::string& text,
            Eigen::MatrixXd* out_joints,
            std::vector<int>* out_parents,
            int* out_image_size)
        {
            if (!out_joints || !out_parents)
                return false;

            constexpr int kSketchJointCount = 22;
            double image_size_d = 512.0;
            (void)parse_json_number_field_cloth(text, "image_size", &image_size_d);
            if (out_image_size)
                *out_image_size = std::clamp((int)std::lround(image_size_d), 2, 8192);

            const size_t joints_key = text.find("\"joints\"");
            if (joints_key == std::string::npos)
                return false;
            size_t pos = text.find('[', joints_key);
            if (pos == std::string::npos)
                return false;
            ++pos;

            Eigen::MatrixXd joints = Eigen::MatrixXd::Zero(kSketchJointCount, 3);
            std::vector<int> parents((size_t)kSketchJointCount, -1);
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
                std::vector<double> values;
                if (parse_json_number_field_cloth(object, "index", &index_d)
                    && parse_json_number_field_cloth(object, "parent", &parent_d)
                    && parse_json_number_array_field_cloth(object, "position", &values)
                    && values.size() >= 3)
                {
                    const int index = (int)std::lround(index_d);
                    if (index >= 0 && index < kSketchJointCount)
                    {
                        joints.row(index) = Eigen::RowVector3d(values[0], values[1], values[2]);
                        parents[(size_t)index] = (int)std::lround(parent_d);
                        ++parsed;
                    }
                }
                pos = object_end + 1;
            }

            if (parsed < kSketchJointCount)
                return false;
            *out_joints = std::move(joints);
            *out_parents = std::move(parents);
            return true;
        }

        static Eigen::MatrixXd sketch2rig_joints_to_camera_scene_axes_cloth(
            const Eigen::MatrixXd& joints,
            int image_size)
        {
            const int size = std::max(2, image_size);
            constexpr double kAngleY = 0.7348438501358032;
            constexpr double kAngleX = kAngleY;
            const double cx = 0.5 * (double)(size - 1);
            const double cy = 0.5 * (double)(size - 1);
            const double fx = 0.5 * (double)(size - 1) / std::tan(0.5 * kAngleX);
            const double fy = 0.5 * (double)(size - 1) / std::tan(0.5 * kAngleY);

            Eigen::MatrixXd out(joints.rows(), 3);
            out.setZero();
            for (int i = 0; i < joints.rows(); ++i)
            {
                const double px = joints(i, 0);
                const double py = joints(i, 1);
                const double z = joints(i, 2);
                const double u = (px >= -0.25 && px <= 1.25) ? px * (double)(size - 1) : px;
                const double v = (py >= -0.25 && py <= 1.25) ? py * (double)(size - 1) : py;
                out(i, 0) = (u - cx) / fx * z;
                out(i, 1) = -(v - cy) / fy * z;
                out(i, 2) = -z;
            }
            return out;
        }

        static bool rig_height_if_valid_cloth(const Eigen::MatrixXd& joints, double* out_height)
        {
            if (!out_height || joints.rows() <= 0)
                return false;
            const double h = joints.col(1).maxCoeff() - joints.col(1).minCoeff();
            if (!(h > 1e-6) || !std::isfinite(h))
                return false;
            *out_height = h;
            return true;
        }

        static double smpl_native_height_from_betas_cloth(
            const SmplhModel& model,
            const Eigen::VectorXd& betas)
        {
            const int V = model.vertex_count();
            const int B = model.beta_count();
            if (V <= 0 || model.J_regressor.cols() != V)
                return 1.6;

            Eigen::VectorXd b = Eigen::VectorXd::Zero(std::max(0, B));
            const int n = std::min<int>((int)betas.size(), B);
            for (int i = 0; i < n; ++i)
                b[i] = betas[i];

            Eigen::Map<const Eigen::VectorXd> v_template_flat(model.v_template.data(), (size_t)V * 3);
            Eigen::VectorXd v_shaped_flat = v_template_flat;
            if (B > 0)
                v_shaped_flat.noalias() += model.shapedirs * b;

            Eigen::Map<const MatrixX3dRM> v_shaped(v_shaped_flat.data(), V, 3);
            const Eigen::MatrixXd joints = model.J_regressor * v_shaped;
            double h = 0.0;
            return rig_height_if_valid_cloth(joints, &h) ? h : 1.6;
        }

        static bool load_sketch2rig_scene_joints_cloth(
            const std::filesystem::path& rig_path,
            double target_height,
            Eigen::MatrixXd* out_scene_joints,
            std::vector<int>* out_parents)
        {
            std::string text;
            if (rig_path.empty() || !read_text_file_cloth(rig_path, &text))
                return false;

            Eigen::MatrixXd joints;
            std::vector<int> parents;
            int image_size = 512;
            if (!parse_sketch2rig_joints_cloth(text, &joints, &parents, &image_size))
                return false;

            Eigen::MatrixXd camera_joints = sketch2rig_joints_to_camera_scene_axes_cloth(joints, image_size);
            if (camera_joints.rows() <= 0)
                return false;

            const Eigen::Vector3d root_offset = camera_joints.row(0).transpose();
            camera_joints.rowwise() -= root_offset.transpose();

            double predicted_height = 0.0;
            if (rig_height_if_valid_cloth(camera_joints, &predicted_height)
                && predicted_height > 1e-6
                && std::isfinite(predicted_height)
                && target_height > 1e-6
                && std::isfinite(target_height))
            {
                const double scene_scale = std::clamp(target_height / predicted_height, 0.05, 20.0);
                camera_joints *= scene_scale;
            }

            *out_scene_joints = std::move(camera_joints);
            *out_parents = std::move(parents);
            return true;
        }

        static bool fit_smpl_vertices_to_scene_joints_cloth(
            const SmplhModel& model,
            const Eigen::MatrixXd& vertices,
            const Eigen::MatrixXd& scene_joints,
            const std::vector<int>& parents,
            SmplSceneTransform* out_transform)
        {
            if (!out_transform
                || vertices.cols() != 3
                || scene_joints.cols() != 3
                || model.J_regressor.cols() != vertices.rows())
                return false;

            const Eigen::MatrixXd smpl_joints = model.J_regressor * vertices;
            const int joint_count = std::min<int>(
                22,
                std::min<int>(smpl_joints.rows(), scene_joints.rows()));
            if (joint_count < 2)
                return false;

            Eigen::VectorXd weights = Eigen::VectorXd::Zero(joint_count);
            for (int child = 0; child < joint_count && child < (int)parents.size(); ++child)
            {
                const int parent = parents[(size_t)child];
                if (parent >= 0 && parent < joint_count)
                {
                    weights[child] += 1.0;
                    weights[parent] += 1.0;
                }
            }

            const double weight_sum = weights.sum();
            if (!(weight_sum > 0.0) || !std::isfinite(weight_sum))
                return false;

            Eigen::Vector3d source_mean = Eigen::Vector3d::Zero();
            Eigen::Vector3d target_mean = Eigen::Vector3d::Zero();
            for (int i = 0; i < joint_count; ++i)
            {
                const Eigen::Vector3d source = smpl_joints.row(i).transpose();
                const Eigen::Vector3d target = scene_joints.row(i).transpose();
                if (!source.allFinite() || !target.allFinite())
                    return false;
                source_mean.noalias() += weights[i] * source;
                target_mean.noalias() += weights[i] * target;
            }
            source_mean /= weight_sum;
            target_mean /= weight_sum;

            double xy_numerator = 0.0;
            double xy_denominator = 0.0;
            double z_numerator = 0.0;
            double z_denominator = 0.0;
            for (int i = 0; i < joint_count; ++i)
            {
                const Eigen::Vector3d source = smpl_joints.row(i).transpose() - source_mean;
                const Eigen::Vector3d target = scene_joints.row(i).transpose() - target_mean;
                const double weight = weights[i];
                xy_numerator += weight * (source.x() * target.x() + source.y() * target.y());
                xy_denominator += weight * (source.x() * source.x() + source.y() * source.y());
                z_numerator += weight * source.z() * target.z();
                z_denominator += weight * source.z() * source.z();
            }

            if (!(xy_denominator > 1e-12) || !std::isfinite(xy_denominator))
                return false;

            const double scale_xy = std::clamp(xy_numerator / xy_denominator, 0.05, 20.0);
            const double scale_z = (z_denominator > 1e-12 && std::isfinite(z_denominator))
                ? std::clamp(z_numerator / z_denominator, 0.05, 20.0)
                : scale_xy;
            if (!std::isfinite(scale_xy) || !std::isfinite(scale_z))
                return false;

            const Eigen::Vector3d scale(scale_xy, scale_xy, scale_z);
            const Eigen::Vector3d translation = target_mean - scale.cwiseProduct(source_mean);
            if (!translation.allFinite())
                return false;

            out_transform->scale = Eigen::RowVector3d(scale.x(), scale.y(), scale.z());
            out_transform->translation = translation.transpose();
            out_transform->valid = true;
            return true;
        }

        static Eigen::MatrixXd compute_cloth_smpl_targets(
            const std::vector<int>& cloth_indices,
            const Eigen::MatrixXd& fallback_cloth_V,
            const ClothToSmplMapping& mapping,
            const Eigen::MatrixXd& body_V,
            const Eigen::MatrixXi& body_F,
            const Eigen::MatrixXd& body_rest_V)
        {
            Eigen::MatrixXd targets((int)cloth_indices.size(), 3);
            targets.setZero();

            for (int ri = 0; ri < (int)cloth_indices.size(); ++ri)
            {
                const int vi = cloth_indices[(size_t)ri];
                if (vi < 0 || vi >= fallback_cloth_V.rows())
                    continue;

                targets.row(ri) = fallback_cloth_V.row(vi);
                if (!mapping.valid() || mapping.face_index.size() <= vi)
                    continue;

                const int fi = mapping.face_index[vi];
                if (fi < 0 || fi >= body_F.rows())
                    continue;

                const int ia = body_F(fi, 0);
                const int ib = body_F(fi, 1);
                const int ic = body_F(fi, 2);
                if (ia < 0 || ib < 0 || ic < 0 ||
                    ia >= body_V.rows() || ib >= body_V.rows() || ic >= body_V.rows())
                {
                    continue;
                }

                const Eigen::RowVector3d bc = mapping.barycentric.row(vi);
                const Eigen::RowVector3d p_def =
                    bc[0] * body_V.row(ia) + bc[1] * body_V.row(ib) + bc[2] * body_V.row(ic);

                Eigen::RowVector3d T_def, B_def, N_def;
                if (!core::triangle_frame(body_V.row(ia), body_V.row(ib), body_V.row(ic), T_def, B_def, N_def))
                {
                    if (!core::triangle_frame(
                            body_rest_V.row(ia),
                            body_rest_V.row(ib),
                            body_rest_V.row(ic),
                            T_def,
                            B_def,
                            N_def))
                    {
                        continue;
                    }
                }

                const Eigen::RowVector3d off_local = mapping.offset_local.row(vi);
                targets.row(ri) = p_def + off_local[0] * T_def + off_local[1] * B_def + off_local[2] * N_def;
            }

            return targets;
        }

        struct ConstraintSkinProjectionResult
        {
            int corrected_vertices = 0;
            int pushed_out_vertices = 0;
            int pulled_in_vertices = 0;
        };

        static bool project_constraints_to_skin_margin(
            Eigen::MatrixXd& cloth_V,
            ClothToSmplMapping& mapping,
            const std::vector<int>& cloth_indices,
            const Eigen::MatrixXd& body_V,
            const Eigen::MatrixXi& body_F,
            double surface_offset,
            ConstraintSkinProjectionResult* result,
            std::string* error)
        {
            if (result)
                *result = ConstraintSkinProjectionResult{};

            if (cloth_indices.empty())
                return true;
            if (cloth_V.cols() != 3 || body_V.cols() != 3)
            {
                if (error)
                    *error = "Expected 3D vertex positions (#V x 3).";
                return false;
            }
            if (body_F.cols() != 3)
            {
                if (error)
                    *error = "Expected body faces to be triangles (#F x 3).";
                return false;
            }
            if (!mapping.valid() || mapping.face_index.size() != cloth_V.rows())
            {
                if (error)
                    *error = "Cloth-to-SMPL mapping is not valid for constrained projection.";
                return false;
            }
            if (body_V.rows() == 0 || body_F.rows() == 0)
            {
                if (error)
                    *error = "Missing body mesh data.";
                return false;
            }
            if (!(surface_offset >= 0.0) || !std::isfinite(surface_offset))
            {
                if (error)
                    *error = "surface_offset must be a finite non-negative value.";
                return false;
            }

            Eigen::MatrixXd P((int)cloth_indices.size(), 3);
            std::vector<int> source_vertices;
            source_vertices.reserve(cloth_indices.size());
            for (int ri = 0; ri < (int)cloth_indices.size(); ++ri)
            {
                const int vi = cloth_indices[(size_t)ri];
                if (vi < 0 || vi >= cloth_V.rows())
                {
                    if (error)
                        *error = "Constrained cloth vertex index is out of range.";
                    return false;
                }
                P.row(ri) = cloth_V.row(vi);
                source_vertices.push_back(vi);
            }

            Eigen::VectorXd S;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;
            Eigen::MatrixXd N_unused;
            igl::signed_distance(P, body_V, body_F, igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, S, I, C, N_unused);

            if (S.size() != P.rows() || I.size() != P.rows() || C.rows() != P.rows() || C.cols() != 3)
            {
                if (error)
                    *error = "Signed-distance query returned unexpected sizes.";
                return false;
            }

            const Eigen::RowVector3d body_centroid = body_V.colwise().mean();
            constexpr double kEps = 1e-9;

            int corrected = 0;
            int pushed_out = 0;
            int pulled_in = 0;

            for (int ri = 0; ri < P.rows(); ++ri)
            {
                const int vi = source_vertices[(size_t)ri];
                const int fi = I[ri];
                if (fi < 0 || fi >= body_F.rows() || !std::isfinite(S[ri]))
                    continue;

                const int ia = body_F(fi, 0);
                const int ib = body_F(fi, 1);
                const int ic = body_F(fi, 2);
                if (ia < 0 || ib < 0 || ic < 0 ||
                    ia >= body_V.rows() || ib >= body_V.rows() || ic >= body_V.rows())
                {
                    continue;
                }

                Eigen::RowVector3d dir = (S[ri] >= 0.0) ? (P.row(ri) - C.row(ri)) : (C.row(ri) - P.row(ri));
                double dir_norm = dir.norm();
                if (!(dir_norm > 1e-12) || !std::isfinite(dir_norm))
                {
                    const Eigen::RowVector3d a = body_V.row(ia);
                    const Eigen::RowVector3d b = body_V.row(ib);
                    const Eigen::RowVector3d c = body_V.row(ic);
                    dir = (b - a).cross(c - a);
                    dir_norm = dir.norm();
                    if (!(dir_norm > 1e-12) || !std::isfinite(dir_norm))
                        dir = C.row(ri) - body_centroid;
                    else if (dir.dot(((a + b + c) / 3.0) - body_centroid) < 0.0)
                        dir = -dir;
                    dir_norm = dir.norm();
                }
                if (!(dir_norm > 1e-12) || !std::isfinite(dir_norm))
                    continue;
                dir /= dir_norm;

                const Eigen::RowVector3d corrected_p = C.row(ri) + surface_offset * dir;
                if (!corrected_p.allFinite())
                    continue;

                if (S[ri] < surface_offset - kEps)
                    pushed_out++;
                else if (S[ri] > surface_offset + kEps)
                    pulled_in++;

                const Eigen::RowVector3d a = body_V.row(ia);
                const Eigen::RowVector3d b = body_V.row(ib);
                const Eigen::RowVector3d c = body_V.row(ic);
                const Eigen::RowVector3d bc = core::barycentric_on_triangle(C.row(ri), a, b, c);
                if (!bc.allFinite())
                    continue;
                const Eigen::RowVector3d off = corrected_p - C.row(ri);

                Eigen::RowVector3d T, B, N;
                if (!core::triangle_frame(a, b, c, T, B, N))
                {
                    mapping.offset_local.row(vi) = off;
                }
                else
                {
                    mapping.offset_local(vi, 0) = off.dot(T);
                    mapping.offset_local(vi, 1) = off.dot(B);
                    mapping.offset_local(vi, 2) = off.dot(N);
                }

                cloth_V.row(vi) = corrected_p;
                mapping.face_index[vi] = fi;
                mapping.barycentric.row(vi) = bc;
                mapping.offset.row(vi) = off;
                corrected++;
            }

            if (corrected != P.rows())
            {
                if (error)
                {
                    *error = "Could only project " + std::to_string(corrected)
                        + " of " + std::to_string((int)P.rows())
                        + " constrained vertices to the skin margin.";
                }
                return false;
            }

            if (result)
            {
                result->corrected_vertices = corrected;
                result->pushed_out_vertices = pushed_out;
                result->pulled_in_vertices = pulled_in;
            }
            return true;
        }

        struct BodyProxyPrepStats
        {
            int frames_processed = 0;
            int self_collision_contacts = 0;
            double inflation_margin = 0.0;
            double self_clearance = 0.0;
        };

        static Eigen::MatrixXd body_proxy_vertex_normals(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F)
        {
            Eigen::MatrixXd N = Eigen::MatrixXd::Zero(V.rows(), 3);
            if (V.rows() == 0 || V.cols() != 3 || F.cols() != 3)
                return N;

            const Eigen::RowVector3d centroid = V.colwise().mean();
            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int i0 = F(fi, 0);
                const int i1 = F(fi, 1);
                const int i2 = F(fi, 2);
                if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows())
                    continue;

                const Eigen::RowVector3d a = V.row(i0);
                const Eigen::RowVector3d b = V.row(i1);
                const Eigen::RowVector3d c = V.row(i2);
                Eigen::RowVector3d n = (b - a).cross(c - a);
                const double nn = n.norm();
                if (!(nn > 1e-20) || !std::isfinite(nn))
                    continue;

                const Eigen::RowVector3d fc = (a + b + c) / 3.0;
                if (n.dot(fc - centroid) < 0.0)
                    n = -n;

                N.row(i0) += n;
                N.row(i1) += n;
                N.row(i2) += n;
            }

            for (int vi = 0; vi < N.rows(); ++vi)
            {
                Eigen::RowVector3d n = N.row(vi);
                double nn = n.norm();
                if (!(nn > 1e-20) || !std::isfinite(nn))
                {
                    n = V.row(vi) - centroid;
                    nn = n.norm();
                }
                if (!(nn > 1e-20) || !std::isfinite(nn))
                    n = Eigen::RowVector3d(0.0, 1.0, 0.0);
                else
                    n /= nn;
                N.row(vi) = n;
            }

            return N;
        }

        static void inflate_body_proxy(
            Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            double margin)
        {
            if (!(margin > 0.0) || V.rows() == 0 || V.cols() != 3)
                return;

            const Eigen::MatrixXd N = body_proxy_vertex_normals(V, F);
            if (N.rows() != V.rows())
                return;

            V += margin * N;
        }

        static bool body_proxy_face_is_local_to_vertex(
            int vi,
            int face,
            const Eigen::MatrixXi& F,
            const std::vector<std::vector<int>>& one_ring)
        {
            if (vi < 0 || face < 0 || face >= F.rows())
                return true;

            for (int k = 0; k < 3; ++k)
            {
                const int fv = F(face, k);
                if (fv == vi)
                    return true;
                if (vi >= 0 && vi < (int)one_ring.size())
                {
                    const auto& ring = one_ring[(size_t)vi];
                    if (std::find(ring.begin(), ring.end(), fv) != ring.end())
                        return true;
                }
            }

            return false;
        }

        static std::vector<std::vector<int>> build_body_proxy_one_ring(
            int vertex_count,
            const Eigen::MatrixXi& F)
        {
            std::vector<std::vector<int>> one_ring((size_t)std::max(0, vertex_count));
            const auto add_edge = [&](int a, int b)
            {
                if (a < 0 || b < 0 || a >= vertex_count || b >= vertex_count || a == b)
                    return;
                auto& ring = one_ring[(size_t)a];
                if (std::find(ring.begin(), ring.end(), b) == ring.end())
                    ring.push_back(b);
            };

            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int i0 = F(fi, 0);
                const int i1 = F(fi, 1);
                const int i2 = F(fi, 2);
                add_edge(i0, i1);
                add_edge(i1, i0);
                add_edge(i1, i2);
                add_edge(i2, i1);
                add_edge(i2, i0);
                add_edge(i0, i2);
            }

            return one_ring;
        }

        static int relax_body_proxy_self_collisions(
            Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            double clearance,
            int passes)
        {
            if (!(clearance > 0.0) || passes <= 0 || V.rows() == 0 || F.rows() == 0 || V.cols() != 3 || F.cols() != 3)
                return 0;

            const std::vector<std::vector<int>> one_ring = build_body_proxy_one_ring((int)V.rows(), F);
            const double search_radius = std::max(0.02, clearance * 5.0);
            const double max_dist2 = search_radius * search_radius;

            int contacts = 0;
            for (int pass = 0; pass < passes; ++pass)
            {
                collision::TriangleBvh bvh;
                bvh.build(V, F);
                if (!bvh.bounds().valid())
                    break;

                const Eigen::RowVector3d centroid = V.colwise().mean();
                int pass_contacts = 0;

                for (int vi = 0; vi < V.rows(); ++vi)
                {
                    const Eigen::Vector3d p = V.row(vi);
                    int face = -1;
                    Eigen::Vector3d q;
                    double dist2 = std::numeric_limits<double>::infinity();
                    if (!bvh.closest_triangle(p, V, F, face, q, dist2, max_dist2, vi))
                        continue;
                    if (body_proxy_face_is_local_to_vertex(vi, face, F, one_ring))
                        continue;

                    const int i0 = F(face, 0);
                    const int i1 = F(face, 1);
                    const int i2 = F(face, 2);
                    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows())
                        continue;

                    const Eigen::Vector3d a = V.row(i0);
                    const Eigen::Vector3d b = V.row(i1);
                    const Eigen::Vector3d c = V.row(i2);

                    Eigen::Vector3d bary;
                    q = core::closest_point_on_triangle(p, a, b, c, &bary);
                    Eigen::Vector3d n = (b - a).cross(c - a);
                    const double nn = n.norm();
                    if (!(nn > 1e-20) || !std::isfinite(nn) || !bary.allFinite())
                        continue;
                    n /= nn;

                    const Eigen::Vector3d face_centroid = (a + b + c) / 3.0;
                    if (n.dot(face_centroid - centroid.transpose()) < 0.0)
                        n = -n;

                    const double signed_distance = n.dot(p - q);
                    if (!(signed_distance < clearance) || !std::isfinite(signed_distance))
                        continue;

                    const double b0 = std::clamp(bary[0], 0.0, 1.0);
                    const double b1 = std::clamp(bary[1], 0.0, 1.0);
                    const double b2 = std::clamp(bary[2], 0.0, 1.0);
                    const double denom = 1.0 + b0 * b0 + b1 * b1 + b2 * b2;
                    if (!(denom > 1e-12))
                        continue;

                    const double depth = std::min(clearance - signed_distance, 0.5 * search_radius);
                    const Eigen::Vector3d corr = (depth / denom) * n;
                    if (!corr.allFinite())
                        continue;

                    V.row(vi) += corr.transpose();
                    V.row(i0) -= (b0 * corr).transpose();
                    V.row(i1) -= (b1 * corr).transpose();
                    V.row(i2) -= (b2 * corr).transpose();

                    pass_contacts++;
                }

                contacts += pass_contacts;
                if (pass_contacts == 0)
                    break;
            }

            return contacts;
        }

        static BodyProxyPrepStats prepare_xpbd_body_proxy_frame(
            Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            double inflation_margin,
            double self_clearance)
        {
            BodyProxyPrepStats stats;
            stats.frames_processed = 1;
            stats.inflation_margin = inflation_margin;
            stats.self_clearance = self_clearance;

            inflate_body_proxy(V, F, inflation_margin);
            stats.self_collision_contacts = relax_body_proxy_self_collisions(V, F, self_clearance, 4);
            return stats;
        }
    }

    void Renderer::process_cloth_clear_simulation(AppState& state)
    {
        m_cloth_sim_frames.clear();
        m_smpl_sim_frames.clear();
        m_cloth_sim_append_prefix_frames.clear();
        m_smpl_sim_append_prefix_frames.clear();
        m_collision_proxy_append_prefix_frames.clear();
        m_cloth_sim_append_result = false;
        m_collision_proxy_sim_frames.clear();
        m_collision_proxy_sim_F.resize(0, 3);
        m_cloth_sim_restart_pending = false;
        m_cloth_sim_play_on_finish = true;
        m_cloth_sim_restart_play_on_finish = true;
        m_cloth_sim_reset_frame0_valid = false;
        m_cloth_sim_preview_state.reset();
        m_cloth_sim_preview_enabled.reset();
        m_cloth_sim_preview_last_seq_applied = 0;
        m_smplh_sim_preview_mesh_active = false;
        m_smplh_mesh_name.clear();
        state.controls.frame_count = 0;
        state.controls.current_frame = 0;
        state.controls.playing = false;
        state.controls.sim_steps_done = 0;
        state.controls.sim_steps_total = 0;
        state.controls.sim_pipeline_stage = 0;
        state.controls.sim_sequence_steps_done = 0;
        state.controls.sim_sequence_steps_total = 0;
        state.controls.sim_proxy_steps_done = 0;
        state.controls.sim_proxy_steps_total = 0;
        state.controls.sim_solver_steps_done = 0;
        state.controls.sim_solver_steps_total = 0;
        state.controls.sim_sequence_seconds = 0.0;
        state.controls.sim_proxy_seconds = 0.0;
        state.controls.sim_solver_seconds = 0.0;
        state.controls.sim_pipeline_seconds = 0.0;
        state.controls.sim_sequence_timing_details.clear();
        state.controls.sim_proxy_timing_details.clear();
        state.controls.sim_solver_timing_details.clear();
        state.controls.sim_stage.clear();
        console_add_line(state, "Cleared cloth simulation frames");
    }

    void Renderer::process_cloth_cancel_simulation(AppState& state)
    {
        if (m_cloth_sim_job_running && m_cloth_sim_cancel_flag)
        {
            m_cloth_sim_cancel_flag->store(true);
            state.controls.sim_stage = "Stopping";
            console_add_line(state, "Cancel requested for cloth simulation");
        }
    }

    void Renderer::process_cloth_simulate(AppState& state)
    {
        state.controls.sim_stage = "Preparing";
        const bool continue_from_last_frame = state.cloth_simulate_continue_from_last_frame;
        const float duration_override_seconds = state.cloth_simulate_duration_override_seconds;
        const int max_frames_override = state.cloth_simulate_max_frames_override;
        const bool pose_transition_enabled = state.cloth_simulate_pose_transition_enabled;
        const std::string transition_from_pose_path = state.cloth_simulate_transition_from_pose_path;
        const int pose_transition_frames = state.cloth_simulate_pose_transition_frames;
        const bool sequence_override_enabled = state.cloth_simulate_sequence_enabled;
        const std::vector<std::string> sequence_pose_paths = state.cloth_simulate_sequence_pose_paths;
        const std::vector<std::string> sequence_rig_paths = state.cloth_simulate_sequence_rig_paths;
        const int sequence_frame_rate = std::clamp(state.cloth_simulate_sequence_frame_rate, 1, 240);
        const int sequence_transition_frames = std::clamp(state.cloth_simulate_sequence_transition_frames, 0, 240);
        const bool sequence_initial_smooth = state.cloth_simulate_sequence_initial_smooth;
        const bool play_on_finish = state.cloth_simulate_play_on_finish;
        state.cloth_simulate_play_on_finish = true;
        state.cloth_simulate_continue_from_last_frame = false;
        state.cloth_simulate_duration_override_seconds = -1.0f;
        state.cloth_simulate_max_frames_override = 0;
        state.cloth_simulate_pose_transition_enabled = false;
        state.cloth_simulate_transition_from_pose_path.clear();
        state.cloth_simulate_pose_transition_frames = 0;
        state.cloth_simulate_sequence_enabled = false;
        state.cloth_simulate_sequence_pose_paths.clear();
        state.cloth_simulate_sequence_rig_paths.clear();
        state.cloth_simulate_sequence_frame_rate = 12;
        state.cloth_simulate_sequence_transition_frames = 0;
        state.cloth_simulate_sequence_initial_smooth = false;

        if (m_cloth_sim_job_running)
        {
            m_cloth_sim_restart_pending = true;
            m_cloth_sim_restart_play_on_finish = play_on_finish;
            process_cloth_cancel_simulation(state);
            console_add_line(state, "Simulation restart requested; cancel requested");
            return;
        }

        state.controls.sim_pipeline_stage = 0;
        state.controls.sim_steps_done = 0;
        state.controls.sim_steps_total = 0;
        state.controls.sim_sequence_steps_done = 0;
        state.controls.sim_sequence_steps_total = 0;
        state.controls.sim_proxy_steps_done = 0;
        state.controls.sim_proxy_steps_total = 0;
        state.controls.sim_solver_steps_done = 0;
        state.controls.sim_solver_steps_total = 0;
        state.controls.sim_sequence_seconds = 0.0;
        state.controls.sim_proxy_seconds = 0.0;
        state.controls.sim_solver_seconds = 0.0;
        state.controls.sim_pipeline_seconds = 0.0;
        state.controls.sim_sequence_timing_details.clear();
        state.controls.sim_proxy_timing_details.clear();
        state.controls.sim_solver_timing_details.clear();

        m_cloth_sim_restart_pending = false;
        m_cloth_sim_append_result = continue_from_last_frame && !m_cloth_sim_frames.empty();
        m_cloth_sim_append_prefix_frames = m_cloth_sim_append_result
            ? m_cloth_sim_frames
            : std::vector<Eigen::MatrixXd>();
        m_smpl_sim_append_prefix_frames = m_cloth_sim_append_result
            ? m_smpl_sim_frames
            : std::vector<Eigen::MatrixXd>();
        m_collision_proxy_append_prefix_frames = m_cloth_sim_append_result
            ? m_collision_proxy_sim_frames
            : std::vector<Eigen::MatrixXd>();
        m_smpl_sim_frames.clear();

        if (m_garment_mesh_name.empty() || !state.show_garment_mesh)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but no garment is loaded/visible");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but garment mesh is empty");
            return;
        }

        ensure_smplh_mesh_loaded(state);
        if (!m_smplh_loaded || m_smplh_vertices.rows() == 0 || m_smplh_model.F.rows() == 0)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but SMPL body is not available");
            return;
        }

        ensure_garment_to_smpl_mapping(state);
        if (!m_garment_to_smpl.valid()
            || m_garment_to_smpl.face_index.size() != m_garment_rest_vertices.rows())
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but cloth-to-SMPL mapping is not available");
            return;
        }

        Eigen::MatrixXi cloth_F = garment.F;
        Eigen::MatrixXi body_F = m_smplh_model.F;
        if (m_smplh_loaded_pose_mirror_x && body_F.cols() >= 3)
            body_F.col(1).swap(body_F.col(2));

        state.xpbd_params.simulation_steps = std::clamp(state.xpbd_params.simulation_steps, 1, 1000000);
        state.xpbd_params.duration_seconds = std::max(state.xpbd_params.duration_seconds, 0.0f);
        state.xpbd_params.dt = solver_dt_seconds(state.xpbd_params);

        const int initial_simulation_steps = state.xpbd_params.simulation_steps;
        const double dt_for_steps = (double)state.xpbd_params.dt;
        int steps = initial_simulation_steps;
        if (duration_override_seconds > 0.0f)
        {
            const int duration_steps = std::max(
                1,
                (int)std::ceil((double)duration_override_seconds / dt_for_steps));
            steps = max_frames_override > 0
                ? std::min(duration_steps, std::max(0, max_frames_override - 1))
                : duration_steps;
        }
        const int max_frames_for_progress = std::max(1, steps + 1);
        const double duration_for_steps = (double)steps * dt_for_steps;

        const int n_joints = m_smplh_model.joint_count();
        const int dof = 3 * n_joints;
        std::vector<Eigen::VectorXd> sequence_pose_axis_angles;
        std::vector<std::string> sequence_pose_keys;
        bool sequence_override_ready = false;
        if (sequence_override_enabled && !sequence_pose_paths.empty() && n_joints > 0)
        {
            sequence_pose_axis_angles.reserve(sequence_pose_paths.size());
            sequence_pose_keys.reserve(sequence_pose_paths.size());
            for (const std::string& pose_path_string : sequence_pose_paths)
            {
                Eigen::VectorXd pose_axis_angle;
                std::string pose_err;
                if (!smplh_load_pose_npz(
                        std::filesystem::path(pose_path_string),
                        pose_axis_angle,
                        nullptr,
                        &pose_err,
                        nullptr)
                    || (int)pose_axis_angle.size() != dof)
                {
                    console_add_line(state, "Sequence override skipped: failed to load pose "
                        + pose_path_string
                        + (pose_err.empty() ? std::string() : (": " + pose_err)));
                    sequence_pose_axis_angles.clear();
                    sequence_pose_keys.clear();
                    break;
                }
                sequence_pose_axis_angles.push_back(std::move(pose_axis_angle));
                sequence_pose_keys.push_back(pose_path_string);
            }
            sequence_override_ready = sequence_pose_axis_angles.size() == sequence_pose_paths.size()
                && !sequence_pose_axis_angles.empty();
        }

        Eigen::VectorXd simulation_target_axis_angle = m_smplh_loaded_pose_axis_angle;
        if (sequence_override_ready)
            simulation_target_axis_angle = sequence_pose_axis_angles.front();

        SmplSceneTransform current_scene_transform;
        current_scene_transform.scale = Eigen::RowVector3d(
            (double)state.scene_view_smpl_fit_scale[0],
            (double)state.scene_view_smpl_fit_scale[1],
            (double)state.scene_view_smpl_fit_scale[2]);
        current_scene_transform.translation = Eigen::RowVector3d(
            (double)state.scene_view_smpl_fit_translation[0],
            (double)state.scene_view_smpl_fit_translation[1],
            (double)state.scene_view_smpl_fit_translation[2]);
        current_scene_transform.valid = state.scene_view_smpl_fit_valid
            && current_scene_transform.scale.allFinite()
            && current_scene_transform.translation.allFinite();

        std::vector<SmplSceneTransform> sequence_scene_transforms;
        if (sequence_override_ready)
        {
            sequence_scene_transforms.assign(sequence_pose_axis_angles.size(), current_scene_transform);
            const double target_height = smpl_native_height_from_betas_cloth(m_smplh_model, m_smplh_params.betas);
            int fitted_count = 0;
            if (sequence_rig_paths.size() == sequence_pose_axis_angles.size())
            {
                for (int i = 0; i < (int)sequence_pose_axis_angles.size(); ++i)
                {
                    Eigen::MatrixXd scene_joints;
                    std::vector<int> parents;
                    if (!load_sketch2rig_scene_joints_cloth(
                            std::filesystem::path(sequence_rig_paths[(size_t)i]),
                            target_height,
                            &scene_joints,
                            &parents))
                    {
                        console_add_line(state, "Sequence alignment warning: could not load rig for frame "
                            + std::to_string(i) + "; using fallback transform");
                        continue;
                    }

                    SmplhParams fit_params;
                    fit_params.betas = m_smplh_params.betas;
                    fit_params.pose_axis_angle = sequence_pose_axis_angles[(size_t)i];
                    MatrixX3dRM fit_vertices_rm;
                    std::string fit_err;
                    if (!smplh_evaluate_lbs(m_smplh_model, fit_params, fit_vertices_rm, &fit_err))
                    {
                        console_add_line(state, "Sequence alignment warning: LBS failed for frame "
                            + std::to_string(i) + "; using fallback transform");
                        continue;
                    }

                    Eigen::MatrixXd fit_vertices = fit_vertices_rm;
                    if (m_smplh_loaded_pose_mirror_x)
                        fit_vertices.col(0) *= -1.0;

                    SmplSceneTransform fitted;
                    if (!fit_smpl_vertices_to_scene_joints_cloth(
                            m_smplh_model,
                            fit_vertices,
                            scene_joints,
                            parents,
                            &fitted))
                    {
                        console_add_line(state, "Sequence alignment warning: fit failed for frame "
                            + std::to_string(i) + "; using fallback transform");
                        continue;
                    }

                    sequence_scene_transforms[(size_t)i] = fitted;
                    ++fitted_count;
                }
            }
            else
            {
                console_add_line(state, "Sequence alignment warning: rig path count does not match pose count; using fallback transforms");
            }

            if (fitted_count > 0)
            {
                console_add_line(state, "Sequence alignment: fitted "
                    + std::to_string(fitted_count)
                    + " sketch-frame 3D transforms");
            }
        }

        Eigen::VectorXd transition_start_axis_angle;
        bool pose_transition_ready = false;
        if (continue_from_last_frame
            && pose_transition_enabled
            && pose_transition_frames > 0
            && !transition_from_pose_path.empty()
            && n_joints > 0)
        {
            bool ignored_mirror = false;
            std::string pose_err;
            if (smplh_load_pose_npz(
                    std::filesystem::path(transition_from_pose_path),
                    transition_start_axis_angle,
                    nullptr,
                    &pose_err,
                    &ignored_mirror)
                && (int)transition_start_axis_angle.size() == dof)
            {
                pose_transition_ready = true;
            }
            else
            {
                console_add_line(state, "Sequence pose transition skipped: failed to load previous pose"
                    + (pose_err.empty() ? std::string() : (": " + pose_err)));
            }
        }
        const bool smooth_sequence_pose_ready =
            ((pose_transition_ready && (int)simulation_target_axis_angle.size() == dof)
                || (sequence_override_ready
                    && sequence_initial_smooth
                    && state.controls.smooth_sequence
                    && (int)simulation_target_axis_angle.size() == dof)
                || (!sequence_override_ready
                    && !continue_from_last_frame
                    && state.controls.smooth_sequence
                    && state.selection.pose_index >= 0
                    && m_smplh_loaded
                    && n_joints > 0
                    && (int)simulation_target_axis_angle.size() == dof
                    && m_garment_rest_vertices.rows() > 0));
        const float smooth_sequence_frac = std::clamp(state.controls.smooth_sequence_length, 0.0f, 1.0f);
        const int initial_steps_for_smooth = std::min(initial_simulation_steps, std::max(0, steps));
        const int smooth_sequence_steps = pose_transition_ready
            ? std::clamp(pose_transition_frames, 1, std::max(1, steps))
            : smooth_sequence_pose_ready
            ? static_cast<int>(std::round(smooth_sequence_frac * static_cast<float>(initial_steps_for_smooth)))
            : 0;
        const bool use_smooth_sequence_start_pose = smooth_sequence_steps > 0;

        SmplhParams tpose_params;
        tpose_params.betas = m_smplh_params.betas;
        tpose_params.pose_axis_angle = Eigen::VectorXd::Zero(dof);
        if (use_smooth_sequence_start_pose)
            tpose_params.pose_axis_angle.head<3>() = simulation_target_axis_angle.head<3>();

        MatrixX3dRM body_tpose_rm;
        std::string tpose_err;
        if (!smplh_evaluate_lbs(m_smplh_model, tpose_params, body_tpose_rm, &tpose_err))
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Failed to build T-pose body for simulation: " + tpose_err);
            return;
        }

        Eigen::MatrixXd body_tpose = body_tpose_rm;
        Eigen::MatrixXd cloth_V0(m_garment_rest_vertices.rows(), 3);
        const auto& F_body_map = m_smplh_model.F;
        for (int vi = 0; vi < cloth_V0.rows(); ++vi)
        {
            const int fi = m_garment_to_smpl.face_index[vi];
            if (fi < 0 || fi >= F_body_map.rows())
            {
                cloth_V0.row(vi) = m_garment_rest_vertices.row(vi);
                continue;
            }

            const int ia = F_body_map(fi, 0);
            const int ib = F_body_map(fi, 1);
            const int ic = F_body_map(fi, 2);
            if (ia < 0 || ib < 0 || ic < 0
                || ia >= body_tpose.rows() || ib >= body_tpose.rows() || ic >= body_tpose.rows())
            {
                cloth_V0.row(vi) = m_garment_rest_vertices.row(vi);
                continue;
            }

            const Eigen::RowVector3d bc = m_garment_to_smpl.barycentric.row(vi);
            const Eigen::RowVector3d p_def =
                bc[0] * body_tpose.row(ia) + bc[1] * body_tpose.row(ib) + bc[2] * body_tpose.row(ic);

            Eigen::RowVector3d T_def, B_def, N_def;
            if (!core::triangle_frame(body_tpose.row(ia), body_tpose.row(ib), body_tpose.row(ic), T_def, B_def, N_def))
            {
                if (!core::triangle_frame(
                        m_smplh_model.v_template.row(ia),
                        m_smplh_model.v_template.row(ib),
                        m_smplh_model.v_template.row(ic),
                        T_def,
                        B_def,
                        N_def))
                {
                    cloth_V0.row(vi) = m_garment_rest_vertices.row(vi);
                    continue;
                }
            }

            const Eigen::RowVector3d off_local = m_garment_to_smpl.offset_local.row(vi);
            const Eigen::RowVector3d off_def = off_local[0] * T_def + off_local[1] * B_def + off_local[2] * N_def;
            cloth_V0.row(vi) = p_def + off_def;
        }

        if (m_smplh_loaded_pose_mirror_x)
        {
            body_tpose.col(0) *= -1.0;
            cloth_V0.col(0) *= -1.0;
        }

        const SmplSceneTransform initial_scene_transform =
            sequence_override_ready && !sequence_scene_transforms.empty()
            ? sequence_scene_transforms.front()
            : current_scene_transform;
        if (initial_scene_transform.valid)
        {
            apply_scene_transform(body_tpose, initial_scene_transform);
            apply_scene_transform(cloth_V0, initial_scene_transform);
        }
        else
        {
            apply_smpl_scene_transform(state, body_tpose);
            apply_smpl_scene_transform(state, cloth_V0);
        }
        const Eigen::MatrixXd cloth_reference_V = cloth_V0;

        if (continue_from_last_frame
            && !m_cloth_sim_frames.empty()
            && m_cloth_sim_frames.back().rows() == cloth_V0.rows()
            && m_cloth_sim_frames.back().cols() == cloth_V0.cols())
        {
            cloth_V0 = m_cloth_sim_frames.back();
            console_add_line(state, "Simulation starts from previous cloth frame");
        }
        else
        {
            ClothPushOutResult push_res;
            std::string push_err;
            const double offset = std::max(0.0, (double)state.garment_min_cloth_skin_distance);
            const bool push_ok = push_cloth_vertices_outside_body(
                cloth_V0,
                cloth_F,
                body_tpose,
                body_F,
                offset,
                &push_res,
                &push_err);
            if (!push_ok)
            {
                state.controls.simulating = false;
                state.controls.sim_stage.clear();
                console_add_line(state, "ERROR: Simulation T-pose cloth push-out failed: "
                    + (push_err.empty() ? std::string("(unknown error)") : push_err));
                return;
            }
            console_add_line(state,
                std::string("Simulation starts from ")
                    + (use_smooth_sequence_start_pose ? "root-aligned T-pose cloth" : "T-pose cloth")
                    + " with betas; push-out moved "
                    + std::to_string(push_res.moved_vertices) + " vertices");
        }

        std::vector<int> fixed_vertex_indices;
        if ((int)m_garment_constraints.size() == cloth_V0.rows())
        {
            fixed_vertex_indices.reserve(m_garment_constraints.size());
            for (int vi = 0; vi < (int)m_garment_constraints.size(); ++vi)
                if (m_garment_constraints[(size_t)vi])
                    fixed_vertex_indices.push_back(vi);
        }

        ClothToSmplMapping constraint_attachment_mapping = m_garment_to_smpl;
        if (!fixed_vertex_indices.empty())
        {
            ConstraintSkinProjectionResult constraint_res;
            std::string constraint_err;
            const double offset = std::max(0.0, (double)state.garment_min_cloth_skin_distance);
            const bool constraint_ok = project_constraints_to_skin_margin(
                cloth_V0,
                constraint_attachment_mapping,
                fixed_vertex_indices,
                body_tpose,
                body_F,
                offset,
                &constraint_res,
                &constraint_err);
            if (!constraint_ok)
            {
                state.controls.simulating = false;
                state.controls.sim_stage.clear();
                console_add_line(state, "ERROR: Simulation constrained cloth projection failed: "
                    + (constraint_err.empty() ? std::string("(unknown error)") : constraint_err));
                return;
            }
            console_add_line(state,
                "Constrained T-pose vertices corrected to skin margin: "
                + std::to_string(constraint_res.corrected_vertices)
                + " updated, pushed_out=" + std::to_string(constraint_res.pushed_out_vertices)
                + ", pushed_in=" + std::to_string(constraint_res.pulled_in_vertices));
        }

        Eigen::MatrixXd body_V = m_smplh_vertices;
        if (sequence_override_ready && (int)simulation_target_axis_angle.size() == dof)
        {
            SmplhParams initial_body_params;
            initial_body_params.betas = m_smplh_params.betas;
            initial_body_params.pose_axis_angle = simulation_target_axis_angle;
            MatrixX3dRM initial_body_rm;
            std::string initial_body_err;
            if (smplh_evaluate_lbs(m_smplh_model, initial_body_params, initial_body_rm, &initial_body_err))
            {
                body_V = initial_body_rm;
                if (m_smplh_loaded_pose_mirror_x)
                    body_V.col(0) *= -1.0;
                apply_scene_transform(body_V, initial_scene_transform);
            }
            else
            {
                console_add_line(state, "Sequence alignment warning: initial body eval failed: " + initial_body_err);
            }
        }

        const auto wind_dir_from_angles = [](double yaw_deg, double pitch_deg)
        {
            constexpr double kPi = 3.14159265358979323846264338327950288;
            const double yaw = yaw_deg * (kPi / 180.0);
            const double pitch = pitch_deg * (kPi / 180.0);
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const double cp = std::cos(pitch);
            const double sp = std::sin(pitch);
            // Yaw about +Y, pitch about +X' (after yaw). Base forward is +Z.
            return Eigen::Vector3d(sy * cp, -sp, cy * cp);
        };

        ClothXpbdParams xpbd_params;
        xpbd_params.dt = (double)state.xpbd_params.dt;
        xpbd_params.duration_seconds = duration_for_steps;
        xpbd_params.reference_positions = cloth_reference_V;
        xpbd_params.solver_iterations = state.xpbd_params.solver_iterations;
        xpbd_params.gravity_y = (double)state.xpbd_params.gravity;
        xpbd_params.damping = (double)state.xpbd_params.damping;

        xpbd_params.stretch_enabled = state.xpbd_params.stretch_enabled;
        xpbd_params.stretch_compliance = std::max(0.0, (double)state.xpbd_params.stretch_compliance);
        xpbd_params.base_stretch_mult = (double)state.xpbd_params.base_stretch_mult;
        xpbd_params.substeps = std::max(1, state.xpbd_params.substeps);
        xpbd_params.bend_enabled = state.xpbd_params.bend_enabled;
        xpbd_params.bending_method = state.xpbd_params.bending_method;
        xpbd_params.bend_compliance = std::max(0.0, (double)state.xpbd_params.bend_compliance);
        xpbd_params.dihedral_compliance = std::max(0.0, (double)state.xpbd_params.dihedral_compliance);
        xpbd_params.collision_enabled = state.xpbd_params.collision_enabled;
        xpbd_params.collision_compliance = std::max(0.0, (double)state.xpbd_params.collision_compliance);
        xpbd_params.collision_thickness = (double)state.xpbd_params.collision_thickness;
        xpbd_params.self_collision_enabled = state.xpbd_params.self_collision_enabled;
        xpbd_params.self_collision_compliance = std::max(0.0, (double)state.xpbd_params.self_collision_compliance);
        xpbd_params.self_collision_thickness = (double)state.xpbd_params.self_collision_thickness;

        xpbd_params.enable_friction = state.xpbd_params.enable_friction;
        xpbd_params.friction_mu = (double)state.xpbd_params.friction_mu;

        xpbd_params.wind_dir = wind_dir_from_angles((double)state.xpbd_params.wind_yaw_deg, (double)state.xpbd_params.wind_pitch_deg);
        xpbd_params.wind_force = std::max(0.0, (double)state.xpbd_params.wind_force);
        xpbd_params.wind_random_magnitude = std::max(0.0, (double)state.xpbd_params.wind_random_magnitude);
        xpbd_params.wind_random_time_scale = std::max(0.0, (double)state.xpbd_params.wind_random_time_scale);
        xpbd_params.wind_random_spatial_scale = std::max(0.0, (double)state.xpbd_params.wind_random_spatial_scale);

        xpbd_params.max_frames = max_frames_for_progress;

        const bool use_gpu_simulation = state.xpbd_params.device == SimulationDevice::Gpu;
        if (use_gpu_simulation && !state.gpu_simulation_available)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: CUDA simulation is unavailable.");
            return;
        }

        const bool save_smpl_sequence = state.controls.save_smpl_sequence;
        const CollisionGeometryType collision_geometry = state.xpbd_params.collision_geometry;
        const bool simplify_smpl = collision_geometry == CollisionGeometryType::SimplifiedSmpl;
        const bool use_simple_proxy = collision_geometry == CollisionGeometryType::SimpleProxy;
        const bool prepare_collision_proxy = simplify_smpl || use_simple_proxy;
        const double keep_percent = std::clamp((double)state.xpbd_params.smpl_geometry_percent, 1.0, 100.0);
        const int simplification_units = simplify_smpl && keep_percent < 99.5 ? 1 : 0;
        int sequence_override_body_steps = 0;
        int sequence_initial_hold_steps = 0;
        if (sequence_override_ready)
        {
            const int hold_steps = std::max(
                1,
                (int)std::llround((1.0 / (double)sequence_frame_rate) / std::max(1e-6, xpbd_params.dt)));
            const int blend_frame_count = std::max(1, sequence_transition_frames);
            const int transition_steps = std::max(
                1,
                (int)std::llround(((double)blend_frame_count / (double)sequence_frame_rate)
                    / std::max(1e-6, xpbd_params.dt)));
            sequence_initial_hold_steps = sequence_initial_smooth
                ? std::max(0, initial_steps_for_smooth - smooth_sequence_steps)
                : hold_steps;
            sequence_override_body_steps = sequence_initial_hold_steps;
            for (int i = 1; i < (int)sequence_pose_axis_angles.size(); ++i)
            {
                sequence_override_body_steps += sequence_pose_keys[(size_t)i] == sequence_pose_keys[(size_t)i - 1u]
                    ? hold_steps
                    : transition_steps;
            }
        }
        const int body_sequence_units = smooth_sequence_steps + sequence_override_body_steps;
        const int target_units = fixed_vertex_indices.empty() ? 0 : std::max(1, body_sequence_units);
        const int proxy_units = xpbd_params.collision_enabled && prepare_collision_proxy ? body_sequence_units + 1 : 0;
        const int sequence_units = body_sequence_units + target_units + simplification_units;
        const int preparation_units = sequence_units + proxy_units;

        m_cloth_sim_progress_steps_total = preparation_units + steps;
        state.controls.sim_steps_total = m_cloth_sim_progress_steps_total;
        state.controls.sim_steps_done = 0;
        state.controls.sim_sequence_steps_total = sequence_units;
        state.controls.sim_sequence_steps_done = 0;
        state.controls.sim_proxy_steps_total = proxy_units;
        state.controls.sim_proxy_steps_done = 0;
        state.controls.sim_solver_steps_total = steps;
        state.controls.sim_solver_steps_done = 0;

        m_cloth_sim_frames.clear();
        state.controls.frame_count = 0;
        m_collision_proxy_sim_frames.clear();
        m_collision_proxy_sim_F.resize(0, 3);
        state.controls.current_frame = 0;
        state.controls.playing = false;

        // Snapshot frame 0 so preview updates don't corrupt the restart baseline.
        m_cloth_sim_reset_frame0 = cloth_V0;
        m_cloth_sim_reset_frame0_valid = true;
        m_cloth_sim_reset_pose_version = m_pose_version;
        m_cloth_sim_reset_garment_version = m_garment_version;

        m_cloth_sim_pose_version = m_pose_version;
        m_cloth_sim_garment_version = m_garment_version;
        m_cloth_sim_require_pose_version_match = !sequence_override_ready;
        m_cloth_sim_play_on_finish = play_on_finish;

        m_cloth_sim_preview_state = std::make_shared<ClothSimPreviewState>();
        m_cloth_sim_preview_last_seq_applied = 0;
        m_cloth_sim_preview_enabled = std::make_shared<std::atomic_bool>(state.controls.preview);
        const auto preview_state = m_cloth_sim_preview_state;
        const auto preview_enabled = m_cloth_sim_preview_enabled;

        m_cloth_sim_cancel_flag = std::make_shared<std::atomic_bool>(false);
        const auto cancel_flag = m_cloth_sim_cancel_flag;

        m_cloth_sim_progress_steps_done = std::make_shared<std::atomic_int>(0);
        const auto progress_steps_done = m_cloth_sim_progress_steps_done;
        const int initial_progress_stage = sequence_units > 0 ? 0 : (proxy_units > 0 ? 3 : 4);
        m_cloth_sim_progress_stage = std::make_shared<std::atomic_int>(initial_progress_stage);
        const auto progress_stage = m_cloth_sim_progress_stage;

        const Eigen::RowVector3d display_scale(
            (double)state.scene_view_smpl_fit_scale[0],
            (double)state.scene_view_smpl_fit_scale[1],
            (double)state.scene_view_smpl_fit_scale[2]);
        const Eigen::RowVector3d display_translation(
            (double)state.scene_view_smpl_fit_translation[0],
            (double)state.scene_view_smpl_fit_translation[1],
            (double)state.scene_view_smpl_fit_translation[2]);
        const bool apply_display_transform = state.scene_view_smpl_fit_valid
            && display_scale.allFinite() && display_translation.allFinite();

        m_cloth_sim_job_running = true;
        state.controls.simulating = true;
        state.controls.sim_pipeline_stage = initial_progress_stage < 3 ? 0 : (initial_progress_stage == 3 ? 1 : 2);
        state.controls.sim_stage = initial_progress_stage < 3
            ? "Building SMPL sequence"
            : (initial_progress_stage == 3 ? "Preparing SMPL proxy" : "Simulating XPBD");
        console_add_line(state, use_gpu_simulation
            ? "Starting cloth simulation (CUDA Jacobi XPBD)..."
            : "Starting cloth simulation (CPU XPBD)...");

        m_cloth_sim_future = std::async(std::launch::async,
            [cloth_V0 = std::move(cloth_V0),
                cloth_F = std::move(cloth_F),
                body_V = std::move(body_V),
                body_F = std::move(body_F),
                xpbd_params,
                body_tpose = std::move(body_tpose),
                fixed_vertex_indices = std::move(fixed_vertex_indices),
                constraint_attachment_mapping = std::move(constraint_attachment_mapping),
                smplh_model = m_smplh_model,
                target_aa = simulation_target_axis_angle,
                smpl_betas = m_smplh_params.betas,
                mirror_display_x = m_smplh_loaded_pose_mirror_x,
                display_scale,
                display_translation,
                apply_display_transform,
                current_scene_transform,
                sequence_scene_transforms = std::move(sequence_scene_transforms),
                smooth_sequence_steps,
                sequence_override_ready,
                sequence_pose_axis_angles = std::move(sequence_pose_axis_angles),
                sequence_pose_keys = std::move(sequence_pose_keys),
                sequence_frame_rate,
                sequence_transition_frames,
                sequence_initial_hold_steps,
                sequence_initial_smooth,
                pose_transition_ready,
                transition_start_axis_angle = std::move(transition_start_axis_angle),
                n_joints,
                dof,
                save_smpl_sequence,
                simplify_smpl,
                use_simple_proxy,
                prepare_collision_proxy,
                keep_percent,
                use_gpu_simulation,
                preparation_units,
                cancel_flag,
                progress_steps_done,
                progress_stage,
                preview_state,
                preview_enabled]() mutable -> ClothSimJobResult
            {
                ClothSimJobResult jr;
                std::string err;
                ClothXpbdResult xpbd_res;
                const auto job_start = std::chrono::steady_clock::now();

                const auto seconds_since = [](std::chrono::steady_clock::time_point start)
                {
                    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                };
                const auto append_timing = [](std::vector<std::string>& out, const char* label, double seconds)
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << label << ": " << seconds << "s";
                    out.push_back(oss.str());
                };

                const auto advance_progress = [&]() { progress_steps_done->fetch_add(1); };
                const auto fallback_scene_transform = [&]() {
                    SmplSceneTransform fallback = current_scene_transform;
                    if (!fallback.valid && apply_display_transform)
                    {
                        fallback.scale = display_scale;
                        fallback.translation = display_translation;
                        fallback.valid = true;
                    }
                    return fallback;
                };
                const SmplSceneTransform default_transform = fallback_scene_transform();
                const auto sequence_transform_at = [&](int index) {
                    if (index >= 0
                        && index < (int)sequence_scene_transforms.size()
                        && sequence_scene_transforms[(size_t)index].valid)
                        return sequence_scene_transforms[(size_t)index];
                    return default_transform;
                };

                if (smooth_sequence_steps > 0)
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(0);
                    const auto aa_to_quat = [](const Eigen::Vector3d& value)
                    {
                        const double angle = value.norm();
                        return angle < 1e-12
                            ? Eigen::Quaterniond::Identity()
                            : Eigen::Quaterniond(Eigen::AngleAxisd(angle, value / angle));
                    };
                    const Eigen::Quaterniond root_target = aa_to_quat(target_aa.segment<3>(0));
                    const bool use_transition_start =
                        pose_transition_ready && (int)transition_start_axis_angle.size() == dof;
                    SmplhParams interp_params;
                    interp_params.betas = smpl_betas;
                    interp_params.pose_axis_angle.resize(dof);
                    xpbd_params.body_sequence.reserve((size_t)smooth_sequence_steps);
                    if (save_smpl_sequence)
                        jr.smpl_frames.reserve((size_t)smooth_sequence_steps);

                    for (int i = 0; i < smooth_sequence_steps; ++i)
                    {
                        if (cancel_flag->load())
                        {
                            jr.error = "Simulation preparation cancelled";
                            return jr;
                        }
                        const double t = smooth_sequence_steps == 1
                            ? 1.0 : (double)i / (double)(smooth_sequence_steps - 1);
                        for (int j = 0; j < n_joints; ++j)
                        {
                            const Eigen::Quaterniond target = aa_to_quat(target_aa.segment<3>(3 * j));
                            const Eigen::Quaterniond start = use_transition_start
                                ? aa_to_quat(transition_start_axis_angle.segment<3>(3 * j))
                                : (j == 0 ? root_target : Eigen::Quaterniond::Identity());
                            const Eigen::AngleAxisd result(start.slerp(t, target));
                            interp_params.pose_axis_angle.segment<3>(3 * j) = result.angle() * result.axis();
                        }

                        MatrixX3dRM skinned;
                        if (!smplh_evaluate_lbs(smplh_model, interp_params, skinned, &err))
                        {
                            jr.error = "LBS failed for smooth sequence step " + std::to_string(i) + ": " + err;
                            return jr;
                        }
                        Eigen::MatrixXd frame = skinned;
                        if (mirror_display_x)
                            frame.col(0) *= -1.0;
                        const SmplSceneTransform smooth_start_transform =
                            pose_transition_ready ? default_transform : sequence_transform_at(0);
                        const SmplSceneTransform smooth_target_transform =
                            sequence_override_ready ? sequence_transform_at(0) : default_transform;
                        apply_scene_transform(
                            frame,
                            lerp_scene_transform(smooth_start_transform, smooth_target_transform, t));
                        if (save_smpl_sequence)
                            jr.smpl_frames.push_back(frame);
                        xpbd_params.body_sequence.push_back(std::move(frame));
                        advance_progress();
                    }
                    const double seconds = seconds_since(step_start);
                    jr.sequence_seconds += seconds;
                    append_timing(jr.sequence_timing_details, "Smooth SMPL sequence", seconds);
                }

                if (sequence_override_ready && !sequence_pose_axis_angles.empty())
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(0);
                    const auto aa_to_quat = [](const Eigen::Vector3d& value)
                    {
                        const double angle = value.norm();
                        return angle < 1e-12
                            ? Eigen::Quaterniond::Identity()
                            : Eigen::Quaterniond(Eigen::AngleAxisd(angle, value / angle));
                    };
                    const auto append_pose_frame = [&](const Eigen::VectorXd& aa, int pose_index) -> bool
                    {
                        SmplhParams params;
                        params.betas = smpl_betas;
                        params.pose_axis_angle = aa;
                        MatrixX3dRM skinned;
                        if (!smplh_evaluate_lbs(smplh_model, params, skinned, &err))
                            return false;
                        Eigen::MatrixXd frame = skinned;
                        if (mirror_display_x)
                            frame.col(0) *= -1.0;
                        apply_scene_transform(frame, sequence_transform_at(pose_index));
                        if (save_smpl_sequence)
                            jr.smpl_frames.push_back(frame);
                        xpbd_params.body_sequence.push_back(std::move(frame));
                        advance_progress();
                        return true;
                    };
                    const auto append_interpolation = [&](
                        const Eigen::VectorXd& from,
                        const Eigen::VectorXd& to,
                        int from_index,
                        int to_index,
                        int count) -> bool
                    {
                        SmplhParams params;
                        params.betas = smpl_betas;
                        params.pose_axis_angle.resize(dof);
                        const SmplSceneTransform from_transform = sequence_transform_at(from_index);
                        const SmplSceneTransform to_transform = sequence_transform_at(to_index);
                        for (int i = 0; i < count; ++i)
                        {
                            if (cancel_flag->load())
                            {
                                jr.error = "Simulation preparation cancelled";
                                return false;
                            }
                            const double t = (double)(i + 1) / (double)std::max(1, count);
                            for (int j = 0; j < n_joints; ++j)
                            {
                                const Eigen::Quaterniond start = aa_to_quat(from.segment<3>(3 * j));
                                const Eigen::Quaterniond target = aa_to_quat(to.segment<3>(3 * j));
                                const Eigen::AngleAxisd result(start.slerp(t, target));
                                params.pose_axis_angle.segment<3>(3 * j) = result.angle() * result.axis();
                            }
                            MatrixX3dRM skinned;
                            if (!smplh_evaluate_lbs(smplh_model, params, skinned, &err))
                                return false;
                            Eigen::MatrixXd frame = skinned;
                            if (mirror_display_x)
                                frame.col(0) *= -1.0;
                            apply_scene_transform(
                                frame,
                                lerp_scene_transform(from_transform, to_transform, t));
                            if (save_smpl_sequence)
                                jr.smpl_frames.push_back(frame);
                            xpbd_params.body_sequence.push_back(std::move(frame));
                            advance_progress();
                        }
                        return true;
                    };

                    const int hold_steps = std::max(
                        1,
                        (int)std::llround((1.0 / (double)std::max(1, sequence_frame_rate))
                            / std::max(1e-6, xpbd_params.dt)));
                    const int blend_frame_count = std::max(1, sequence_transition_frames);
                    const int transition_steps = std::max(
                        1,
                        (int)std::llround(((double)blend_frame_count / (double)std::max(1, sequence_frame_rate))
                            / std::max(1e-6, xpbd_params.dt)));

                    if (sequence_initial_hold_steps > 0)
                    {
                        xpbd_params.body_sequence.reserve(xpbd_params.body_sequence.size() + (size_t)sequence_initial_hold_steps);
                        for (int i = 0; i < sequence_initial_hold_steps; ++i)
                        {
                            if (cancel_flag->load())
                            {
                                jr.error = "Simulation preparation cancelled";
                                return jr;
                            }
                            if (!append_pose_frame(sequence_pose_axis_angles.front(), 0))
                            {
                                jr.error = "LBS failed for sequence hold frame: " + err;
                                return jr;
                            }
                        }
                    }

                    for (int i = 1; i < (int)sequence_pose_axis_angles.size(); ++i)
                    {
                        const bool same_pose = sequence_pose_keys[(size_t)i] == sequence_pose_keys[(size_t)i - 1u];
                        if (same_pose)
                        {
                            for (int step = 0; step < hold_steps; ++step)
                            {
                                if (cancel_flag->load())
                                {
                                    jr.error = "Simulation preparation cancelled";
                                    return jr;
                                }
                                if (!append_pose_frame(sequence_pose_axis_angles[(size_t)i], i))
                                {
                                    jr.error = "LBS failed for sequence hold frame: " + err;
                                    return jr;
                                }
                            }
                        }
                        else if (!append_interpolation(
                                sequence_pose_axis_angles[(size_t)i - 1u],
                                sequence_pose_axis_angles[(size_t)i],
                                i - 1,
                                i,
                                transition_steps))
                        {
                            if (jr.error.empty())
                                jr.error = "LBS failed for sequence interpolation: " + err;
                            return jr;
                        }
                    }

                    const double seconds = seconds_since(step_start);
                    jr.sequence_seconds += seconds;
                    append_timing(jr.sequence_timing_details, "Canva SMPL sequence", seconds);
                }

                if (!fixed_vertex_indices.empty())
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(1);
                    Eigen::MatrixXd rest_body = smplh_model.v_template;
                    if (mirror_display_x)
                        rest_body.col(0) *= -1.0;
                    xpbd_params.fixed_vertex_indices = fixed_vertex_indices;
                    auto& targets = xpbd_params.fixed_vertex_target_sequence;
                    if (xpbd_params.body_sequence.empty())
                    {
                        targets.push_back(compute_cloth_smpl_targets(
                            fixed_vertex_indices, cloth_V0, constraint_attachment_mapping,
                            body_V, body_F, rest_body));
                        advance_progress();
                    }
                    else
                    {
                        targets.reserve(xpbd_params.body_sequence.size());
                        for (const Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            targets.push_back(compute_cloth_smpl_targets(
                                fixed_vertex_indices, cloth_V0, constraint_attachment_mapping,
                                frame, body_F, rest_body));
                            advance_progress();
                        }
                    }
                    const double seconds = seconds_since(step_start);
                    jr.sequence_seconds += seconds;
                    append_timing(jr.sequence_timing_details, "Cloth constraint targets", seconds);
                }

                if (simplify_smpl && keep_percent < 99.5)
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(2);
                    SmplSimplification simplification;
                    Eigen::MatrixXd simplified_body;
                    bool simpl_ok = build_smpl_simplification(
                        body_tpose, body_F, smplh_model.weights, keep_percent,
                        simplification, &err)
                        && apply_smpl_simplification(body_V, simplification, simplified_body, &err);
                    std::vector<Eigen::MatrixXd> sequence;
                    if (simpl_ok)
                    {
                        sequence.reserve(xpbd_params.body_sequence.size());
                        for (const Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            Eigen::MatrixXd simplified_frame;
                            if (!apply_smpl_simplification(frame, simplification, simplified_frame, &err))
                            {
                                simpl_ok = false;
                                break;
                            }
                            sequence.push_back(std::move(simplified_frame));
                        }
                    }
                    if (simpl_ok)
                    {
                        body_V = std::move(simplified_body);
                        body_F = simplification.F;
                        xpbd_params.body_sequence = std::move(sequence);
                    }
                    err.clear();
                    advance_progress();
                    const double seconds = seconds_since(step_start);
                    jr.sequence_seconds += seconds;
                    append_timing(jr.sequence_timing_details, "SMPL simplification", seconds);
                }

                if (xpbd_params.collision_enabled && prepare_collision_proxy && body_V.rows() > 0 && body_F.rows() > 0)
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(3);
                    if (use_simple_proxy)
                    {
                        SmplCollisionProxy proxy;
                        Eigen::MatrixXd proxy_body;
                        Eigen::MatrixXd proxy_capsules;
                        Eigen::MatrixXi proxy_F;
                        if (!build_smpl_collision_proxy(body_tpose, smplh_model, proxy, &err)
                            || !apply_smpl_collision_proxy(proxy, body_V, smplh_model, proxy_body, proxy_F, &err)
                            || !apply_smpl_collision_proxy_capsules(proxy, body_V, smplh_model, proxy_capsules, &err))
                        {
                            jr.error = "Simple collision proxy failed: " + err;
                            return jr;
                        }
                        body_V = std::move(proxy_body);
                        body_F = proxy_F;
                        xpbd_params.body_capsules = std::move(proxy_capsules);
                        jr.collision_proxy_F = proxy_F;
                        jr.collision_proxy_frames.push_back(body_V);
                        advance_progress();

                        std::vector<Eigen::MatrixXd> proxy_sequence;
                        std::vector<Eigen::MatrixXd> capsule_sequence;
                        proxy_sequence.reserve(xpbd_params.body_sequence.size());
                        capsule_sequence.reserve(xpbd_params.body_sequence.size());
                        jr.collision_proxy_frames.reserve(xpbd_params.body_sequence.size() + 1u);
                        for (const Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            if (cancel_flag->load())
                            {
                                jr.error = "Simulation preparation cancelled";
                                return jr;
                            }
                            Eigen::MatrixXd proxy_frame;
                            Eigen::MatrixXd capsule_frame;
                            Eigen::MatrixXi frame_F;
                            if (!apply_smpl_collision_proxy(proxy, frame, smplh_model, proxy_frame, frame_F, &err)
                                || !apply_smpl_collision_proxy_capsules(proxy, frame, smplh_model, capsule_frame, &err))
                            {
                                jr.error = "Simple collision proxy frame failed: " + err;
                                return jr;
                            }
                            proxy_sequence.push_back(proxy_frame);
                            capsule_sequence.push_back(std::move(capsule_frame));
                            jr.collision_proxy_frames.push_back(std::move(proxy_frame));
                            advance_progress();
                        }
                        xpbd_params.body_sequence = std::move(proxy_sequence);
                        xpbd_params.body_capsule_sequence = std::move(capsule_sequence);
                        append_timing(jr.proxy_timing_details, "Build simple capsule proxy", seconds_since(step_start));
                    }
                    else
                    {
                        const double inflation = std::clamp(
                            std::max(0.0015, 0.5 * std::max(0.0, xpbd_params.collision_thickness)),
                            0.0015, 0.008);
                        const double clearance = std::clamp(std::max(0.0020, 1.25 * inflation), 0.0020, 0.012);
                        prepare_xpbd_body_proxy_frame(body_V, body_F, inflation, clearance);
                        advance_progress();
                        for (Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            if (cancel_flag->load())
                            {
                                jr.error = "Simulation preparation cancelled";
                                return jr;
                            }
                            prepare_xpbd_body_proxy_frame(frame, body_F, inflation, clearance);
                            advance_progress();
                        }
                        append_timing(jr.proxy_timing_details, "Inflate and relax SMPL proxy", seconds_since(step_start));
                    }
                    jr.proxy_seconds += seconds_since(step_start);
                }

                progress_stage->store(4);
                progress_steps_done->store(preparation_units);

                struct PreviewCb
                {
                    std::shared_ptr<ClothSimPreviewState> state;
                    std::shared_ptr<std::atomic_bool> enabled;
                    std::shared_ptr<std::atomic_int> progress;
                    int progress_offset = 0;
                    Eigen::MatrixXi body_F;
                    std::chrono::steady_clock::time_point last;
                    std::chrono::milliseconds min_interval{ 100 }; // low-FPS preview
                };

                auto on_frame_xpbd = [](const Eigen::MatrixXd& frame_V, const Eigen::MatrixXd* body_V, int steps_simulated, void* user)
                {
                    auto* pcb = static_cast<PreviewCb*>(user);
                    if (!pcb || !pcb->state || !pcb->enabled)
                        return;
                    if (pcb->progress)
                        pcb->progress->store(pcb->progress_offset + steps_simulated);
                    if (!pcb->enabled->load())
                        return;

                    const auto now = std::chrono::steady_clock::now();
                    if (pcb->last.time_since_epoch().count() != 0 && (now - pcb->last) < pcb->min_interval)
                        return;
                    pcb->last = now;

                    {
                        std::lock_guard<std::mutex> lk(pcb->state->mutex);
                        pcb->state->V = frame_V;
                        pcb->state->valid = true;
                        if (body_V)
                        {
                            pcb->state->body_V = *body_V;
                            pcb->state->body_F = pcb->body_F;
                            pcb->state->body_valid = true;
                            pcb->state->body_topology_valid = (pcb->body_F.rows() > 0 && pcb->body_F.cols() == 3);
                        }
                        else
                        {
                            pcb->state->body_valid = false;
                            pcb->state->body_topology_valid = false;
                        }
                        pcb->state->seq.fetch_add(1, std::memory_order_release);
                    }
                };

                PreviewCb preview_cb;
                preview_cb.state = preview_state;
                preview_cb.enabled = preview_enabled;
                preview_cb.progress = progress_steps_done;
                preview_cb.progress_offset = preparation_units;
                preview_cb.body_F = body_F;

                const auto solver_start = std::chrono::steady_clock::now();
                bool ok = use_gpu_simulation
                    ? run_cloth_cuda_jacobi(
                        cloth_V0, cloth_F,
                        body_V, body_F,
                        xpbd_params,
                        &xpbd_res,
                        &err,
                        cancel_flag.get(),
                        nullptr,
                        on_frame_xpbd,
                        &preview_cb)
                    : run_cloth_xpbd(
                        cloth_V0, cloth_F,
                        body_V, body_F,
                        xpbd_params,
                        &xpbd_res,
                        &err,
                        cancel_flag.get(),
                        nullptr,
                        on_frame_xpbd,
                        &preview_cb);
                jr.solver_seconds = seconds_since(solver_start);
                if (ok)
                {
                    jr.frames = std::move(xpbd_res.frames);
                    jr.steps_simulated = xpbd_res.stats.steps_simulated;
                    jr.last_step_contacts = xpbd_res.stats.last_step_contacts + xpbd_res.stats.last_step_self_contacts;

                    const ClothXpbdStats& s = xpbd_res.stats;
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    const auto pct = [&](double ms)
                    {
                        return (s.ms_total > 0.0) ? (100.0 * ms / s.ms_total) : 0.0;
                    };

                    const double ms_collision_broadphase = s.ms_candidates + s.ms_collision_broad;
                    const double ms_collision_narrowphase = s.ms_collision_narrow;

                    oss << (use_gpu_simulation
                        ? "CUDA Jacobi XPBD timing breakdown (ms)\n"
                        : "CPU XPBD timing breakdown (ms)\n");
                    oss << "  total: " << s.ms_total << "\n";
                    oss << "  bvh build: " << s.ms_bvh_build << " (" << pct(s.ms_bvh_build) << "%)\n";
                    oss << "  predict: " << s.ms_predict << " (" << pct(s.ms_predict) << "%)\n";
                    oss << "  wind: " << s.ms_wind << " (" << pct(s.ms_wind) << "%)\n";
                    oss << "  stretch: " << s.ms_stretch << " (" << pct(s.ms_stretch) << "%)\n";
                    oss << "  bend: " << s.ms_bend << " (" << pct(s.ms_bend) << "%)\n";
                    oss << "  collision contacts total: " << s.ms_collision_bvh << " (" << pct(s.ms_collision_bvh) << "%)\n";
                    oss << "    broad phase: " << ms_collision_broadphase << " (" << pct(ms_collision_broadphase) << "%)\n";
                    oss << "      candidates (AABB): " << s.ms_candidates << " (" << pct(s.ms_candidates) << "%)\n";
                    oss << "      filters/bookkeeping: " << s.ms_collision_broad << " (" << pct(s.ms_collision_broad) << "%)\n";
                    oss << "    narrow phase (" << (use_simple_proxy ? "capsule query" : "BVH query") << "): "
                        << ms_collision_narrowphase << " (" << pct(ms_collision_narrowphase) << "%)\n";
                    oss << "  collision solve: " << s.ms_collision_solve << " (" << pct(s.ms_collision_solve) << "%)\n";
                    oss << "  self collision contacts: " << s.ms_self_collision_bvh << " (" << pct(s.ms_self_collision_bvh) << "%)\n";
                    oss << "  self collision solve: " << s.ms_self_collision_solve << " (" << pct(s.ms_self_collision_solve) << "%)\n";
                    oss << "  friction: " << s.ms_friction << " (" << pct(s.ms_friction) << "%)";
                    jr.timing_report = oss.str();

                    append_timing(jr.solver_timing_details, "XPBD wall time", jr.solver_seconds);
                    const auto append_ms = [&](const char* label, double ms)
                    {
                        std::ostringstream detail;
                        detail << std::fixed << std::setprecision(2)
                            << label << ": " << (ms / 1000.0) << "s (" << pct(ms) << "%)";
                        jr.solver_timing_details.push_back(detail.str());
                    };
                    append_ms("BVH build", s.ms_bvh_build);
                    append_ms("Predict", s.ms_predict);
                    append_ms("Wind", s.ms_wind);
                    append_ms("Stretch", s.ms_stretch);
                    append_ms("Bend", s.ms_bend);
                    append_ms("Collision contacts", s.ms_collision_bvh);
                    append_ms("Collision solve", s.ms_collision_solve);
                    append_ms("Self collision contacts", s.ms_self_collision_bvh);
                    append_ms("Self collision solve", s.ms_self_collision_solve);
                    append_ms("Friction", s.ms_friction);
                }

                jr.pipeline_seconds = seconds_since(job_start);
                jr.ok = ok;
                jr.error = std::move(err);
                return jr;
            });
    }

    void Renderer::update_cloth_simulation_job(AppState& state)
    {
        if (!m_cloth_sim_job_running)
            return;

        if (m_cloth_sim_preview_enabled)
            m_cloth_sim_preview_enabled->store(state.controls.preview);

        if (m_cloth_sim_progress_steps_done)
        {
            const int done = std::max(0, m_cloth_sim_progress_steps_done->load());
            state.controls.sim_steps_done = done;
            state.controls.sim_steps_total = std::max(0, m_cloth_sim_progress_steps_total);
            state.controls.sim_sequence_steps_done = std::clamp(
                done, 0, state.controls.sim_sequence_steps_total);
            state.controls.sim_proxy_steps_done = std::clamp(
                done - state.controls.sim_sequence_steps_total,
                0,
                state.controls.sim_proxy_steps_total);
            state.controls.sim_solver_steps_done = std::clamp(
                done - state.controls.sim_sequence_steps_total - state.controls.sim_proxy_steps_total,
                0,
                state.controls.sim_solver_steps_total);
        }
        if (m_cloth_sim_progress_stage)
        {
            static constexpr const char* stages[] = {
                "Building smooth SMPL sequence",
                "Building cloth constraints",
                "Simplifying SMPL",
                "Preparing SMPL proxy",
                "Simulating XPBD"
            };
            const int stage = std::clamp(m_cloth_sim_progress_stage->load(), 0, 4);
            state.controls.sim_stage = stages[stage];
            state.controls.sim_pipeline_stage = stage < 3 ? 0 : (stage == 3 ? 1 : 2);
        }

        if (!future_ready(m_cloth_sim_future))
            return;

        ClothSimJobResult jr = m_cloth_sim_future.get();
        m_cloth_sim_job_running = false;
        state.controls.simulating = false;

        const bool was_cancelled = (m_cloth_sim_cancel_flag && m_cloth_sim_cancel_flag->load());
        if (!was_cancelled)
        {
            state.controls.sim_steps_done = std::max(state.controls.sim_steps_done, state.controls.sim_steps_total);
            state.controls.sim_sequence_steps_done = state.controls.sim_sequence_steps_total;
            state.controls.sim_proxy_steps_done = state.controls.sim_proxy_steps_total;
            state.controls.sim_solver_steps_done = state.controls.sim_solver_steps_total;
        }

        m_cloth_sim_preview_state.reset();
        m_cloth_sim_preview_enabled.reset();
        m_cloth_sim_preview_last_seq_applied = 0;

        if (!jr.ok)
        {
            state.controls.sim_stage.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            m_cloth_sim_last_error = jr.error;
            console_add_line(state, std::string("ERROR: Cloth simulation failed: ") + (jr.error.empty() ? "(unknown error)" : jr.error));
            return;
        }

        if (m_cloth_sim_restart_pending)
        {
            // Discard result and restart immediately to avoid a transient "stopped" state.
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            state.controls.playing = false;
            state.controls.current_frame = 0;

            m_cloth_sim_restart_pending = false;
            state.cloth_simulate_play_on_finish = m_cloth_sim_restart_play_on_finish;
            console_add_line(state, "Restarting cloth simulation…");
            process_cloth_simulate(state);
            return;
        }

        const bool pose_changed =
            m_cloth_sim_require_pose_version_match && m_cloth_sim_pose_version != m_pose_version;
        const bool garment_changed = m_cloth_sim_garment_version != m_garment_version;
        if (pose_changed || garment_changed)
        {
            console_add_line(state, "Cloth simulation finished, but selection changed; result discarded");
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            state.controls.sim_stage.clear();
            return;
        }

        if (m_cloth_sim_append_result && !m_cloth_sim_append_prefix_frames.empty())
        {
            m_cloth_sim_frames = std::move(m_cloth_sim_append_prefix_frames);
            m_cloth_sim_frames.insert(
                m_cloth_sim_frames.end(),
                std::make_move_iterator(jr.frames.begin()),
                std::make_move_iterator(jr.frames.end()));

            m_smpl_sim_frames = std::move(m_smpl_sim_append_prefix_frames);
            m_smpl_sim_frames.insert(
                m_smpl_sim_frames.end(),
                std::make_move_iterator(jr.smpl_frames.begin()),
                std::make_move_iterator(jr.smpl_frames.end()));

            m_collision_proxy_sim_frames = std::move(m_collision_proxy_append_prefix_frames);
            m_collision_proxy_sim_frames.insert(
                m_collision_proxy_sim_frames.end(),
                std::make_move_iterator(jr.collision_proxy_frames.begin()),
                std::make_move_iterator(jr.collision_proxy_frames.end()));
        }
        else
        {
            m_cloth_sim_frames = std::move(jr.frames);
            m_smpl_sim_frames = std::move(jr.smpl_frames);
            m_collision_proxy_sim_frames = std::move(jr.collision_proxy_frames);
        }
        m_cloth_sim_append_prefix_frames.clear();
        m_smpl_sim_append_prefix_frames.clear();
        m_collision_proxy_append_prefix_frames.clear();
        m_cloth_sim_append_result = false;
        m_collision_proxy_sim_F = std::move(jr.collision_proxy_F);
        state.controls.frame_count = (int)m_cloth_sim_frames.size();
        state.controls.sim_sequence_seconds = jr.sequence_seconds;
        state.controls.sim_proxy_seconds = jr.proxy_seconds;
        state.controls.sim_solver_seconds = jr.solver_seconds;
        state.controls.sim_pipeline_seconds = jr.pipeline_seconds;
        state.controls.sim_sequence_timing_details = std::move(jr.sequence_timing_details);
        state.controls.sim_proxy_timing_details = std::move(jr.proxy_timing_details);
        state.controls.sim_solver_timing_details = std::move(jr.solver_timing_details);

        if (was_cancelled)
        {
            state.controls.current_frame = std::max(0, state.controls.frame_count - 1);
            state.controls.playing = false;
        }
        else
        {
            state.controls.current_frame = 0;
            state.controls.playing = m_cloth_sim_play_on_finish && (state.controls.frame_count > 1);
        }

        console_add_line(state,
            std::string(was_cancelled ? "Cloth simulation stopped: " : "Cloth simulation ready: ")
                + std::to_string(state.controls.frame_count) + " frames");
        state.controls.sim_stage = was_cancelled ? "Stopped" : "Ready";

        if (!jr.timing_report.empty())
        {
            std::istringstream iss(jr.timing_report);
            std::string line;
            while (std::getline(iss, line))
            {
                if (!line.empty())
                    console_add_line(state, line);
            }
        }
    }

    void Renderer::apply_cloth_simulation_frame(AppState& state)
    {
        if (m_garment_mesh_name.empty())
            return;
        if (m_cloth_sim_job_running)
        {
            if (!state.show_garment_mesh || !state.controls.preview || !m_cloth_sim_preview_state)
                return;

            const bool pose_changed =
                m_cloth_sim_require_pose_version_match && m_cloth_sim_pose_version != m_pose_version;
            const bool garment_changed = m_cloth_sim_garment_version != m_garment_version;
            if (pose_changed || garment_changed)
                return;

            const std::uint64_t seq = m_cloth_sim_preview_state->seq.load(std::memory_order_acquire);
            if (seq == m_cloth_sim_preview_last_seq_applied)
                return;

            auto& garment = m_viewer.data(m_garment_mesh_index);
            auto& body = m_viewer.data(m_smplh_mesh_index);

            Eigen::MatrixXd garment_V;
            bool want_body_update = false;
            Eigen::MatrixXd body_V;
            Eigen::MatrixXi body_F;
            bool body_topology_valid = false;
            {
                std::lock_guard<std::mutex> lk(m_cloth_sim_preview_state->mutex);
                if (!m_cloth_sim_preview_state->valid)
                    return;
                if (m_cloth_sim_preview_state->V.rows() != garment.V.rows())
                    return;
                garment_V = m_cloth_sim_preview_state->V;

                if (m_cloth_sim_preview_state->body_valid)
                {
                    want_body_update = true;
                    body_V = m_cloth_sim_preview_state->body_V;
                    body_topology_valid = m_cloth_sim_preview_state->body_topology_valid;
                    if (body_topology_valid)
                        body_F = m_cloth_sim_preview_state->body_F;
                }
                m_cloth_sim_preview_last_seq_applied = seq;
            }
            garment.set_vertices(garment_V);
            garment.compute_normals();

            if (want_body_update)
            {
                if (body_topology_valid
                    && (!m_smplh_sim_preview_mesh_active
                        || body.V.rows() != body_V.rows()
                        || body.F.rows() != body_F.rows()))
                {
                    body.clear();
                    body.set_mesh(body_V, body_F);
                    body.show_faces = 1;
                    body.show_lines = 0;
                    body.show_overlay = 0;
                    body.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
                    m_smplh_sim_preview_mesh_active = true;
                }
                else if (body.V.rows() == body_V.rows())
                {
                    body.set_vertices(body_V);
                }
                body.compute_normals();
                body.set_visible(true);
            }
            return;
        }

        if (m_cloth_sim_frames.empty())
            return;

        const bool pose_changed =
            m_cloth_sim_require_pose_version_match && m_cloth_sim_pose_version != m_pose_version;
        const bool garment_changed = m_cloth_sim_garment_version != m_garment_version;
        if (pose_changed || garment_changed)
        {
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        const int frame_count = (int)m_cloth_sim_frames.size();
        state.controls.frame_count = frame_count;
        const int fi = clamp_frame(state.controls.current_frame, frame_count);

        if (fi < 0 || fi >= frame_count)
            return;
        if (m_cloth_sim_frames[(size_t)fi].rows() != garment.V.rows())
        {
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            return;
        }

        if (state.show_garment_mesh)
        {
            garment.set_vertices(m_cloth_sim_frames[(size_t)fi]);
            garment.compute_normals();
        }

        const bool want_proxy_sequence =
            state.controls.save_smpl_sequence
            && state.controls.show_collision_proxy_sequence
            && !m_collision_proxy_sim_frames.empty()
            && m_collision_proxy_sim_F.rows() > 0;

        if (want_proxy_sequence)
        {
            const int body_fi = std::min(fi, (int)m_collision_proxy_sim_frames.size() - 1);
            const Eigen::MatrixXd& body_V = m_collision_proxy_sim_frames[(size_t)body_fi];
            const Eigen::MatrixXi& body_F = m_collision_proxy_sim_F;
            auto& body = m_viewer.data(m_smplh_mesh_index);

            if (body_V.rows() > 0 && body_F.rows() > 0)
            {
                if (!m_smplh_sim_preview_mesh_active
                    || body.V.rows() != body_V.rows()
                    || body.F.rows() != body_F.rows())
                {
                    body.clear();
                    body.set_mesh(body_V, body_F);
                    body.show_faces = 1;
                    body.show_lines = 0;
                    body.show_overlay = 0;
                    body.set_colors(Eigen::RowVector3d(0.72, 0.80, 0.92));
                    m_smplh_sim_preview_mesh_active = true;
                }
                else
                {
                    body.set_vertices(body_V);
                }
                body.compute_normals();
                body.set_visible(true);
            }
        }
        else if (!m_smpl_sim_frames.empty())
        {
            const int body_fi = std::min(fi, (int)m_smpl_sim_frames.size() - 1);
            const Eigen::MatrixXd& body_V = m_smpl_sim_frames[(size_t)body_fi];
            auto& body = m_viewer.data(m_smplh_mesh_index);

            if (body_V.rows() == m_smplh_model.v_template.rows())
            {
                if (m_smplh_sim_preview_mesh_active
                    || body.V.rows() != body_V.rows()
                    || body.F.rows() != m_smplh_model.F.rows())
                {
                    Eigen::MatrixXi body_F = m_smplh_model.F;
                    if (m_smplh_loaded_pose_mirror_x && body_F.cols() >= 3)
                        body_F.col(1).swap(body_F.col(2));
                    body.clear();
                    body.set_mesh(body_V, body_F);
                    body.show_faces = 1;
                    body.show_lines = 0;
                    body.show_overlay = 0;
                    body.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
                    m_smplh_sim_preview_mesh_active = false;
                }
                else
                {
                    body.set_vertices(body_V);
                }
                body.compute_normals();
                body.set_visible(true);
            }
        }
    }
}
