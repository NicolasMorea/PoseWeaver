#pragma once

#include <Eigen/Core>
#include <string>

namespace poseweaver
{
    struct ClothToSmplMapping
    {
        Eigen::VectorXi face_index;     // (#V_cloth)
        Eigen::MatrixXd barycentric;    // (#V_cloth x 3)
        Eigen::MatrixXd offset;         // (#V_cloth x 3)
        Eigen::MatrixXd offset_local;   // (#V_cloth x 3)

        bool valid() const
        {
            return face_index.size() > 0 &&
                   barycentric.rows() == face_index.size() && barycentric.cols() == 3 &&
                   offset.rows() == face_index.size() && offset.cols() == 3 &&
                   offset_local.rows() == face_index.size() && offset_local.cols() == 3;
        }

        void clear()
        {
            face_index.resize(0);
            barycentric.resize(0, 0);
            offset.resize(0, 0);
			offset_local.resize(0, 0);
        }
    };

    // Computes a mapping from cloth vertices to the closest point on the SMPL surface.
    bool compute_cloth_to_smpl_mapping(
        const Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& smpl_V,
        const Eigen::MatrixXi& smpl_F,
        ClothToSmplMapping* out,
        std::string* error);
}
