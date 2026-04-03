#pragma once

#include <Eigen/Core>
#include <optional>
#include <filesystem>
#include <string>

namespace poseweaver
{
	using MatrixX3dRM = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
	using MatrixXdrm = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatrixXirm = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatrixXllrm = Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	struct SmplhModel
	{
		// Topology
		MatrixXirm F; // (#F x 3)

		// Base model
		MatrixX3dRM v_template; // (#V x 3)
		MatrixXdrm shapedirs;   // (#V*3 x #betas)
		MatrixXdrm posedirs;    // (#V*3 x (#J-1)*9)
		MatrixXdrm J_regressor; // (#J x #V)
		MatrixXdrm weights;     // (#V x #J)
		MatrixXllrm kintree_table; // (2 x #J)

		int vertex_count() const { return (int)v_template.rows(); }
		int face_count() const { return (int)F.rows(); }
		int joint_count() const { return (int)weights.cols(); }
		int beta_count() const { return (int)shapedirs.cols(); }

		bool load_from_npz(const std::filesystem::path& model_npz_path, std::string* error_message = nullptr);
	};

	struct SmplhParams
	{
		Eigen::VectorXd betas;
		Eigen::VectorXd pose_axis_angle; //(3 * #J)
	};

	bool smplh_evaluate_lbs(
		const SmplhModel& model,
		const SmplhParams& params,
		MatrixX3dRM& V_out,
		std::string* error_message = nullptr);

	bool smplh_load_pose_npz(
		const std::filesystem::path& pose_npz_path,
		Eigen::VectorXd& pose_axis_angle_out,
		std::optional<Eigen::VectorXd>* betas_out = nullptr,
		std::string* error_message = nullptr);
}
