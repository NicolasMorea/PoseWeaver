#include "smpl/Smplh.h"

#include <cnpy.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <optional>
#include <sstream>
#include <vector>

namespace poseweaver
{
	namespace
	{
		static void set_error(std::string* out, const std::string& msg)
		{
			if (out)
				*out = msg;
		}

		static bool expect_shape(
			const cnpy::NpyArray& a,
			std::initializer_list<size_t> dims,
			std::string* error_message,
			const char* name)
		{
			if (a.shape.size() != dims.size())
			{
				std::ostringstream oss;
				oss << "SMPLH npz array '" << name << "' has rank " << a.shape.size() << ", expected " << dims.size();
				set_error(error_message, oss.str());
				return false;
			}

			size_t i = 0;
			for (const auto d : dims)
			{
				if (d != (size_t)-1 && a.shape[i] != d)
				{
					std::ostringstream oss;
					oss << "SMPLH npz array '" << name << "' has shape mismatch at dim " << i << ": got " << a.shape[i] << ", expected " << d;
					set_error(error_message, oss.str());
					return false;
				}
				++i;
			}

			if (a.fortran_order)
			{
				std::ostringstream oss;
				oss << "SMPLH npz array '" << name << "' is Fortran-ordered; this loader expects C-order";
				set_error(error_message, oss.str());
				return false;
			}

			return true;
		}

		template <typename Scalar>
		static bool expect_dtype(const cnpy::NpyArray& a, std::string* error_message, const char* name)
		{
			if (a.word_size != sizeof(Scalar))
			{
				std::ostringstream oss;
				oss << "SMPLH npz array '" << name << "' has word_size " << a.word_size << ", expected " << sizeof(Scalar);
				set_error(error_message, oss.str());
				return false;
			}
			return true;
		}

		template <typename Scalar>
		static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> copy_2d_rm(
			const cnpy::NpyArray& a)
		{
			const int rows = (int)a.shape[0];
			const int cols = (int)a.shape[1];
			Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(rows, cols);
			const Scalar* src = a.data<Scalar>();
			std::copy(src, src + (size_t)rows * (size_t)cols, out.data());
			return out;
		}

		static Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> copy_faces_to_int_rm(
			const cnpy::NpyArray& a)
		{
			const int rows = (int)a.shape[0];
			const int cols = (int)a.shape[1];
			Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(rows, cols);
			if (a.word_size == sizeof(uint32_t))
			{
				const uint32_t* src = a.data<uint32_t>();
				for (int i = 0; i < rows * cols; ++i)
					out.data()[i] = (int)src[i];
				return out;
			}
			if (a.word_size == sizeof(int32_t))
			{
				const int32_t* src = a.data<int32_t>();
				for (int i = 0; i < rows * cols; ++i)
					out.data()[i] = (int)src[i];
				return out;
			}
			if (a.word_size == sizeof(int64_t))
			{
				const int64_t* src = a.data<int64_t>();
				for (int i = 0; i < rows * cols; ++i)
					out.data()[i] = (int)src[i];
				return out;
			}
			return out;
		}

		static Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> copy_int64_2d_rm(
			const cnpy::NpyArray& a)
		{
			const int rows = (int)a.shape[0];
			const int cols = (int)a.shape[1];
			Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(rows, cols);
			const int64_t* src = a.data<int64_t>();
			for (int i = 0; i < rows * cols; ++i)
				out.data()[i] = (long long)src[i];
			return out;
		}

		static Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flatten_3d_to_2d_rm(
			const cnpy::NpyArray& a)
		{
			const int d0 = (int)a.shape[0];
			const int d1 = (int)a.shape[1];
			const int d2 = (int)a.shape[2];
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(d0 * d1, d2);
			const double* src = a.data<double>();
			std::copy(src, src + (size_t)d0 * (size_t)d1 * (size_t)d2, out.data());
			return out;
		}

		static Eigen::Matrix3d rodrigues(const Eigen::Vector3d& aa)
		{
			const double theta = aa.norm();
			if (theta < 1e-12)
				return Eigen::Matrix3d::Identity();
			return Eigen::AngleAxisd(theta, aa / theta).toRotationMatrix();
		}

		static bool copy_pose_array_to_axis_angle(
			const cnpy::NpyArray& poses,
			Eigen::VectorXd& pose_axis_angle_out,
			std::string* error_message)
		{
			if (poses.word_size != sizeof(double))
			{
				set_error(error_message, "Pose npz array 'poses' must be float64 (double)");
				return false;
			}

			// Supported shapes:
			//  - (1, 3J)
			//  - (3J,)
			//  - (J, 3)
			const double* src = poses.data<double>();
			if (poses.shape.size() == 2 && poses.shape[0] == 1)
			{
				const size_t n = poses.shape[1];
				pose_axis_angle_out.resize((int)n);
				std::copy(src, src + n, pose_axis_angle_out.data());
				return true;
			}
			if (poses.shape.size() == 1)
			{
				const size_t n = poses.shape[0];
				pose_axis_angle_out.resize((int)n);
				std::copy(src, src + n, pose_axis_angle_out.data());
				return true;
			}
			if (poses.shape.size() == 2 && poses.shape[1] == 3)
			{
				const size_t j = poses.shape[0];
				pose_axis_angle_out.resize((int)(j * 3));
				if (!poses.fortran_order)
				{
					std::copy(src, src + j * 3, pose_axis_angle_out.data());
				}
				else
				{
					for (size_t r = 0; r < j; ++r)
					{
						pose_axis_angle_out[(int)(r * 3 + 0)] = src[0 * j + r];
						pose_axis_angle_out[(int)(r * 3 + 1)] = src[1 * j + r];
						pose_axis_angle_out[(int)(r * 3 + 2)] = src[2 * j + r];
					}
				}
				return true;
			}

			std::ostringstream oss;
			oss << "Pose npz array 'poses' has unsupported shape (";
			for (size_t i = 0; i < poses.shape.size(); ++i)
			{
				if (i) oss << ", ";
				oss << poses.shape[i];
			}
			oss << ")";
			set_error(error_message, oss.str());
			return false;
		}
	}

	bool SmplhModel::load_from_npz(const std::filesystem::path& model_npz_path, std::string* error_message)
	{
		SmplhModel tmp;
		cnpy::npz_t npz;
		try
		{
			npz = cnpy::npz_load(model_npz_path.string());
		}
		catch (const std::exception& e)
		{
			std::ostringstream oss;
			oss << "Failed to load SMPLH model npz: " << model_npz_path.string() << " (" << e.what() << ")";
			set_error(error_message, oss.str());
			return false;
		}

		auto must_get = [&](const char* key) -> const cnpy::NpyArray*
		{
			auto it = npz.find(key);
			if (it == npz.end())
			{
				std::ostringstream oss;
				oss << "SMPLH model npz is missing array '" << key << "'";
				set_error(error_message, oss.str());
				return nullptr;
			}
			return &it->second;
		};

		const cnpy::NpyArray* a_f = must_get("f");
		const cnpy::NpyArray* a_v_template = must_get("v_template");
		const cnpy::NpyArray* a_shapedirs = must_get("shapedirs");
		const cnpy::NpyArray* a_posedirs = must_get("posedirs");
		const cnpy::NpyArray* a_J_reg = must_get("J_regressor");
		const cnpy::NpyArray* a_kintree = must_get("kintree_table");
		const cnpy::NpyArray* a_weights = must_get("weights");
		if (!a_f || !a_v_template || !a_shapedirs || !a_posedirs || !a_J_reg || !a_kintree || !a_weights)
			return false;

		if (!expect_shape(*a_v_template, { 6890, 3 }, error_message, "v_template"))
			return false;
		if (!expect_dtype<double>(*a_v_template, error_message, "v_template"))
			return false;
		tmp.v_template.resize((int)a_v_template->shape[0], 3);
		std::copy(a_v_template->data<double>(), a_v_template->data<double>() + (size_t)tmp.v_template.rows() * 3, tmp.v_template.data());

		if (!expect_shape(*a_f, { (size_t)-1, 3 }, error_message, "f"))
			return false;
		tmp.F = copy_faces_to_int_rm(*a_f);

		if (!expect_shape(*a_shapedirs, { (size_t)tmp.vertex_count(), 3, (size_t)-1 }, error_message, "shapedirs"))
			return false;
		if (!expect_dtype<double>(*a_shapedirs, error_message, "shapedirs"))
			return false;
		tmp.shapedirs = flatten_3d_to_2d_rm(*a_shapedirs);

		if (!expect_shape(*a_posedirs, { (size_t)tmp.vertex_count(), 3, (size_t)-1 }, error_message, "posedirs"))
			return false;
		if (!expect_dtype<double>(*a_posedirs, error_message, "posedirs"))
			return false;
		tmp.posedirs = flatten_3d_to_2d_rm(*a_posedirs);

		if (!expect_shape(*a_weights, { (size_t)tmp.vertex_count(), (size_t)-1 }, error_message, "weights"))
			return false;
		if (!expect_dtype<double>(*a_weights, error_message, "weights"))
			return false;
		tmp.weights = copy_2d_rm<double>(*a_weights);

		const int joint_count = (int)tmp.weights.cols();
		if (!expect_shape(*a_J_reg, { (size_t)joint_count, (size_t)tmp.vertex_count() }, error_message, "J_regressor"))
			return false;
		if (!expect_dtype<double>(*a_J_reg, error_message, "J_regressor"))
			return false;
		tmp.J_regressor = copy_2d_rm<double>(*a_J_reg);

		if (!expect_shape(*a_kintree, { 2, (size_t)joint_count }, error_message, "kintree_table"))
			return false;
		if (!expect_dtype<int64_t>(*a_kintree, error_message, "kintree_table"))
			return false;
		tmp.kintree_table = copy_int64_2d_rm(*a_kintree);

		// Validate posedirs columns are (J-1)*9.
		if (tmp.posedirs.cols() != (joint_count - 1) * 9)
		{
			std::ostringstream oss;
			oss << "SMPLH posedirs has " << tmp.posedirs.cols() << " columns, expected (" << joint_count << "-1)*9 = " << (joint_count - 1) * 9;
			set_error(error_message, oss.str());
			return false;
		}

		*this = std::move(tmp);
		return true;
	}

	bool smplh_evaluate_lbs(
		const SmplhModel& model,
		const SmplhParams& params,
		MatrixX3dRM& V_out,
		std::string* error_message)
	{
		const int V = model.vertex_count();
		const int J = model.joint_count();
		const int B = model.beta_count();
		if (V <= 0 || J <= 0)
		{
			set_error(error_message, "SMPLH model is empty (not loaded?)");
			return false;
		}
		if (params.betas.size() != 0 && params.betas.size() != B)
		{
			std::ostringstream oss;
			oss << "SMPLH betas has size " << params.betas.size() << ", expected " << B;
			set_error(error_message, oss.str());
			return false;
		}
		if (params.pose_axis_angle.size() != 0 && params.pose_axis_angle.size() != 3 * J)
		{
			std::ostringstream oss;
			oss << "SMPLH pose_axis_angle has size " << params.pose_axis_angle.size() << ", expected " << (3 * J);
			set_error(error_message, oss.str());
			return false;
		}

		Eigen::VectorXd betas = (params.betas.size() == B) ? params.betas : Eigen::VectorXd::Zero(B);
		Eigen::VectorXd pose = (params.pose_axis_angle.size() == 3 * J) ? params.pose_axis_angle : Eigen::VectorXd::Zero(3 * J);

		Eigen::Map<const Eigen::VectorXd> v_template_flat(model.v_template.data(), (size_t)V * 3);
		Eigen::VectorXd v_shaped_flat = v_template_flat;
		if (B > 0)
			v_shaped_flat.noalias() += model.shapedirs * betas;

		Eigen::Map<const MatrixX3dRM> v_shaped(v_shaped_flat.data(), V, 3);
		Eigen::MatrixXd joints = model.J_regressor * v_shaped;

		std::vector<Eigen::Matrix3d> R((size_t)J);
		for (int j = 0; j < J; ++j)
		{
			const Eigen::Vector3d aa = pose.segment<3>(3 * j);
			R[(size_t)j] = rodrigues(aa);
		}

		Eigen::VectorXd lrotmin((J - 1) * 9);
		for (int j = 1; j < J; ++j)
		{
			const Eigen::Matrix3d diff = R[(size_t)j] - Eigen::Matrix3d::Identity();
			for (int r = 0; r < 3; ++r)
				for (int c = 0; c < 3; ++c)
					lrotmin[(j - 1) * 9 + r * 3 + c] = diff(r, c);
		}

		Eigen::VectorXd v_posed_flat = v_shaped_flat;
		v_posed_flat.noalias() += model.posedirs * lrotmin;
		Eigen::Map<const MatrixX3dRM> v_posed(v_posed_flat.data(), V, 3);

		// Extract parents from kintree_table.
		Eigen::VectorXi parents(J);
		parents.setConstant(-1);
		{
			if (model.kintree_table.rows() != 2 || model.kintree_table.cols() != J)
			{
				set_error(error_message, "SMPLH kintree_table has unexpected shape");
				return false;
			}
			// Typical SMPL(H) layout is either:
			//   row0: parents, row1: joint ids
			// or the reverse.
			//
			// Some files encode the root parent as -1, while others store it as
			// 2^32-1 (i.e. uint32(-1) widened into int64).
			using RowVectorXll = Eigen::Matrix<long long, 1, Eigen::Dynamic, Eigen::RowMajor>;
			RowVectorXll row0 = model.kintree_table.row(0);
			RowVectorXll row1 = model.kintree_table.row(1);

			auto is_identity_ids_row = [&](const RowVectorXll& r) -> bool
			{
				for (int col = 0; col < J; ++col)
					if (r[col] != col)
						return false;
				return true;
			};

			const bool row0_is_ids = is_identity_ids_row(row0);
			const bool row1_is_ids = is_identity_ids_row(row1);

			const RowVectorXll* ids_row_ptr = nullptr;
			const RowVectorXll* parents_row_ptr = nullptr;
			if (row0_is_ids && !row1_is_ids)
			{
				ids_row_ptr = &row0;
				parents_row_ptr = &row1;
			}
			else if (row1_is_ids && !row0_is_ids)
			{
				ids_row_ptr = &row1;
				parents_row_ptr = &row0;
			}
			else
			{
				// Fall back to detecting a root sentinel.
				const bool row0_has_root = (row0[0] == -1 || row0[0] == 4294967295LL);
				const bool row1_has_root = (row1[0] == -1 || row1[0] == 4294967295LL);
				if (row0_has_root == row1_has_root)
				{
					set_error(error_message, "SMPLH kintree_table ambiguous: cannot determine ids vs parents row");
					return false;
				}
				parents_row_ptr = row0_has_root ? &row0 : &row1;
				ids_row_ptr = row0_has_root ? &row1 : &row0;
			}

			const RowVectorXll& parents_row = *parents_row_ptr;
			const RowVectorXll& ids_row = *ids_row_ptr;

			std::vector<int> id_to_col((size_t)J, -1);
			for (int col = 0; col < J; ++col)
			{
				const long id = ids_row[col];
				if (id < 0 || id >= J)
				{
					set_error(error_message, "SMPLH kintree_table joint ids are out of expected range");
					return false;
				}
				id_to_col[(size_t)id] = col;
			}
			for (int col = 0; col < J; ++col)
			{
				const long pid = parents_row[col];
				if (pid == -1 || pid == 4294967295LL)
				{
					parents[col] = -1;
					continue;
				}
				if (pid < 0 || pid >= J)
				{
					set_error(error_message, "SMPLH kintree_table parent id is out of expected range");
					return false;
				}
				parents[col] = id_to_col[(size_t)pid];
				if (parents[col] < 0)
				{
					set_error(error_message, "SMPLH kintree_table parent id does not map to a joint column");
					return false;
				}
			}
		}

		std::vector<Eigen::Matrix4d> G((size_t)J);
		std::vector<Eigen::Matrix4d> T((size_t)J);
		for (int j = 0; j < J; ++j)
		{
			Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
			A.block<3, 3>(0, 0) = R[(size_t)j];
			if (parents[j] == -1)
			{
				A.block<3, 1>(0, 3) = joints.row(j).transpose();
				G[(size_t)j] = A;
			}
			else
			{
				A.block<3, 1>(0, 3) = (joints.row(j) - joints.row(parents[j])).transpose();
				G[(size_t)j] = G[(size_t)parents[j]] * A;
			}

			Eigen::Matrix4d rest = Eigen::Matrix4d::Identity();
			rest.block<3, 1>(0, 3) = -joints.row(j).transpose();
			T[(size_t)j] = G[(size_t)j] * rest;
		}

		V_out.resize(V, 3);
		for (int i = 0; i < V; ++i)
		{
			Eigen::Vector4d v;
			v << v_posed(i, 0), v_posed(i, 1), v_posed(i, 2), 1.0;
			Eigen::Vector3d out = Eigen::Vector3d::Zero();
			for (int j = 0; j < J; ++j)
			{
				const double w = model.weights(i, j);
				if (w == 0.0)
					continue;
				out.noalias() += w * (T[(size_t)j] * v).head<3>();
			}
			V_out.row(i) = out.transpose();
		}

		return true;
	}

	bool smplh_load_pose_npz(
		const std::filesystem::path& pose_npz_path,
		Eigen::VectorXd& pose_axis_angle_out,
		std::optional<Eigen::VectorXd>* betas_out,
		std::string* error_message)
	{
		cnpy::NpyArray poses;
		try
		{
			poses = cnpy::npz_load(pose_npz_path.string(), "poses");
		}
		catch (const std::exception& e)
		{
			std::ostringstream oss;
			oss << "Failed to load pose npz array 'poses': " << pose_npz_path.string() << " (" << e.what() << ")";
			set_error(error_message, oss.str());
			return false;
		}

		if (!copy_pose_array_to_axis_angle(poses, pose_axis_angle_out, error_message))
			return false;

		// The poses (AMASS) are rotated such that the body points along +Z (blue axis)
		// So we rotate the global/root joint by -90 degrees about +X (red axis) so +Z maps to +Y
		if (pose_axis_angle_out.size() >= 3)
		{
			const Eigen::Vector3d aa_root = pose_axis_angle_out.head<3>();
			const Eigen::Matrix3d R_root = rodrigues(aa_root);
			const double pi = std::acos(-1.0);
			const Eigen::Matrix3d R_fix = Eigen::AngleAxisd(-0.5 * pi, Eigen::Vector3d::UnitX()).toRotationMatrix();
			const Eigen::Matrix3d R_new = R_fix * R_root;

			const Eigen::AngleAxisd aa_new(R_new);
			pose_axis_angle_out.head<3>() = aa_new.axis() * aa_new.angle();
		}

		if (betas_out)
		{
			try
			{
				cnpy::NpyArray b = cnpy::npz_load(pose_npz_path.string(), "betas");
				if (b.word_size != sizeof(double) || b.shape.size() != 1)
				{
					set_error(error_message, "Pose npz array 'betas' must be 1D float64");
					return false;
				}
				Eigen::VectorXd betas((int)b.shape[0]);
				std::copy(b.data<double>(), b.data<double>() + b.shape[0], betas.data());
				*betas_out = std::move(betas);
			}
			catch (const std::exception&)
			{
				*betas_out = std::nullopt;
			}
		}

		return true;
	}
}
