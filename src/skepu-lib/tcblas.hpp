#pragma once

#include <iostream>
#include <iomanip>
#include <skepu>
#include <skepu-lib/blas.hpp>
#include <cstring>

// #ifdef SKEPU_CUDA

// #ifndef SKEPU_PRECOMPILED

// static PrecompilerMarker startOfTcblasHPP;

#ifdef SKEPU_CUDA
#include <cuda.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
// #include <cutlass/gemm/device/gemv.h>
// #include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#endif // SKEPU_CUDA

#define gpuID 0
#define streamID 0
#define usePitch false
#define markOnlyLocalCopiesInvalid true



namespace skepu {

namespace tcblas {

#ifdef SKEPU_CUDA
using CutlassRowMajor = cutlass::layout::RowMajor;
using CutlassColMajor = cutlass::layout::ColumnMajor;
using CutlassMMAOp = cutlass::arch::OpClassTensorOp;
using CutlassArch = cutlass::arch::Sm80;
using CutlassShapeThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
using CutlassShapeWarp = cutlass::gemm::GemmShape<64, 64, 16>;
using CutlassShapeOp = cutlass::gemm::GemmShape<16, 8, 8>;
using CutlassEpilogue = cutlass::epilogue::thread::LinearCombination<
	float,
	128 / cutlass::sizeof_bits<float>::value,
	float,
	float>;
using CutlassThreadBlockSchedule = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
// Number of pipelines you want to use
constexpr int CutlassNumStages = 3;
// Alignment
constexpr int CutlassAlignment = 4;
#endif // SKEPU_CUDA


// Uses Tensor Cores through cutlass
template<typename TA, typename TB, typename TC>
void gemm_tc_cutlass_fastF32(
	blas::Op                         transA,
	blas::Op                         transB,
	blas::size_type                  m,
	blas::size_type                  n,
	blas::size_type                  k,
	blas::scalar_type<TA, TB, TC>    alpha,
	Matrix<TA> BLAS_CONST&           A,
	blas::size_type                  lda,
	Matrix<TB> BLAS_CONST&           B,
	blas::size_type                  ldb,
	blas::scalar_type<TA, TB, TC>    beta,
	Matrix<TC>&                      C,
	blas::size_type                  ldc
)
{
	typedef blas::scalar_type<TA, TB, TC> scalar_t;

  // constants
  const scalar_t zero = 0;
  const scalar_t one  = 1;

	// quick return
	if (m == 0 || n == 0 || k == 0)
		return;

	#ifdef SKEPU_CUDA
	if (transA == blas::Op::NoTrans)
	{
		if (transB == blas::Op::NoTrans)
		{
			using cutlass_gemm = cutlass::gemm::device::Gemm<
			float,
			CutlassRowMajor,
			float,
			CutlassRowMajor,
			float,
			CutlassRowMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
			>;

			AccessMode readMode = AccessMode::Read;
			AccessMode readWriteMode = AccessMode::ReadWrite;
			typename backend::DeviceMemPointer_CU<TA>* device_pointer_a = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TB>* device_pointer_b = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TC>* device_pointer_c = C.updateDevice_CU(C.data(), C.total_rows(), C.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

			typename cutlass_gemm::Arguments cutlass_args{
				{int(m), int(n), int(k)},
				{device_pointer_a->getDeviceDataPointer(), k},
				{device_pointer_b->getDeviceDataPointer(), n},
				{device_pointer_c->getDeviceDataPointer(), n},
				{device_pointer_c->getDeviceDataPointer(), n},
				{alpha, beta},
				1
			};

			size_t my_workspace_size = cutlass_gemm::get_workspace_size(cutlass_args);

			cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

			cutlass_gemm gemm_op;

			cutlass::Status my_status = gemm_op.can_implement(cutlass_args);
			my_status = gemm_op.initialize(cutlass_args, my_workspace.get());
			my_status = gemm_op();

			device_pointer_c->changeDeviceData(true);
		}
		else if (transB == blas::Op::Trans)
		{
			using cutlass_gemm = cutlass::gemm::device::Gemm<
			float,
			CutlassRowMajor,
			float,
			CutlassColMajor,
			float,
			CutlassRowMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
			>;

			AccessMode readMode = AccessMode::Read;
			AccessMode readWriteMode = AccessMode::ReadWrite;
			typename backend::DeviceMemPointer_CU<TA>* device_pointer_a = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TB>* device_pointer_b = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TC>* device_pointer_c = C.updateDevice_CU(C.data(), C.total_rows(), C.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

			typename cutlass_gemm::Arguments cutlass_args{
				{int(m), int(n), int(k)},
				{device_pointer_a->getDeviceDataPointer(), k},
				{device_pointer_b->getDeviceDataPointer(), k},
				{device_pointer_c->getDeviceDataPointer(), n},
				{device_pointer_c->getDeviceDataPointer(), n},
				{alpha, beta},
				1
			};

			size_t my_workspace_size = cutlass_gemm::get_workspace_size(cutlass_args);

			cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

			cutlass_gemm gemm_op;

			cutlass::Status my_status = gemm_op.can_implement(cutlass_args);
			my_status = gemm_op.initialize(cutlass_args, my_workspace.get());
			my_status = gemm_op();

			device_pointer_c->changeDeviceData(true);
		}
	}
	else if (transA == blas::Op::Trans)
	{
		if (transB == blas::Op::NoTrans)
		{
			using cutlass_gemm = cutlass::gemm::device::Gemm<
			float,
			CutlassColMajor,
			float,
			CutlassRowMajor,
			float,
			CutlassRowMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
			>;

			AccessMode readMode = AccessMode::Read;
			AccessMode readWriteMode = AccessMode::ReadWrite;
			typename backend::DeviceMemPointer_CU<TA>* device_pointer_a = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TB>* device_pointer_b = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TC>* device_pointer_c = C.updateDevice_CU(C.data(), C.total_rows(), C.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

			typename cutlass_gemm::Arguments cutlass_args{
				{int(m), int(n), int(k)},
				{device_pointer_a->getDeviceDataPointer(), m},
				{device_pointer_b->getDeviceDataPointer(), n},
				{device_pointer_c->getDeviceDataPointer(), n},
				{device_pointer_c->getDeviceDataPointer(), n},
				{alpha, beta},
				1
			};

			size_t my_workspace_size = cutlass_gemm::get_workspace_size(cutlass_args);

			cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

			cutlass_gemm gemm_op;

			cutlass::Status my_status = gemm_op.can_implement(cutlass_args);
			my_status = gemm_op.initialize(cutlass_args, my_workspace.get());
			my_status = gemm_op();

			device_pointer_c->changeDeviceData(true);
		}
		else if (transB == blas::Op::Trans)
		{
			using cutlass_gemm = cutlass::gemm::device::Gemm<
			float,
			CutlassColMajor,
			float,
			CutlassColMajor,
			float,
			CutlassRowMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
			>;

			AccessMode readMode = AccessMode::Read;
			AccessMode readWriteMode = AccessMode::ReadWrite;
			typename backend::DeviceMemPointer_CU<TA>* device_pointer_a = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TB>* device_pointer_b = B.updateDevice_CU(B.data(), B.total_rows(), B.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
			typename backend::DeviceMemPointer_CU<TC>* device_pointer_c = C.updateDevice_CU(C.data(), C.total_rows(), C.total_cols(), gpuID, streamID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

			typename cutlass_gemm::Arguments cutlass_args{
				{int(m), int(n), int(k)},
				{device_pointer_a->getDeviceDataPointer(), m},
				{device_pointer_b->getDeviceDataPointer(), k},
				{device_pointer_c->getDeviceDataPointer(), n},
				{device_pointer_c->getDeviceDataPointer(), n},
				{alpha, beta},
				1
			};

			size_t my_workspace_size = cutlass_gemm::get_workspace_size(cutlass_args);

			cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

			cutlass_gemm gemm_op;

			cutlass::Status my_status = gemm_op.can_implement(cutlass_args);
			my_status = gemm_op.initialize(cutlass_args, my_workspace.get());
			my_status = gemm_op();

			device_pointer_c->changeDeviceData(true);
		}
	}

	#endif // SKEPU_CUDA
}



template<typename TA, typename TB, typename TC>
void gemm_tc_cutlass_fastF32(
	blas::Op                         transA,
	blas::Op                         transB,
	blas::size_type                  m,
	blas::size_type                  n,
	blas::size_type                  k,
	blas::scalar_type<TA, TB, TC>    alpha,
	TA const*                        device_pointer_A,
	blas::size_type                  lda,
	TB const*                        device_pointer_B,
	blas::size_type                  ldb,
	blas::scalar_type<TA, TB, TC>    beta,
	TC*                              device_pointer_C,
	blas::size_type                  ldc
)
{
#ifdef SKEPU_CUDA
	using cutlass_gemm = cutlass::gemm::device::Gemm<
	float,
	CutlassRowMajor,
	float,
	CutlassRowMajor,
	float,
	CutlassRowMajor,
	float,
	CutlassMMAOp,
	CutlassArch,
	CutlassShapeThreadBlock,
	CutlassShapeWarp,
	CutlassShapeOp,
	CutlassEpilogue,
	CutlassThreadBlockSchedule,
	CutlassNumStages,
	CutlassAlignment,
	CutlassAlignment,
	false, // If true, kernel supports split-K with serial reduction
	cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
	>;

	typename cutlass_gemm::Arguments cutlass_args{
		{int(m), int(n), int(k)},
		{device_pointer_A, k},
		{device_pointer_B, n},
		{device_pointer_C, n},
		{device_pointer_C, n},
		{alpha, beta},
		1
	};

	size_t my_workspace_size = cutlass_gemm::get_workspace_size(cutlass_args);

	cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

	cutlass_gemm gemm_op;

	cutlass::Status my_status = gemm_op.can_implement(cutlass_args);
	my_status = gemm_op.initialize(cutlass_args, my_workspace.get());
	my_status = gemm_op();
#endif // SKEPU_CUDA
}



template<typename TA, typename TX, typename TY, typename TS = blas::scalar_type<TA, TX, TY>>
void gemv_tc_cutlass_fastF32(
	blas::Op 	              trans,
	blas::size_type           m,
	blas::size_type           n,
	TS                        alpha,
	Matrix<TA> BLAS_CONST&    A,
	blas::size_type           lda,
	Vector<TX> BLAS_CONST&    x,
	blas::stride_type         incx,
	TS                        beta,
	Vector<TY> &              y,
	blas::stride_type         incy
)
{
	const TS zero = 0, one  = 1;

	if (m == 0 || n == 0 || (alpha == zero && beta == one))
    return;

#ifdef SKEPU_CUDA
	if (trans == blas::Op::NoTrans)
	{
		using cutlass_gemv = cutlass::gemm::device::Gemm<
			float,
			CutlassRowMajor,
			float,
			CutlassColMajor,
			float,
			CutlassColMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
		>;
		AccessMode readMode = AccessMode::Read;
		AccessMode readWriteMode = AccessMode::ReadWrite;
		typename backend::DeviceMemPointer_CU<TA>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
		typename backend::DeviceMemPointer_CU<TX>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
		typename backend::DeviceMemPointer_CU<TY>* device_pointer_y = y.updateDevice_CU(y.data(), y.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

		typename cutlass_gemv::Arguments cutlass_args{
			{int(m), int(1), int(n)},
			{device_pointer_A->getDeviceDataPointer(), n},
			{device_pointer_x->getDeviceDataPointer(), 1},
			{device_pointer_y->getDeviceDataPointer(), 1},
			{device_pointer_y->getDeviceDataPointer(), 1},
			{alpha, beta},
			1
		};

		size_t my_workspace_size = cutlass_gemv::get_workspace_size(cutlass_args);

		cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

		cutlass_gemv gemv_op;

		cutlass::Status my_status = gemv_op.can_implement(cutlass_args);
		my_status = gemv_op.initialize(cutlass_args, my_workspace.get());
		my_status = gemv_op();

		device_pointer_y->changeDeviceData(true);
	} else if (trans == blas::Op::Trans) {

		using cutlass_gemv = cutlass::gemm::device::Gemm<
			float,
			CutlassColMajor,
			float,
			CutlassColMajor,
			float,
			CutlassColMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
		>;
		AccessMode readMode = AccessMode::Read;
		AccessMode readWriteMode = AccessMode::ReadWrite;
		typename backend::DeviceMemPointer_CU<TA>* device_pointer_A = A.updateDevice_CU(A.data(), A.total_rows(), A.total_cols(), gpuID, streamID, readMode, usePitch, markOnlyLocalCopiesInvalid);
		typename backend::DeviceMemPointer_CU<TX>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
		typename backend::DeviceMemPointer_CU<TY>* device_pointer_y = y.updateDevice_CU(y.data(), y.size(), gpuID, readWriteMode, usePitch, markOnlyLocalCopiesInvalid);

		typename cutlass_gemv::Arguments cutlass_args{
			{int(n), int(1), int(m)},
			{device_pointer_A->getDeviceDataPointer(), n},
			{device_pointer_x->getDeviceDataPointer(), 1},
			{device_pointer_y->getDeviceDataPointer(), 1},
			{device_pointer_y->getDeviceDataPointer(), 1},
			{alpha, beta},
			1
		};

		size_t my_workspace_size = cutlass_gemv::get_workspace_size(cutlass_args);

		cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

		cutlass_gemv gemv_op;

		cutlass::Status my_status = gemv_op.can_implement(cutlass_args);
		my_status = gemv_op.initialize(cutlass_args, my_workspace.get());
		my_status = gemv_op();

		device_pointer_y->changeDeviceData(true);
	}
#endif // SKEPU_CUDA
}


template<typename TA, typename TX, typename TY, typename TS = blas::scalar_type<TA, TX, TY>>
void gemv_tc_cutlass_fastF32(
	blas::Op 	                      trans,
	blas::size_type                   m,
	blas::size_type                   n,
	TS                                alpha,
	TA const*                         device_pointer_A,
	blas::size_type                   lda,
	TX const*                         device_pointer_x,
	blas::stride_type                 incx,
	TS                                beta,
	TY*                               device_pointer_y,
	blas::stride_type                 incy
)
{
	const TS zero = 0, one  = 1;

	if (m == 0 || n == 0 || (alpha == zero && beta == one))
    return;

#ifdef SKEPU_CUDA
	if (trans == blas::Op::NoTrans)
	{
		using cutlass_gemv = cutlass::gemm::device::Gemm<
			float,
			CutlassRowMajor,
			float,
			CutlassColMajor,
			float,
			CutlassColMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
		>;

		typename cutlass_gemv::Arguments cutlass_args{
			{int(m), int(1), int(n)},
			{device_pointer_A, n},
			{device_pointer_x, 1},
			{device_pointer_y, 1},
			{device_pointer_y, 1},
			{alpha, beta},
			1
		};

		size_t my_workspace_size = cutlass_gemv::get_workspace_size(cutlass_args);

		cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

		cutlass_gemv gemv_op;

		cutlass::Status my_status = gemv_op.can_implement(cutlass_args);
		my_status = gemv_op.initialize(cutlass_args, my_workspace.get());
		my_status = gemv_op();
	} else if (trans == blas::Op::Trans) {

		using cutlass_gemv = cutlass::gemm::device::Gemm<
			float,
			CutlassColMajor,
			float,
			CutlassColMajor,
			float,
			CutlassColMajor,
			float,
			CutlassMMAOp,
			CutlassArch,
			CutlassShapeThreadBlock,
			CutlassShapeWarp,
			CutlassShapeOp,
			CutlassEpilogue,
			CutlassThreadBlockSchedule,
			CutlassNumStages,
			CutlassAlignment,
			CutlassAlignment,
			false, // If true, kernel supports split-K with serial reduction
			cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
		>;

		typename cutlass_gemv::Arguments cutlass_args{
			{int(n), int(1), int(m)},
			{device_pointer_A, n},
			{device_pointer_x, 1},
			{device_pointer_y, 1},
			{device_pointer_y, 1},
			{alpha, beta},
			1
		};

		size_t my_workspace_size = cutlass_gemv::get_workspace_size(cutlass_args);

		cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

		cutlass_gemv gemv_op;

		cutlass::Status my_status = gemv_op.can_implement(cutlass_args);
		my_status = gemv_op.initialize(cutlass_args, my_workspace.get());
		my_status = gemv_op();
	}
#endif // SKEPU_CUDA
}


template<typename TX, typename TY>
blas::scalar_type<TX, TY> dot_tc_cutlass_fastF32(
	blas::size_type                  n,
	Vector<TX> BLAS_CONST&           x,
	blas::stride_type                incx,
	Vector<TY> BLAS_CONST&           y,
	blas::stride_type                incy
)
{
	blas::scalar_type<TX, TY> result = 0;
#ifdef SKEPU_CUDA
	using cutlass_gemv = cutlass::gemm::device::Gemm<
		float,
		CutlassRowMajor,
		float,
		CutlassColMajor,
		float,
		CutlassColMajor,
		float,
		CutlassMMAOp,
		CutlassArch,
		CutlassShapeThreadBlock,
		CutlassShapeWarp,
		CutlassShapeOp,
		CutlassEpilogue,
		CutlassThreadBlockSchedule,
		CutlassNumStages,
		CutlassAlignment,
		CutlassAlignment,
		false, // If true, kernel supports split-K with serial reduction
		cutlass::arch::OpMultiplyAddFastF32 // <- uses multiple tf32 calculations to perform fp32
	>;
	AccessMode readMode = AccessMode::Read;
	typename backend::DeviceMemPointer_CU<TX>* device_pointer_x = x.updateDevice_CU(x.data(), x.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
	typename backend::DeviceMemPointer_CU<TY>* device_pointer_y = y.updateDevice_CU(y.data(), y.size(), gpuID, readMode, usePitch, markOnlyLocalCopiesInvalid);
	TY* device_pointer_result = nullptr;
	cudaMalloc(reinterpret_cast<void **>(&device_pointer_result), sizeof(TY));
	TY* host_pointer_result = (TY *)malloc(sizeof(TY));

	typename cutlass_gemv::Arguments cutlass_args{
		{int(1), int(1), int(n)},
		{device_pointer_x->getDeviceDataPointer(), n},
		{device_pointer_y->getDeviceDataPointer(), 1},
		{device_pointer_result, 1},
		{device_pointer_result, 1},
		{1, 0},
		1
	};

	size_t my_workspace_size = cutlass_gemv::get_workspace_size(cutlass_args);

	cutlass::device_memory::allocation<uint8_t> my_workspace(my_workspace_size);

	cutlass_gemv gemv_op;

	cutlass::Status my_status = gemv_op.can_implement(cutlass_args);
	my_status = gemv_op.initialize(cutlass_args, my_workspace.get());
	my_status = gemv_op();

  	cudaMemcpy(host_pointer_result, device_pointer_result, sizeof(TY), cudaMemcpyDeviceToHost);
	result = *host_pointer_result;
	free(host_pointer_result);
	cudaFree(device_pointer_result);
#endif // SKEPU_CUDA
	return result;
}

}
}

// static PrecompilerMarker endOfTcblasHPP;

// #endif // SKEPU_PRECOMPILED

// #endif // SKEPU_CUDA



// Copied from cutlass example program

// 	using my_gemm = cutlass::gemm::device::Gemm<
// 	float,
// 	RowMajor,
// 	float,
// 	RowMajor,
// 	float,
// 	RowMajor,
// 	float,
// 	cutlass::arch::OpClassTensorOp,
// 	cutlass::arch::Sm80,
// 	cutlass::gemm::GemmShape<128, 128, 16>,
// 	cutlass::gemm::GemmShape<64, 64, 16>,
// 	cutlass::gemm::GemmShape<16, 8, 8>,
// 	cutlass::epilogue::thread::LinearCombination<
// 	  float,
// 	  128 / cutlass::sizeof_bits<float>::value,
// 	  float,
// 	  float>,
// 	cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
// 	3,
// 	4,
// 	4,
// 	false
// 	// ,cutlass::arch::OpMultiplyAddFastF32
// 	,cutlass::arch::OpMultiplyAdd
//   >;